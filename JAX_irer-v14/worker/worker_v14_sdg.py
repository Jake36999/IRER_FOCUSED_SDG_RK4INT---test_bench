from pydantic import BaseModel, ValidationError
# Define the Schema
class JobSchema(BaseModel):
    job_id: str
    params: dict
    grid_size: int = 32
    total_steps: int = 100
    config_hash: str | None = None

class SDGParams(BaseModel):
    dt: float
    epsilon: float
    alpha: float
    sigma_k: float
    c1: float
    c3: float
    splash_fraction: float
    dx: float


import os
import json
import time
import redis
import h5py
import io
import jax
import jax.numpy as jnp
from minio import Minio
from ir_physics import models, solver


class SDGWorker:
        def compute_config_hash(self, params):
            import hashlib
            s = json.dumps(params, sort_keys=True)
            return hashlib.sha256(s.encode()).hexdigest()
    def __init__(self):
        self.redis_host = os.getenv("REDIS_HOST", "localhost")
        self.minio_endpoint = os.getenv("MINIO_ENDPOINT", "localhost:9000")
        self.minio_user = os.getenv("MINIO_ACCESS_KEY", "irer_minio_admin")
        self.minio_pass = os.getenv("MINIO_SECRET_KEY", "irer_minio_password")
        self.worker_id = os.getenv("WORKER_ID", f"worker_{os.getpid()}")
        self.bucket_name = "irer-artifacts"

        # Initialize Connections
        self.redis = redis.Redis(host=self.redis_host, port=6379, db=0, decode_responses=True)
        self.minio = Minio(
            self.minio_endpoint,
            access_key=self.minio_user,
            secret_key=self.minio_pass,
            secure=False
        )
        if not self.minio.bucket_exists(self.bucket_name):
            self.minio.make_bucket(self.bucket_name)

    def run(self):
        print(f"[{self.worker_id}] Ready. Waiting for jobs...")
        while True:
            job = self.redis.blpop("jobs:sdg_physics", timeout=10)
            if not job: continue
            _, payload_str = job
            try:
                self.process_job(payload_str)
            except Exception as e:
                print(f"[{self.worker_id}] CRITICAL FAILURE: {e}")
                self.redis.lpush("dlq:sdg_physics", payload_str)

    def process_job(self, payload_str):
        try:
            # 1. Schema Validation
            job_dict = json.loads(payload_str)
            job = JobSchema(**job_dict)
            params = SDGParams(**job.params)

            print(f"[{self.worker_id}] >>> Starting Job: {job.job_id}")

            # 2. Config Hash Enforcement
            calculated_hash = self.compute_config_hash(job.params)
            if job.config_hash and job.config_hash != calculated_hash:
                print(f"[{self.worker_id}] SECURITY ALERT: Config Hash Mismatch! {calculated_hash} != {job.config_hash}")
                # In strict mode, we would raise/return here. For V14, we log it.

            # 3. Setup JAX Params
            sim_params = models.SimParams(
                dt=params.dt,
                epsilon=params.epsilon,
                alpha=params.alpha,
                sigma_k=params.sigma_k,
                c1=params.c1,
                c3=params.c3,
                splash_fraction=params.splash_fraction,
                dx=params.dx
            )

            # 4. Initialize State
            grid_size = job.grid_size
            total_steps = job.total_steps

            key = jax.random.PRNGKey(42)
            initial_field = jax.random.normal(key, shape=(grid_size, grid_size, grid_size), dtype=jnp.complex64)

            initial_state = models.SimState(
                time_idx=0,
                field=initial_field,
                omega=jnp.ones_like(initial_field, dtype=jnp.float32),
                h_norm=jnp.array(0.0),
                config_hash=0
            )

            # 5. Execute JAX Solver with Hardening
            print(f"[{self.worker_id}] Heartbeat: Starting JAX Solver for {job.job_id}...")
            start_time = time.time()
            try:
                final_state, h_norm_history = solver.run_simulation_scan(
                    initial_state, 
                    sim_params, 
                    total_steps, 
                    grid_size
                )
                max_h_norm = float(jnp.max(h_norm_history).block_until_ready())
            except jax.errors.ConcretizationTypeError as e:
                print(f"[{self.worker_id}] CRITICAL JAX ERROR: Concretization Failure. Check static args.")
                raise e  # Let the top-level handler move this to DLQ
            except Exception as e:
                print(f"[{self.worker_id}] UNKNOWN JAX ERROR: {e}")
                raise e

            duration = time.time() - start_time

            # 6. Upload & Report
            self.upload_artifact(job.job_id, final_state, h_norm_history, job.params)

            result_payload = {
                "job_id": job.job_id,
                "status": "SUCCESS",
                "artifact_url": f"s3://{self.bucket_name}/{job.job_id}.h5",
                "metrics": {"max_h_norm": max_h_norm, "duration": duration}
            }
            self.redis.lpush("results:physics", json.dumps(result_payload))
            print(f"[{self.worker_id}] Job {job.job_id} Complete. Max H-Norm: {max_h_norm:.4f}")
        except ValidationError as e:
            print(f"[{self.worker_id}] INVALID JOB PAYLOAD: {e}")
            self.redis.lpush("dlq:sdg_physics", payload_str)
            return
        except Exception as e:
            print(f"[{self.worker_id}] UNKNOWN ERROR: {e}")
            self.redis.lpush("dlq:sdg_physics", payload_str)
            return

    def upload_artifact(self, job_id, state, history, params):
        bio = io.BytesIO()
        from ir_physics.backend import lazy_load_backend
        backend = lazy_load_backend()
        xp = backend["xp"]
        with h5py.File(bio, 'w') as f:
            f.create_dataset("rho", data=xp.abs(state.field)**2)
            f.create_dataset("omega", data=state.omega)
            f.create_dataset("h_norm_history", data=history)
            for k, v in params.items():
                f.attrs[k] = v
        bio.seek(0)
        self.minio.put_object(
            self.bucket_name, f"{job_id}.h5", bio, bio.getbuffer().nbytes,
            content_type="application/x-hdf5"
        )

if __name__ == "__main__":
    worker = SDGWorker()
    worker.run()