#!/usr/bin/env python3
"""Autonomous Backlog Worker Daemon - Bypasses broken Orchestrator/Redis queues"""

import json
import logging
import os
import subprocess
import sys
import uuid
import sqlite3
from filelock import FileLock
from orchestrator.diagnostics.runtime_audit import log_lifecycle_event
from orchestrator.scheduling.queue_manager import QueueManager

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")

QUEUE_FILE = "backlog_queue.json"
RESULT_FILE = "result_queue.json"
DB_FILE = "simulation_ledger.db"
SQLITE_TIMEOUT = float(os.environ.get("ASTE_SQLITE_TIMEOUT", "30.0"))
SQLITE_BUSY_TIMEOUT_MS = int(os.environ.get("ASTE_SQLITE_BUSY_TIMEOUT_MS", "30000"))
SIM_TIMEOUT = float(os.environ.get("ASTE_SIM_TIMEOUT", "7200"))
VAL_TIMEOUT = float(os.environ.get("ASTE_VAL_TIMEOUT", "1800"))


def push_result_to_queue(result_payload):
    """Atomically append a result payload for orchestrator consumption."""
    try:
        lock = FileLock(f"{RESULT_FILE}.lock", timeout=15)
        with lock:
            if os.path.exists(RESULT_FILE):
                with open(RESULT_FILE, "r", encoding="utf-8") as f:
                    results = json.load(f)
            else:
                results = []
            results.append(result_payload)
            tmp = f"{RESULT_FILE}.tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)
            os.replace(tmp, RESULT_FILE)
    except Exception as exc:
        logging.error(f"Failed to push result payload: {exc}")

def pop_job_from_queue():
    """Safely pops the first job from the JSON backlog."""
    try:
        lock = FileLock(f"{QUEUE_FILE}.lock", timeout=15)
        with lock:
            if not os.path.exists(QUEUE_FILE):
                return None
            with open(QUEUE_FILE, "r", encoding="utf-8") as f:
                queue = json.load(f)
            if not queue:
                return None
            job = queue.pop(0)
            tmp = f"{QUEUE_FILE}.tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(queue, f, indent=4)
            os.replace(tmp, QUEUE_FILE)
            return job
    except Exception as e:
        logging.error(f"Queue read error: {e}")
        return None

def write_to_ledger(config_hash, params, status, provenance=None):
    """Writes the job results directly to the SQLite ledger."""
    try:
        # ASTE D.4: Enable Write-Ahead Logging for multi-GPU concurrency
        conn = sqlite3.connect(DB_FILE, timeout=SQLITE_TIMEOUT, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute(f"PRAGMA busy_timeout={SQLITE_BUSY_TIMEOUT_MS};")
        c = conn.cursor()
        
        # Ensure tables exist for Hunter compatibility
        c.execute('''CREATE TABLE IF NOT EXISTS runs 
                     (config_hash TEXT PRIMARY KEY, generation INTEGER, status TEXT, fitness REAL, origin TEXT DEFAULT 'NATURAL', timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
        c.execute("PRAGMA table_info(runs)")
        existing_columns = {row[1] for row in c.fetchall()}
        if "origin" not in existing_columns:
            c.execute("ALTER TABLE runs ADD COLUMN origin TEXT DEFAULT 'NATURAL'")
        c.execute('''CREATE TABLE IF NOT EXISTS parameters 
                     (config_hash TEXT PRIMARY KEY, param_D REAL, param_eta REAL, param_rho_vac REAL, param_a_coupling REAL, param_splash_coupling REAL, param_splash_fraction REAL)''')
        c.execute('''CREATE TABLE IF NOT EXISTS metrics 
                     (config_hash TEXT PRIMARY KEY, log_prime_sse REAL, n_bragg_peaks INTEGER, bragg_prime_sse REAL, collapse_event_count INTEGER, pcs REAL)''')
        
        generation = params.get("generation", -1)
        origin = str(params.get("origin", "NATURAL"))
        
        if status == "SUCCESS" and provenance:
            sf = provenance.get("spectral_fidelity", {})
            am = provenance.get("aletheia_metrics", {})
            fitness = sf.get("log_prime_sse", 999.0)
            
            # Record Run Status
            c.execute("INSERT OR REPLACE INTO runs (config_hash, generation, status, fitness, origin) VALUES (?, ?, ?, ?, ?)", 
                      (config_hash, generation, status, fitness, origin))
                      
            # Record Parameters
            c.execute("INSERT OR REPLACE INTO parameters (config_hash, param_D, param_eta, param_rho_vac, param_a_coupling, param_splash_coupling, param_splash_fraction) VALUES (?, ?, ?, ?, ?, ?, ?)",
                      (config_hash, params.get("param_D"), params.get("param_eta"), params.get("param_rho_vac"), params.get("param_a_coupling"), params.get("param_splash_coupling"), params.get("param_splash_fraction")))
            
            # Record Deep Metrics
            c.execute("INSERT OR REPLACE INTO metrics (config_hash, log_prime_sse, n_bragg_peaks, bragg_prime_sse, collapse_event_count, pcs) VALUES (?, ?, ?, ?, ?, ?)",
                      (config_hash, sf.get("log_prime_sse"), sf.get("n_bragg_peaks"), sf.get("bragg_prime_sse"), sf.get("collapse_event_count"), am.get("pcs")))
        else:
            # Record Failures so the Hunter learns to avoid them
            c.execute("INSERT OR REPLACE INTO runs (config_hash, generation, status, fitness, origin) VALUES (?, ?, ?, ?, ?)", 
                      (config_hash, generation, "FAIL", 999.0, origin))
            
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        logging.error(f"Failed to write to SQL ledger: {e}")
        return False


def record_fail_result(job_id, job_params, config_hash, output_h5_path, reason=None):
    """Enforce fail accounting sinks: ledger + result queue payload."""
    write_to_ledger(config_hash, job_params, "FAIL")
    payload = {
        "job_id": job_id,
        "generation": int(job_params.get("generation", -1)),
        "config_hash": config_hash,
        "artifact_url": output_h5_path,
        "status": "FAIL",
    }
    if reason:
        payload["error"] = reason
        payload["reason"] = reason
    payload["config"] = job_params
    push_result_to_queue(payload)


def main():
    logging.info("[WorkerDaemon] Autonomous Backlog Engine Started.")
    repo_root = os.path.dirname(os.path.abspath(__file__))
    worker_id = os.environ.get("ASTE_WORKER_ID", f"worker_{os.getpid()}")
    queue_manager = QueueManager(queue_file=QUEUE_FILE, result_file=RESULT_FILE)

    os.makedirs("simulation_data", exist_ok=True)
    os.makedirs("provenance_reports", exist_ok=True)

    jobs_completed = 0

    while True:
        queue_manager.set_worker_heartbeat(worker_id)
        claimed_job = queue_manager.claim_job(worker_id)
        if not claimed_job:
            logging.info("[WorkerDaemon] Queue empty. All backlog jobs complete!")
            break

        claim_token = claimed_job["token"]
        claim_released = False
        job_params = {}
        job_id = str(uuid.uuid4())[:16]
        generation = -1
        config_hash = "UNKNOWN"
        temp_manifest_path = ""
        temp_params_path = ""
        output_h5_path = ""
        provenance_path = ""

        try:
            claimed_payload = json.loads(claimed_job["payload"])

            if isinstance(claimed_payload, dict) and "params" in claimed_payload:
                job_params = dict(claimed_payload.get("params", {}))
                job_id = str(claimed_payload.get("job_id") or str(uuid.uuid4())[:16])
                generation = int(claimed_payload.get("generation", -1))
                config_hash = str(claimed_payload.get("config_hash", job_params.get("config_hash", "UNKNOWN")))
                job_origin = str(claimed_payload.get("origin", job_params.get("origin", "NATURAL")))
            else:
                job_params = dict(claimed_payload) if isinstance(claimed_payload, dict) else {}
                config_hash = str(job_params.get("config_hash", "UNKNOWN"))
                generation = int(job_params.get("generation", -1))
                job_origin = str(job_params.get("origin", "NATURAL"))

            job_params.setdefault("config_hash", config_hash)
            job_params.setdefault("generation", generation)
            job_params.setdefault("origin", job_origin)

            manifest = {
                "config_hash": config_hash,
                "job_id": job_id,
                "generation": generation,
                "seed": 0,
                "origin": "BACKLOG_RE_SIM",
                "params": job_params
            }
            validation_params = dict(job_params)
            validation_params.setdefault("config_hash", config_hash)

            temp_manifest_path = f"temp_manifest_{config_hash}.json"
            temp_params_path = f"temp_params_{config_hash}.json"
            output_h5_path = f"simulation_data/rho_history_{config_hash}.h5"
            provenance_path = os.path.join("provenance_reports", f"provenance_{config_hash}.json")

            with open(temp_manifest_path, "w", encoding="utf-8") as handle:
                json.dump(manifest, handle, indent=2)
            with open(temp_params_path, "w", encoding="utf-8") as handle:
                json.dump(validation_params, handle, indent=2)

            logging.info(f"--- Starting Job {jobs_completed+1} | Hash: {config_hash[:12]} ---")

            log_lifecycle_event(
                stage="worker_start",
                config_hash=config_hash,
                generation=-1,
                job_id=job_id,
                details={"worker_id": "autonomous_daemon"}
            )
        except Exception as exc:
            logging.exception(f"❌ Pre-execution setup failed for claim {claim_token}: {exc}")
            record_fail_result(job_id, job_params, config_hash, output_h5_path, reason="pre_execution_setup_failed")
            queue_manager.complete_job(claim_token)
            claim_released = True
            continue

        try:
            ledger_written = False

            # 1. Run ETDRK4 Integration
            subprocess.run(
                [sys.executable, "worker_cupy.py", "--manifest", temp_manifest_path, "--output", output_h5_path],
                cwd=repo_root,
                check=True,
                timeout=SIM_TIMEOUT,
            )
            queue_manager.set_worker_heartbeat(worker_id)

            # 2. Run SFP Validation Pipeline
            subprocess.run(
                [sys.executable, "validation_pipeline.py", "--input", output_h5_path, "--params", temp_params_path, "--output_dir", "provenance_reports"],
                cwd=repo_root,
                check=True,
                timeout=VAL_TIMEOUT,
            )
            queue_manager.set_worker_heartbeat(worker_id)

            # 3. Read Provenance & Write to SQLite Ledger
            if os.path.exists(provenance_path):
                with open(provenance_path, "r", encoding="utf-8") as f:
                    provenance = json.load(f)
                ledger_written = write_to_ledger(config_hash, job_params, "SUCCESS", provenance)
                if ledger_written:
                    push_result_to_queue(
                        {
                            "job_id": job_id,
                            "generation": int(job_params.get("generation", -1)),
                            "config_hash": config_hash,
                            "artifact_url": output_h5_path,
                            "status": "SUCCESS",
                            "provenance_path": provenance_path,
                            "config": job_params,
                        }
                    )
                    logging.info(f"✅ Successfully processed and recorded {config_hash[:12]} to DB.")
                else:
                    push_result_to_queue(
                        {
                            "job_id": job_id,
                            "generation": int(job_params.get("generation", -1)),
                            "config_hash": config_hash,
                            "artifact_url": output_h5_path,
                            "status": "FAIL",
                            "config": job_params,
                        }
                    )
                    logging.warning(f"⚠️ Validation succeeded but DB write failed for {config_hash[:12]}.")
            else:
                logging.warning(f"⚠️ Validation succeeded but provenance JSON missing for {config_hash[:12]}.")
                record_fail_result(job_id, job_params, config_hash, output_h5_path, reason="provenance_missing")

            jobs_completed += 1

        except subprocess.CalledProcessError:
            logging.error(f"❌ Job {config_hash[:12]} failed during execution. Recording FAIL to DB.")
            record_fail_result(job_id, job_params, config_hash, output_h5_path, reason="subprocess_failed")
        except subprocess.TimeoutExpired as exc:
            timed_out_stage = "validation" if "validation_pipeline.py" in str(exc.cmd) else "simulation"
            logging.error(
                f"⏱️ Job {config_hash[:12]} timed out during {timed_out_stage}. "
                f"Timeout={exc.timeout}s. Recording FAIL to DB."
            )

            ledger_ok = write_to_ledger(config_hash, job_params, "FAIL")
            if not ledger_ok:
                logging.warning(f"⚠️ Failed to persist timeout FAIL to ledger for {config_hash[:12]}")

            timeout_payload = {
                "job_id": job_id,
                "generation": int(job_params.get("generation", -1)),
                "config_hash": config_hash,
                "artifact_url": output_h5_path,
                "status": "FAIL",
                "reason": "timeout",
                "error": "timeout",
                "config": job_params,
            }
            if all(k in timeout_payload for k in ("job_id", "generation", "config_hash", "status", "config")):
                queue_manager.push_result(json.dumps(timeout_payload))
            else:
                logging.error(f"Invalid timeout payload shape for {config_hash[:12]}; skipping queue push")
            queue_manager.complete_job(claim_token)
            claim_released = True
        except Exception as exc:
            logging.exception(f"❌ Job {config_hash[:12]} experienced an unhandled error: {exc}")
            record_fail_result(job_id, job_params, config_hash, output_h5_path, reason="unhandled_exception")
        finally:
            if not claim_released:
                queue_manager.complete_job(claim_token)
            if temp_manifest_path and os.path.exists(temp_manifest_path):
                os.remove(temp_manifest_path)
            if temp_params_path and os.path.exists(temp_params_path):
                os.remove(temp_params_path)

if __name__ == "__main__":
    main()