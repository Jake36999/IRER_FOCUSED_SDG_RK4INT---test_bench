"""
validation/service.py
The Validator Microservice.
"""
import os
import json
import time
import redis
from minio import Minio
from analytics import validate_artifact

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "localhost:9000")
MINIO_KEY = os.getenv("MINIO_ACCESS_KEY", "irer_minio_admin")
MINIO_SECRET = os.getenv("MINIO_SECRET_KEY", "irer_minio_password")

def main():
    print("[Validator] Initializing V14 Gatekeeper...")
    
    r = redis.Redis(host=REDIS_HOST, port=6379, db=0, decode_responses=True)
    m = Minio(MINIO_ENDPOINT, access_key=MINIO_KEY, secret_key=MINIO_SECRET, secure=False)
    
    bucket = "irer-artifacts"
    
    print("[Validator] Watching 'results:physics'...")
    
    while True:
        # 1. Pop raw result from Worker
        item = r.blpop("results:physics", timeout=10)
        if not item:
            continue
            
        _, payload_raw = item
        result = json.loads(payload_raw)
        job_id = result['job_id']
        s3_url = result['artifact_url'] # e.g., s3://irer-artifacts/job_123.h5
        object_name = s3_url.split("/")[-1]
        
        print(f"[Validator] Inspecting: {job_id}")
        
        # 2. Download Artifact
        local_path = f"/tmp/{job_id}.h5"
        try:
            m.fget_object(bucket, object_name, local_path)
            
            # 3. Run Math (TDA + Spectral + Information Tension)
            is_valid, analytics = validate_artifact(local_path)

            # 4. Stamp Verdict and log specific failure reason
            final_status = "SUCCESS" if is_valid else "REJECTED"
            combined_metrics = {**result.get('metrics', {}), **analytics}

            # Log specific failure reason if rejected
            failure_reason = None
            if not is_valid:
                if combined_metrics.get('information_tension_T', 0.0) > 0.09:
                    failure_reason = "Information Tension (T) instability"
                elif combined_metrics.get('h0', 1) <= 0:
                    failure_reason = "No structure (H0 <= 0)"
                elif any(combined_metrics.get(k, 0) == -1 for k in ['h0', 'h1', 'h2']):
                    failure_reason = "TDA computation failed"
                else:
                    failure_reason = "Unknown"

            output_payload = {
                "job_id": job_id,
                "status": final_status,
                "metrics": combined_metrics,
                "artifact_url": s3_url,
                "failure_reason": failure_reason if failure_reason else None
            }

            r.rpush("results:validated", json.dumps(output_payload))
            if is_valid:
                print(f"   -> Verdict: SUCCESS (T: {combined_metrics.get('information_tension_T'):.4f}, H0: {combined_metrics.get('h0')})")
            else:
                print(f"   -> Verdict: REJECTED (Reason: {failure_reason}, T: {combined_metrics.get('information_tension_T'):.4f}, H0: {combined_metrics.get('h0')})")
            
        except Exception as e:
            print(f"[Validator] ERROR processing {job_id}: {e}")
            r.lpush("dlq:validation", payload_raw)
        finally:
            if os.path.exists(local_path):
                os.remove(local_path)

if __name__ == "__main__":
    main()