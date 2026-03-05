import os
from fastapi import FastAPI
import redis

app = FastAPI()

# Use environment variables for Redis connection (Docker mesh compatible)
REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
REDIS_PORT = int(os.getenv('REDIS_PORT', 6379))
redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0)


@app.get("/status/{job_id}")
async def get_job_status(job_id: str):
    status_bytes = redis_client.get(job_id)
    if status_bytes is None:
        return {"error": "Job not found"}
    # Explicit decode now safe from 'None' error
    status_str = status_bytes.decode('utf-8')
    return {"job_id": job_id, "status": status_str}