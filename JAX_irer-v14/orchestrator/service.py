"""
orchestrator/service.py
The Main Control Loop.
"""

import os
import time
import json
import redis
import logging
from kel import KELClient
from hunter import GeneticHunter

# Config
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 4))


class JSONLogFormatter(logging.Formatter):
    def format(self, record):
        log_record = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "service": "orchestrator",
            "message": record.getMessage()
        }
        return json.dumps(log_record)

def setup_logging():
    handler = logging.StreamHandler()
    handler.setFormatter(JSONLogFormatter())
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers = [handler]

def main():
    setup_logging()
    logging.info("V14 Control Plane Initializing...")

    # 1. Connect Components
    try:
        r = redis.Redis(host=REDIS_HOST, port=6379, db=0, decode_responses=True)
        kel = KELClient()
        hunter = GeneticHunter(kel)
        logging.info("Connected to Mesh (Redis + Postgres).")
    except Exception as e:
        logging.error(f"FATAL: {e}")
        return

    logging.info("Entering Hunt Loop...")
    while True:
        # --- A. Check Queue Depth ---
        # Don't overfill Redis. Keep ~2 batches buffered.
        queue_len = r.llen("jobs:sdg_physics")

        if queue_len < BATCH_SIZE:
            logging.info(f"Queue low ({queue_len}). Spawning Generation {hunter.generation}...")
            batch = hunter.generate_batch(batch_size=BATCH_SIZE)

            for manifest in batch:
                # 1. Log Intent to KEL
                kel.register_job(manifest['job_id'], manifest['params'], manifest['generation'])

                # 2. Push to Redis (The Nervous System)
                r.rpush("jobs:sdg_physics", json.dumps(manifest))
                logging.info(f"Dispatched: {manifest['job_id']}")

        # --- B. Process Results (Non-Blocking) ---
        # Pop results from the Workers
        while True:
            result_raw = r.lpop("results:physics")
            if not result_raw:
                break

            result = json.loads(result_raw)
            job_id = result['job_id']
            status = result['status']
            metrics = result['metrics']
            artifact = result['artifact_url']

            # Update Truth Ledger
            kel.update_result(job_id, status, metrics, artifact)
            logging.info(f"Received: {job_id} [{status}] H-Norm: {metrics.get('max_h_norm', '?')}")

        # Pulse check
        time.sleep(2)

if __name__ == "__main__":
    main()