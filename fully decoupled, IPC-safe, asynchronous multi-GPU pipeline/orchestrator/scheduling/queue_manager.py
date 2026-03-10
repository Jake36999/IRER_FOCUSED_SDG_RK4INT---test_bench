"""Job queue management (file-based with worker support)."""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional, Sequence
from filelock import FileLock


class QueueManager:
    """File-based job queue with worker management."""

    def _locked_json_read(self, path: str):
        lock = FileLock(f"{path}.lock", timeout=15)
        with lock:
            if not os.path.exists(path):
                return []
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)

    def _locked_json_write(self, path: str, payload):
        lock = FileLock(f"{path}.lock", timeout=15)
        with lock:
            tmp = f"{path}.tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            os.replace(tmp, path)

    def __init__(self, queue_file: str = "job_queue.json", result_file: str = "result_queue.json"):
        self.queue_file = queue_file
        self.result_file = result_file
        self.workers_file = f"{self.queue_file}.workers.json"
        self.claims_file = f"{self.queue_file}.claims.json"
        self._ensure_queue_exists()

    def _ensure_queue_exists(self) -> None:
        """Create queue files if they don't exist."""
        if not os.path.exists(self.queue_file):
            self._locked_json_write(self.queue_file, [])
        if not os.path.exists(self.result_file):
            self._locked_json_write(self.result_file, [])
        if not os.path.exists(self.workers_file):
            self._locked_json_write(self.workers_file, {})
        if not os.path.exists(self.claims_file):
            self._locked_json_write(self.claims_file, {})

    def _read_workers(self) -> Dict[str, float]:
        payload = self._locked_json_read(self.workers_file)
        if isinstance(payload, dict):
            return payload
        return {}

    def _write_workers(self, workers: Dict[str, float]) -> None:
        self._locked_json_write(self.workers_file, workers)

    def _read_claims(self) -> Dict[str, Dict[str, Any]]:
        payload = self._locked_json_read(self.claims_file)
        if isinstance(payload, dict):
            return payload
        return {}

    def _write_claims(self, claims: Dict[str, Dict[str, Any]]) -> None:
        self._locked_json_write(self.claims_file, claims)

    def push_job(self, manifest_json: str) -> None:
        """Add a job to the queue."""
        self.push_jobs_batch([manifest_json])

    def push_jobs_batch(self, manifest_json_list: Sequence[str]) -> int:
        """Add many jobs to the queue in a single lock/read/write cycle."""
        if not manifest_json_list:
            return 0

        parsed_jobs: List[Dict[str, Any]] = []
        for payload in manifest_json_list:
            parsed_jobs.append(json.loads(payload))

        lock = FileLock(f"{self.queue_file}.lock", timeout=15)
        with lock:
            if not os.path.exists(self.queue_file):
                queue = []
            else:
                with open(self.queue_file, "r", encoding="utf-8") as f:
                    queue = json.load(f)
            queue.extend(parsed_jobs)

            tmp = f"{self.queue_file}.tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(queue, f, indent=2)
            os.replace(tmp, self.queue_file)
        return len(parsed_jobs)

    def pop_job(self) -> Optional[Dict[str, Any]]:
        """Remove and return the first job from queue."""
        # ASTE D.6: Single-window atomic read-modify-write to prevent TOCTOU race
        lock = FileLock(f"{self.queue_file}.lock", timeout=15)
        with lock:
            if not os.path.exists(self.queue_file):
                return None
            with open(self.queue_file, "r", encoding="utf-8") as f:
                queue = json.load(f)
            if not queue:
                return None
            job = queue.pop(0)

            tmp = f"{self.queue_file}.tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(queue, f, indent=2)
            os.replace(tmp, self.queue_file)

            return job

    def claim_job(self, worker_id: str) -> Optional[Dict[str, Any]]:
        """Claim a job for the worker. Returns dict with 'token' and 'payload' or None if queue empty."""
        job_dict = self.pop_job()
        if job_dict is None:
            return None
        # Generate a simple claim token
        claim_token = f"{worker_id}_{int(time.time())}_{hash(json.dumps(job_dict))}"
        claims = self._read_claims()
        claims[claim_token] = {
            "worker_id": worker_id,
            "claimed_at": time.time(),
            "job": job_dict,
        }
        self._write_claims(claims)
        return {"token": claim_token, "payload": json.dumps(job_dict)}

    def complete_job(self, claim_token: str) -> bool:
        """Mark a claimed job as completed. Returns True if successful."""
        claims = self._read_claims()
        if claim_token in claims:
            del claims[claim_token]
            self._write_claims(claims)
            return True
        return False

    def push_result(self, result_json: str) -> None:
        """Push a result to the result queue."""
        lock = FileLock(f"{self.result_file}.lock", timeout=15)
        with lock:
            if not os.path.exists(self.result_file):
                results = []
            else:
                with open(self.result_file, "r", encoding="utf-8") as f:
                    results = json.load(f)
            results.append(json.loads(result_json))

            tmp = f"{self.result_file}.tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)
            os.replace(tmp, self.result_file)

    def get_results(self) -> List[Dict[str, Any]]:
        """Retrieve all pending results and clear the queue."""
        # ASTE D.6: Single-window atomic read-modify-write to prevent TOCTOU race
        lock = FileLock(f"{self.result_file}.lock", timeout=15)
        with lock:
            if not os.path.exists(self.result_file):
                return []
            with open(self.result_file, "r", encoding="utf-8") as f:
                results = json.load(f)

            # Write empty array back atomically inside the SAME lock context
            tmp = f"{self.result_file}.tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump([], f, indent=2)
            os.replace(tmp, self.result_file)

            return results

    def set_worker_heartbeat(self, worker_id: str, timestamp: Optional[float] = None) -> None:
        """Update the heartbeat timestamp for a worker."""
        workers = self._read_workers()
        workers[worker_id] = timestamp or time.time()
        self._write_workers(workers)

    def clear_worker(self, worker_id: str) -> None:
        """Remove a worker from the registry."""
        workers = self._read_workers()
        workers.pop(worker_id, None)
        self._write_workers(workers)

        # Re-queue any claimed jobs by this worker
        claims = self._read_claims()
        to_requeue = [
            (token, claim_data.get("job", {}))
            for token, claim_data in claims.items()
            if str(claim_data.get("worker_id")) == worker_id
        ]
        for token, job in to_requeue:
            claims.pop(token, None)
            self.push_job(json.dumps(job))
        if to_requeue:
            self._write_claims(claims)

    def peek_all(self) -> List[Dict[str, Any]]:
        """Return all jobs without removing."""
        queue = self._locked_json_read(self.queue_file)
        return list(queue)

    def size(self) -> int:
        """Return current queue size."""
        queue = self._locked_json_read(self.queue_file)
        return len(queue)

    def get_worker_heartbeats(self) -> Dict[str, float]:
        """Get all worker heartbeats."""
        return self._read_workers().copy()

    def recover_stale_workers(self, stale_after_seconds: float) -> List[str]:
        """Revoke stale workers and requeue their claimed jobs."""
        now = time.time()
        stale_workers: List[str] = []
        for worker_id, heartbeat_ts in self.get_worker_heartbeats().items():
            if (now - float(heartbeat_ts)) > float(stale_after_seconds):
                stale_workers.append(worker_id)

        for worker_id in stale_workers:
            self.clear_worker(worker_id)
        return stale_workers
