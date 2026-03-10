#!/usr/bin/env python3

"""
job_dispatcher.py
Handles job dispatching with generation preservation and audit logging.
"""

import logging
import os
import json
import time
from typing import Dict, Any, List
from filelock import FileLock

from orchestrator.job_manifest import JobManifest
from .queue_manager import QueueManager
from orchestrator.diagnostics.runtime_audit import log_lifecycle_event


class JobDispatcher:
    """Manages dispatching of simulation jobs with audit trails."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.queue_manager = QueueManager(queue_file="backlog_queue.json", result_file="result_queue.json")
        self.logger = logging.getLogger(__name__)
        self._backlog_decode_retry_events = 0
        self._backlog_decode_retry_attempts = 0

    def _pop_backlog_seed_configs(self, count: int) -> List[Dict[str, Any]]:
        """Pop first `count` backlog configs under FileLock with atomic rewrite."""
        if count <= 0:
            return []

        backlog_file = "backlog_queue.json"
        if not os.path.exists(backlog_file):
            return []

        seeds: List[Dict[str, Any]] = []
        try:
            lock = FileLock("backlog_queue.json.lock", timeout=15)
            with lock:
                max_decode_retries = int(self.config.get("backlog_seed_json_decode_retries", 2))
                backoff_seconds = float(self.config.get("backlog_seed_json_decode_backoff_seconds", 0.05))
                queue_data = None

                for attempt in range(max_decode_retries + 1):
                    with open(backlog_file, "r", encoding="utf-8") as handle:
                        try:
                            queue_data = json.load(handle)
                            break
                        except json.JSONDecodeError:
                            if attempt >= max_decode_retries:
                                self.logger.warning("Backlog queue JSON malformed after retries; skipping seed ingest for this cycle")
                                return []
                            self._backlog_decode_retry_events += 1
                            self._backlog_decode_retry_attempts += 1
                            sleep_for = backoff_seconds * (2 ** attempt)
                            self.logger.warning(
                                f"Backlog queue JSON decode failed (attempt {attempt + 1}/{max_decode_retries + 1}); "
                                f"retrying in {sleep_for:.2f}s"
                            )
                            time.sleep(sleep_for)

                if queue_data is None:
                    return []

                if not isinstance(queue_data, list):
                    self.logger.warning("Backlog queue payload is not a list; skipping seed ingest for this cycle")
                    return []

                seeds = [item for item in queue_data[:count] if isinstance(item, dict)]
                remaining = queue_data[count:]

                tmp_path = f"{backlog_file}.tmp"
                with open(tmp_path, "w", encoding="utf-8") as handle:
                    json.dump(remaining, handle, indent=2)
                os.replace(tmp_path, backlog_file)
        except Exception as exc:
            self.logger.warning(f"Backlog seed ingest skipped due to lock/read error: {exc}")

        if seeds:
            self.logger.info(f"Backlog seeding injected {len(seeds)} config(s) before stochastic dispatch")
        return seeds

    def dispatch_generation(self, generation: int, candidate_configs: List[Dict[str, Any]]) -> int:
        """
        Dispatch all jobs for a generation.
        Returns the number of jobs dispatched.
        """
        jobs_dispatched = 0
        manifests: List[JobManifest] = []

        pop_size = len(candidate_configs)
        seed_configs = self._pop_backlog_seed_configs(pop_size)
        dispatch_configs = seed_configs + candidate_configs[: max(0, pop_size - len(seed_configs))]

        seeds_per_candidate = self.config.get('seeds_per_candidate', 1)

        for candidate_config in dispatch_configs:
            for seed_idx in range(seeds_per_candidate):
                manifest = JobManifest.from_params(
                    params=candidate_config,
                    generation=generation,
                    seed=seed_idx,
                    origin=str(self.config.get('origin', 'NATURAL')),
                )
                manifests.append(manifest)

        if manifests:
            self.queue_manager.push_jobs_batch([manifest.to_json() for manifest in manifests])

        if self._backlog_decode_retry_events > 0:
            self.logger.info(
                "Backlog seed decode retries observed: "
                f"events={self._backlog_decode_retry_events}, attempts={self._backlog_decode_retry_attempts}"
            )

        for manifest in manifests:
            seed_idx = int(getattr(manifest, "seed", 0))

            log_lifecycle_event(
                stage='dispatch',
                generation=generation,
                config_hash=manifest.config_hash,
                job_id=manifest.job_id,
                details={"seed_idx": seed_idx}
            )

            jobs_dispatched += 1
            self.logger.info(f"Dispatched job {manifest.job_id} for generation {generation}")

        return jobs_dispatched

    def get_queue_depth(self) -> int:
        """Get current queue depth."""
        return self.queue_manager.size()

    def peek_pending_jobs(self) -> List[Dict[str, Any]]:
        """Get list of pending jobs without removing them."""
        return self.queue_manager.peek_all()