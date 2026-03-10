#!/usr/bin/env python3

"""
orchestrator_engine.py
Core orchestration engine with generation accounting, completion gating,
disk-pressure backpressure, and GC integration.
"""

import os
import json
import logging
import time
import sys
import subprocess
import psutil #type: ignore
import sqlite3
import threading
from pathlib import Path
from typing import Dict, Any, List, Optional

from .scheduling.job_dispatcher import JobDispatcher
from .result_processor import ResultProcessor
from .storage.artifact_bleed_manager import ArtifactBleedManager
from .scheduling.queue_manager import QueueManager
from .storage.artifact_gc import purge_old_artifacts
from .diagnostics.runtime_audit import log_lifecycle_event


class OrchestratorEngine:
    """Main orchestration engine coordinating the simulation lifecycle."""

    global_best_hash: Optional[str] = None
    global_best_sse: float = 999.0

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize orchestration runtime state.

                Ownership Contract (Phase B Freeze):
                - Startup checkpoint restoration ownership lives here via `_load_checkpoint()`.
        - `current_generation`, `global_best_hash`, and `global_best_sse` loaded here
          are the authoritative in-memory state for loop control and GC protection.
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Initialize components
        self.job_dispatcher = JobDispatcher(config)
        self.result_processor = ResultProcessor(config)
        self.bleed_manager = ArtifactBleedManager(config)
        self.queue_manager = QueueManager(queue_file="backlog_queue.json", result_file="result_queue.json")

        # Generation state
        self.current_generation = 0
        self.generations_total = config.get('generations', 100)
        self.jobs_per_generation = config.get('population_size', 10) * config.get('seeds_per_candidate', 1)
        self.completed_jobs: set[str] = set()
        self._result_status_by_key: Dict[str, str] = {}
        self.global_best_hash: Optional[str] = None
        self.global_best_sse: float = 999.0
        self.state_file = config.get('orchestrator_state_file', 'orchestrator_state.json')

        # Disk pressure settings
        self.max_disk_gb = config.get('max_disk_gb', 100)
        self.trigger_bleed_threshold_gb = config.get('trigger_bleed_threshold_gb', 80)

        # Polling settings
        self.poll_interval = config.get('poll_interval', 5)
        self.worker_heartbeat_ttl_seconds = float(config.get('worker_heartbeat_ttl_seconds', 90.0))

        # Threading
        self.running = False
        self.engine_thread: Optional[threading.Thread] = None

        self._load_checkpoint()

    def start(self):
        """Start the orchestration engine."""
        if self.running:
            self.logger.warning("Engine already running")
            return

        self.running = True
        self.engine_thread = threading.Thread(target=self._orchestration_loop, daemon=True)
        self.engine_thread.start()
        self.logger.info("Orchestrator engine started")

    def stop(self):
        """Stop the orchestration engine."""
        self.running = False
        if self.engine_thread:
            self.engine_thread.join(timeout=10)
        self.logger.info("Orchestrator engine stopped")

    def _orchestration_loop(self):
        """Main orchestration loop."""
        try:
            while self.running and self.current_generation < self.generations_total:
                # Check disk pressure
                if self._check_disk_pressure():
                    self.logger.info("Disk pressure detected, pausing orchestration")
                    time.sleep(60)
                    continue

                # Process any completed results
                self._process_pending_results()

                # Recover stale workers and requeue lost claims
                self._recover_stale_workers()

                # Check if current generation is complete
                if self._is_generation_complete():
                    self._advance_generation()
                    continue

                # Dispatch jobs if queue is low
                queue_depth = self.job_dispatcher.get_queue_depth()
                min_queue_depth = self.config.get('min_queue_depth', 5)

                if queue_depth < min_queue_depth:
                    # Generate candidate configs (placeholder - would come from hunter)
                    candidate_configs = self._generate_candidate_configs()
                    if candidate_configs:
                        jobs_dispatched = self.job_dispatcher.dispatch_generation(
                            self.current_generation, candidate_configs
                        )
                        self.logger.info(f"Dispatched {jobs_dispatched} jobs for generation {self.current_generation}")

                # Run GC if needed
                self._run_garbage_collection()

                # Sleep before next iteration
                time.sleep(self.poll_interval)

            self.logger.info("Orchestration loop completed")

        except Exception as e:
            self.logger.error(f"Error in orchestration loop: {e}")
            self.running = False

    def _check_disk_pressure(self) -> bool:
        """Check if disk usage exceeds threshold."""
        try:
            disk_usage = psutil.disk_usage(str(Path(self.config['data_dir']).parent))
            used_gb = disk_usage.used / (1024**3)

            if used_gb > self.trigger_bleed_threshold_gb:
                self.logger.warning(f"Disk usage {used_gb:.1f}GB exceeds threshold {self.trigger_bleed_threshold_gb}GB")
                return True

            return False

        except Exception as e:
            self.logger.error(f"Error checking disk pressure: {e}")
            return False

    def _process_pending_results(self):
        """
        Process any pending results from workers.

        Ownership Contract (Phase B Freeze):
        - Completion accounting ownership is centralized here (`self.completed_jobs`).
        - Champion promotion ownership is centralized here (`global_best_*` updates).
        - Champion checkpoint writes are triggered here via `_save_state_atomic()`.
        """
        import os, time
        try:
            results = self.queue_manager.get_results()
            for result_data in results:
                generation = result_data.get('generation', self.current_generation)
                try:
                    generation_int = int(generation)
                except (TypeError, ValueError):
                    generation_int = self.current_generation
                config_hash = result_data.get('config_hash')
                completion_id = str(result_data.get('job_id') or config_hash or "unknown")
                status = str(result_data.get('status', '')).upper()
                dedupe_key = f"gen_{generation_int}_{str(config_hash or completion_id)}"

                prior_status = self._result_status_by_key.get(dedupe_key)
                if prior_status == 'SUCCESS' or (prior_status == 'FAIL' and status == 'FAIL'):
                    self.logger.info(
                        f"Skipping duplicate result replay for key={dedupe_key} status={status or 'UNKNOWN'}"
                    )
                    continue

                # ASTE D.3 Fix: Atomic HDF5 Release Wait with network filesystem guard
                artifact_path = result_data.get('artifact_url')
                if artifact_path and os.path.exists(artifact_path):
                    for _ in range(20):
                        try:
                            if os.path.getsize(artifact_path) > 0:
                                with open(artifact_path, 'rb+'):
                                    break
                        except IOError:
                            pass
                        time.sleep(0.5)

                success = self.result_processor.process_result(result_data)
                self._result_status_by_key[dedupe_key] = 'SUCCESS' if success else (status or 'UNKNOWN')

                if success and result_data.get('artifact_url'):
                    self.completed_jobs.add(f"gen_{generation_int}_{completion_id}")

                    ingested_sse = float(result_data.get('_ingested_log_prime_sse', 999.0))
                    if config_hash and ingested_sse < self.global_best_sse:
                        self.global_best_sse = ingested_sse
                        self.global_best_hash = str(config_hash)
                        self.logger.info(
                            f"New champion accepted: {self.global_best_hash[:12]} (SSE {self.global_best_sse:.6f})"
                        )
                        self._save_state_atomic()

                    # Queue artifact for bleeding if successful
                    self.bleed_manager.queue_for_bleed(
                        result_data['artifact_url'],
                        result_data.get('generation', 0),
                        result_data.get('config_hash', 'unknown')
                    )
                elif status == 'FAIL':
                    self.completed_jobs.add(f"gen_{generation_int}_{completion_id}")
                if not success:
                    self.logger.warning(f"Failed to process result for job {result_data.get('job_id')}")
        except Exception as e:
            self.logger.error(f"Error processing pending results: {e}")

    def _is_generation_complete(self) -> bool:
        """Check if all jobs for current generation are complete."""
        expected_jobs = self.jobs_per_generation
        completed_count = len([
            job_key
            for job_key in self.completed_jobs
            if job_key.startswith(f"gen_{self.current_generation}_")
        ])

        return completed_count >= expected_jobs

    def _advance_generation(self):
        """Advance to next generation."""
        self.logger.info(f"Generation {self.current_generation} complete, advancing to {self.current_generation + 1}")
        self.current_generation += 1
        self.completed_jobs.clear()
        self._result_status_by_key.clear()
        self._save_state_atomic()
        self._trigger_fss_predictor_async()

        # Emit generation advance event
        log_lifecycle_event(
            stage='generation_advance',
            generation=self.current_generation,
            config_hash=self.config.get('config_hash', 'unknown')
        )

    def _trigger_fss_predictor_async(self) -> None:
        """Launch FSS scaling analyzer as a detached process for next-generation prediction."""
        repo_root = Path(__file__).resolve().parent.parent
        script_path = repo_root / "fss_scaling_analyzer.py"
        cmd = [
            sys.executable,
            str(script_path),
            "--generation",
            str(self.current_generation + 1),
            "--db",
            str(repo_root / "simulation_ledger.db"),
        ]

        popen_kwargs: Dict[str, Any] = {
            "stdout": subprocess.DEVNULL,
            "stderr": subprocess.DEVNULL,
            "stdin": subprocess.DEVNULL,
            "close_fds": True,
            "cwd": str(repo_root),
        }

        if os.name == "nt":
            popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
        else:
            popen_kwargs["start_new_session"] = True

        try:
            subprocess.Popen(cmd, **popen_kwargs)
            self.logger.info("Launched async FSS predictor for next generation")
        except Exception as exc:
            self.logger.warning(f"Failed to launch async FSS predictor: {exc}")

    def _generate_candidate_configs(self) -> List[Dict[str, Any]]:
        """Generate candidate configurations for next jobs."""
        # Placeholder - would integrate with NSGA-II hunter
        return [
            {
                'param1': 1.0,
                'param2': 2.0,
                # ... simulation parameters
            }
        ]

    def _run_garbage_collection(self):
        """Run garbage collection on old artifacts."""
        try:
            # Safely fetch known archived files to prevent deleting un-backed-up data
            archived = set()
            if hasattr(self.bleed_manager, 'get_archived_files'):
                archived = self.bleed_manager.get_archived_files()
            elif hasattr(self.bleed_manager, '_archived_files'):
                archived = self.bleed_manager._archived_files

            protected_files = set()
            if self.global_best_hash:
                protected_files.add(
                    os.path.abspath(
                        os.path.join(
                            self.config.get('data_dir', 'simulation_data'),
                            f"rho_history_{self.global_best_hash}.h5"
                        )
                    )
                )

            purged_count = purge_old_artifacts(
                data_dir=self.config.get('data_dir', 'simulation_data'),
                protected_files=protected_files,
                current_generation=self.current_generation,
                require_archived=self.config.get('gc_require_archived', True),
                min_age_seconds=self.config.get('artifact_gc_min_age_seconds', 300),
                archived_files=archived
            )

            if purged_count > 0:
                self.logger.info(f"Garbage collected {purged_count} artifacts")

        except Exception as e:
            self.logger.error(f"Error during garbage collection: {e}")

    def _recover_stale_workers(self) -> None:
        """Revoke stale worker claims and requeue jobs if heartbeat TTL is exceeded."""
        if self.worker_heartbeat_ttl_seconds <= 0:
            return
        try:
            stale_workers = self.queue_manager.recover_stale_workers(self.worker_heartbeat_ttl_seconds)
            for worker_id in stale_workers:
                self.logger.warning(f"Recovered stale worker claim set: {worker_id}")
                log_lifecycle_event(
                    stage='worker_stale_recovered',
                    generation=self.current_generation,
                    config_hash=self.config.get('config_hash', 'unknown'),
                    details={"worker_id": worker_id, "ttl_seconds": self.worker_heartbeat_ttl_seconds}
                )
        except Exception as exc:
            self.logger.error(f"Error during stale worker recovery: {exc}")

    def get_status(self) -> Dict[str, Any]:
        """Get current engine status."""
        return {
            'running': self.running,
            'current_generation': self.current_generation,
            'total_generations': self.generations_total,
            'queue_depth': self.job_dispatcher.get_queue_depth(),
            'completed_jobs': len(self.completed_jobs),
            'global_best_hash': self.global_best_hash,
            'global_best_sse': self.global_best_sse,
            'disk_pressure': self._check_disk_pressure()
        }

    def _load_checkpoint(self) -> None:
        """Restore generation + champion state from orchestrator_state.json if present."""
        if not os.path.exists(self.state_file):
            return
        try:
            with open(self.state_file, 'r', encoding='utf-8') as handle:
                state = json.load(handle)
            self.current_generation = int(state.get('current_generation', self.current_generation))
            self.global_best_hash = state.get('global_best_hash')
            self.global_best_sse = float(state.get('global_best_sse', self.global_best_sse))
            self.logger.info(
                "Restored orchestrator state: "
                f"gen={self.current_generation}, champion={self.global_best_hash}, sse={self.global_best_sse}"
            )
        except Exception as exc:
            self.logger.warning(f"Could not restore orchestrator state: {exc}")

    def _load_state(self) -> None:
        """Backward-compatible alias for checkpoint restore."""
        self._load_checkpoint()

    def _save_state_atomic(self) -> None:
        """Persist generation + champion state atomically via temp-write + replace."""
        try:
            payload = {
                'current_generation': self.current_generation,
                'global_best_hash': self.global_best_hash,
                'global_best_sse': self.global_best_sse,
            }
            tmp_path = f"{self.state_file}.tmp"
            with open(tmp_path, 'w', encoding='utf-8') as handle:
                json.dump(payload, handle, indent=2)
            os.replace(tmp_path, self.state_file)
        except Exception as exc:
            self.logger.warning(f"Could not persist orchestrator state: {exc}")