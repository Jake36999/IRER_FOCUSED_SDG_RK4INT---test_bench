#!/usr/bin/env python3

"""
result_processor.py
Processes simulation results with validation and lifecycle auditing.
"""

import os
import json
import logging
import sqlite3
import sys
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional

from .diagnostics.runtime_audit import log_lifecycle_event


class ResultProcessor:
    """Processes and validates simulation results."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.db_path = config['db_path']
        self.data_dir = config['data_dir']
        self.provenance_dir = config['provenance_dir']
        self.logger = logging.getLogger(__name__)

    def process_result(self, result_data: Dict[str, Any]) -> bool:
        """
        Process a single result from a worker.
        Returns True if processing successful.
        """
        try:
            job_id = result_data.get('job_id')
            generation = result_data.get('generation')
            config_hash = result_data.get('config_hash')

            if not all([job_id, generation is not None, config_hash]):
                self.logger.error(f"Invalid result data: missing required fields")
                return False

            if str(result_data.get("status", "")).upper() == "FAIL":
                log_lifecycle_event(
                    stage='result_fail_ingest',
                    generation=generation,
                    config_hash=config_hash,
                    job_id=job_id,
                    details={"reason": result_data.get("reason") or result_data.get("error") or "worker_reported_fail"}
                )
                return False

            # Validate result using validation pipeline
            validation_result = self._validate_result(result_data)
            if not validation_result:
                self.logger.warning(f"Validation failed for job {job_id}")
                return False

            # Store in database
            self._store_result(result_data, validation_result)

            log_prime_sse = float(validation_result.get("log_prime_sse", 999.0))
            result_data["_ingested_log_prime_sse"] = log_prime_sse

            triage_tier: Optional[str] = None
            if log_prime_sse < 1.0:
                triage_tier = "GOLDEN"
            elif log_prime_sse < 3.0:
                triage_tier = "SILVER"
            if triage_tier:
                result_data["_triage_tier"] = triage_tier
                self._trigger_visual_observer_async(result_data, log_prime_sse, triage_tier)
                if triage_tier == "GOLDEN":
                    self._trigger_predator_sweep_async(result_data)

            # Emit audit event
            log_lifecycle_event(
                stage='result_ingest',
                generation=generation,
                config_hash=config_hash,
                job_id=job_id,
                details={
                    "validation_score": validation_result.get('composite_fitness', 0.0),
                    "log_prime_sse": log_prime_sse,
                    "triage_tier": triage_tier,
                    "golden_threshold": 1.0,
                    "silver_threshold": 3.0,
                }
            )

            # Handle hunter persistence if needed
            if result_data.get('persist_hunter', False):
                self._persist_hunter_state(result_data)
                log_lifecycle_event(
                    stage='hunter_persist',
                    generation=generation,
                    config_hash=config_hash,
                    job_id=job_id
                )

            self.logger.info(f"Successfully processed result for job {job_id}")
            return True

        except Exception as e:
            self.logger.error(f"Error processing result: {e}")
            return False

    def _validate_result(self, result_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Validate result using validation pipeline.

        Ownership Contract (Phase B Freeze):
        - Canonical ingest SSE (`log_prime_sse`) is sourced here from provenance.
        - Downstream orchestrator accounting/champion logic must consume this value
          through `process_result` output payload fields, not recompute SSE elsewhere.
        """
        try:
            provenance_path = result_data.get("provenance_path")
            if provenance_path and os.path.exists(provenance_path):
                with open(provenance_path, "r", encoding="utf-8") as handle:
                    provenance_payload = json.load(handle)

                spectral = provenance_payload.get("spectral_fidelity", {})
                log_prime_sse = float(spectral.get("log_prime_sse", 999.0))
                composite_fitness = 0.0 if log_prime_sse >= 999.0 else (1.0 / (log_prime_sse + 1e-9))

                return {
                    "composite_fitness": composite_fitness,
                    "log_prime_sse": log_prime_sse,
                    "provenance": provenance_payload,
                }

            sys.path.insert(0, str(Path(__file__).parent.parent))
            from validation_pipeline import ValidationPipeline

            artifact_path = result_data.get("artifact_url")
            params_path = result_data.get("params_path")
            if not artifact_path or not params_path:
                self.logger.error("Validation fallback requires artifact_url and params_path")
                return None

            pipeline = ValidationPipeline(
                input_path=artifact_path,
                params_path=params_path,
                output_dir=self.provenance_dir,
            )
            if not pipeline.run():
                return None

            config_hash = str(result_data.get("config_hash", ""))
            generated_provenance = os.path.join(self.provenance_dir, f"provenance_{config_hash}.json")
            if not os.path.exists(generated_provenance):
                return None

            with open(generated_provenance, "r", encoding="utf-8") as handle:
                provenance_payload = json.load(handle)
            spectral = provenance_payload.get("spectral_fidelity", {})
            log_prime_sse = float(spectral.get("log_prime_sse", 999.0))
            composite_fitness = 0.0 if log_prime_sse >= 999.0 else (1.0 / (log_prime_sse + 1e-9))
            return {
                "composite_fitness": composite_fitness,
                "log_prime_sse": log_prime_sse,
                "provenance": provenance_payload,
            }
        except Exception as e:
            self.logger.error(f"Validation error: {e}")
            return None

    def _trigger_visual_observer_async(self, result_data: Dict[str, Any], log_prime_sse: float, triage_tier: str) -> None:
        """Launch visual observer asynchronously with no blocking wait/check."""
        artifact_path = result_data.get("artifact_url")
        if not artifact_path or not os.path.exists(artifact_path):
            return

        repo_root = Path(__file__).resolve().parent.parent
        script_path = repo_root / "visual_plotting" / "gif_pipeline_manager.py"
        if not script_path.exists():
            self.logger.warning(f"Visual observer script missing: {script_path}")
            return

        try:
            sse_value = float(log_prime_sse)
        except (TypeError, ValueError):
            sse_value = 999.0

        tier_value = str(triage_tier or "SILVER").upper()
        cmd = [
            sys.executable,
            str(script_path),
            "--input",
            str(artifact_path),
            "--sse",
            str(sse_value),
            "--tier",
            tier_value,
        ]

        popen_kwargs: Dict[str, Any] = {
            "stdout": subprocess.DEVNULL,
            "stderr": subprocess.DEVNULL,
            "stdin": subprocess.DEVNULL,
            "close_fds": True,
        }

        if os.name == "nt":
            popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
        else:
            popen_kwargs["start_new_session"] = True

        try:
            subprocess.Popen(cmd, cwd=str(repo_root), **popen_kwargs)
            self.logger.info(
                f"Launched async visual observer for {tier_value} result "
                f"(SSE={sse_value:.6f}, hash={result_data.get('config_hash')})"
            )
        except Exception as exc:
            self.logger.warning(f"Failed to launch async visual observer: {exc}")

    def _trigger_predator_sweep_async(self, result_data: Dict[str, Any]) -> None:
        """Launch predator sweep asynchronously for GOLDEN results."""
        config_hash = result_data.get("config_hash")
        if not config_hash:
            return

        repo_root = Path(__file__).resolve().parent.parent
        script_path = repo_root / "predator_sweep.py"
        cmd = [
            sys.executable,
            str(script_path),
            "--target_hash",
            str(config_hash),
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
            self.logger.info(f"Launched async Predator sweep for GOLDEN hash {str(config_hash)[:12]}")
        except Exception as exc:
            self.logger.warning(f"Failed to launch Predator sweep: {exc}")

    def _store_result(self, result_data: Dict[str, Any], validation_result: Dict[str, Any]):
        """Store result in simulation ledger database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Insert or update result record
            cursor.execute('''
                INSERT OR REPLACE INTO simulation_results
                (job_id, generation, config_hash, config_json, result_json, validation_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, datetime('now'))
            ''', (
                result_data['job_id'],
                result_data['generation'],
                result_data['config_hash'],
                json.dumps(result_data.get('config', {})),
                json.dumps(result_data),
                json.dumps(validation_result)
            ))

            conn.commit()
            conn.close()

        except Exception as e:
            self.logger.error(f"Database error: {e}")
            raise

    def _persist_hunter_state(self, result_data: Dict[str, Any]):
        """Persist hunter state if required."""
        # Implementation depends on hunter specifics
        # For now, just log
        self.logger.info(f"Persisting hunter state for job {result_data['job_id']}")