"""Lightweight lifecycle event logger for run accountability."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional
import logging
from filelock import FileLock


DEFAULT_AUDIT_LOG = os.path.join("runtime_logs", "run_lifecycle_audit.jsonl")


def log_lifecycle_event(
    stage: str,
    config_hash: Optional[str] = None,
    generation: Optional[int] = None,
    job_id: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None,
    log_path: Optional[str] = None,
) -> None:
    """Write a single lifecycle event line to audit log.
    
    Args:
        stage: dispatch, worker_start, h5_write, validation_write, result_ingest,
               hunter_persist, bleed_queue, bleed_archived, gc_purge, etc.
        config_hash: config_hash of the run (optional)
        generation: generation number (optional)
        job_id: job_id in orchestrator queue (optional)
        details: arbitrary dict with additional context
        log_path: override default audit log path
    """
    if log_path is None:
        log_path = DEFAULT_AUDIT_LOG
    
    try:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
    except Exception:
        pass
    
    event = {
        "stage": stage,
        "config_hash": config_hash,
        "generation": generation,
        "job_id": job_id,
        "details": details or {},
    }
    
    try:
        lock = FileLock(f"{log_path}.lock", timeout=15)
        with lock:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(event) + "\n")
    except Exception as exc:
        logging.warning(f"[AuditLog] Failed to write {stage} event: {exc}")
