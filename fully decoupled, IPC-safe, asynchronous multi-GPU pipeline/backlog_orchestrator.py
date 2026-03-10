#!/usr/bin/env python3
"""
backlog_orchestrator.py

Strict Backlog Mode runner for static IRER sweeps.

Design goals:
- Linear, immutable backlog execution (no ASTE generation/mutation logic).
- Input from input_configs/*.json and/or prep_backlog.py manifest output.
- ETDRK4 execution + SFP v3.2 validation for each job.
- WAL-backed SQLite ledger writes with exponential backoff.
- Post-validation artifact GC that removes heavy .h5 files while preserving
  provenance JSON and *_etch_ready.csv outputs.

Execution modes:
- direct: run worker + validation subprocesses in this process.
- daemon: enqueue manifests for external worker_daemon.py and consume result queue
  (compatible with orchestrator/ package deployments).
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import glob
import json
import logging
import os
import random
import re
import sqlite3
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import h5py  # type: ignore

from config_utils import generate_canonical_hash

try:
    from orchestrator.job_manifest import JobManifest  # type: ignore
    from orchestrator.scheduling.queue_manager import QueueManager  # type: ignore

    ORCHESTRATOR_QUEUE_AVAILABLE = True
except Exception:
    JobManifest = None  # type: ignore
    QueueManager = None  # type: ignore
    ORCHESTRATOR_QUEUE_AVAILABLE = False

try:
    from orchestrator.storage import artifact_gc as orchestrator_artifact_gc  # type: ignore
except Exception:
    orchestrator_artifact_gc = None


NAN_PATTERN = re.compile(r"(\[nan\]|\bnan\b)", re.IGNORECASE)
KILLSWITCH_PATTERN = re.compile(r"(\[killswitch\]|killswitch)", re.IGNORECASE)


@dataclass(frozen=True)
class BacklogJob:
    config_hash: str
    params_path: str
    params: Dict[str, Any]
    generation: int
    origin: str


class LedgerDB:
    """SQLite writer with WAL mode + exponential backoff."""

    def __init__(
        self,
        db_path: str,
        max_retries: int = 6,
        sqlite_timeout_sec: float = 30.0,
        sqlite_busy_timeout_ms: int = 30000,
    ) -> None:
        self.db_path = db_path
        self.max_retries = max_retries
        self.sqlite_timeout_sec = sqlite_timeout_sec
        self.sqlite_busy_timeout_ms = sqlite_busy_timeout_ms
        self._init_schema()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=self.sqlite_timeout_sec, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute(f"PRAGMA busy_timeout={self.sqlite_busy_timeout_ms};")
        return conn

    def _execute_with_retry(
        self,
        cursor: sqlite3.Cursor,
        query: str,
        params: Sequence[Any] = (),
    ) -> None:
        for attempt in range(self.max_retries):
            try:
                if params:
                    cursor.execute(query, tuple(params))
                else:
                    cursor.execute(query)
                return
            except sqlite3.OperationalError as exc:
                if "database is locked" in str(exc).lower() and attempt < self.max_retries - 1:
                    sleep_s = (2**attempt) * 0.1 + random.uniform(0.0, 0.1)
                    logging.warning(
                        "[Ledger] DB locked, retrying in %.2fs (%d/%d)",
                        sleep_s,
                        attempt + 1,
                        self.max_retries,
                    )
                    time.sleep(sleep_s)
                    continue
                raise

    def _init_schema(self) -> None:
        with self._get_connection() as conn:
            cursor = conn.cursor()

            self._execute_with_retry(
                cursor,
                """
                CREATE TABLE IF NOT EXISTS runs (
                    config_hash TEXT PRIMARY KEY,
                    generation INTEGER,
                    status TEXT DEFAULT 'pending',
                    fitness REAL DEFAULT 0.0,
                    parent_1 TEXT,
                    parent_2 TEXT,
                    origin TEXT DEFAULT 'BACKLOG_STRICT'
                )
                """,
            )
            self._execute_with_retry(
                cursor,
                """
                CREATE TABLE IF NOT EXISTS parameters (
                    config_hash TEXT PRIMARY KEY,
                    param_D REAL,
                    param_eta REAL,
                    param_rho_vac REAL,
                    param_a_coupling REAL,
                    param_splash_coupling REAL,
                    param_splash_fraction REAL,
                    FOREIGN KEY(config_hash) REFERENCES runs(config_hash)
                )
                """,
            )
            self._execute_with_retry(
                cursor,
                """
                CREATE TABLE IF NOT EXISTS metrics (
                    config_hash TEXT PRIMARY KEY,
                    log_prime_sse REAL,
                    primary_harmonic_error REAL,
                    missing_peak_penalty REAL,
                    noise_penalty REAL,
                    sse_null_phase_scramble REAL,
                    sse_null_target_shuffle REAL,
                    pcs REAL,
                    pli REAL,
                    ic REAL,
                    c4_contrast REAL,
                    ablated_c4_contrast REAL,
                    j_info_mean REAL,
                    grad_phase_var REAL,
                    max_amp_peak REAL,
                    clamp_fraction_mean REAL,
                    omega_sat_mean REAL,
                    collapse_event_count INTEGER,
                    dominant_peak_k REAL,
                    secondary_peak_k REAL,
                    stage4_early_reject INTEGER DEFAULT 0,
                    FOREIGN KEY(config_hash) REFERENCES runs(config_hash)
                )
                """,
            )
            self._execute_with_retry(
                cursor,
                """
                CREATE TABLE IF NOT EXISTS results (
                    config_hash TEXT PRIMARY KEY,
                    generation INTEGER,
                    param_D REAL,
                    param_eta REAL,
                    param_rho_vac REAL,
                    param_a_coupling REAL,
                    param_splash_coupling REAL,
                    param_splash_fraction REAL,
                    log_prime_sse REAL,
                    fitness REAL,
                    parent_1 TEXT,
                    parent_2 TEXT
                )
                """,
            )

            for stmt in (
                "ALTER TABLE runs ADD COLUMN origin TEXT DEFAULT 'BACKLOG_STRICT'",
                "ALTER TABLE metrics ADD COLUMN primary_harmonic_error REAL DEFAULT 999.0",
                "ALTER TABLE metrics ADD COLUMN missing_peak_penalty REAL DEFAULT 0.0",
                "ALTER TABLE metrics ADD COLUMN noise_penalty REAL DEFAULT 0.0",
                "ALTER TABLE metrics ADD COLUMN collapse_event_count INTEGER DEFAULT 0",
                "ALTER TABLE metrics ADD COLUMN dominant_peak_k REAL DEFAULT 0.0",
                "ALTER TABLE metrics ADD COLUMN secondary_peak_k REAL DEFAULT 0.0",
                "ALTER TABLE metrics ADD COLUMN stage4_early_reject INTEGER DEFAULT 0",
                "ALTER TABLE metrics ADD COLUMN basin_id INTEGER DEFAULT -1",
                "ALTER TABLE metrics ADD COLUMN origin TEXT DEFAULT 'BACKLOG'",
                "ALTER TABLE metrics ADD COLUMN fitness REAL DEFAULT 0.0",
            ):
                try:
                    self._execute_with_retry(cursor, stmt)
                except sqlite3.OperationalError:
                    pass

            conn.commit()

    def register_job(self, job: BacklogJob) -> None:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            self._execute_with_retry(
                cursor,
                """
                INSERT OR REPLACE INTO runs
                (config_hash, generation, status, fitness, parent_1, parent_2, origin)
                VALUES (?, ?, 'pending', COALESCE((SELECT fitness FROM runs WHERE config_hash=?), 0.0), NULL, NULL, ?)
                """,
                (job.config_hash, job.generation, job.config_hash, job.origin),
            )
            self._execute_with_retry(
                cursor,
                """
                INSERT OR REPLACE INTO parameters
                (config_hash, param_D, param_eta, param_rho_vac, param_a_coupling, param_splash_coupling, param_splash_fraction)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job.config_hash,
                    _safe_float(job.params.get("param_D")),
                    _safe_float(job.params.get("param_eta")),
                    _safe_float(job.params.get("param_rho_vac")),
                    _safe_float(job.params.get("param_a_coupling")),
                    _safe_float(job.params.get("param_splash_coupling"), default=0.5),
                    _safe_float(job.params.get("param_splash_fraction"), default=-0.5),
                ),
            )
            conn.commit()

    def mark_failed(self, config_hash: str) -> None:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            self._execute_with_retry(
                cursor,
                "UPDATE runs SET status='failed', fitness=0.0 WHERE config_hash=?",
                (config_hash,),
            )
            conn.commit()

    def write_success_from_provenance(self, job: BacklogJob, provenance_path: str) -> None:
        try:
            with open(provenance_path, "r", encoding="utf-8") as handle:
                prov = json.load(handle)
        except Exception:
            self.mark_failed(job.config_hash)
            return

        spec = prov.get("spectral_fidelity", {}) if isinstance(prov, dict) else {}
        aletheia = prov.get("aletheia_metrics", {}) if isinstance(prov, dict) else {}
        bridge = prov.get("empirical_bridge", {}) if isinstance(prov, dict) else {}

        log_prime_sse = _safe_float(spec.get("log_prime_sse"), default=999.0)
        primary_error = _safe_float(spec.get("primary_harmonic_error"), default=999.0)
        fitness = 0.0 if log_prime_sse >= 999.0 else max(0.0, 1.0 / (log_prime_sse + 1e-9))
        stage4_early_reject = 1 if log_prime_sse > 15.0 else 0

        with self._get_connection() as conn:
            cursor = conn.cursor()
            self._execute_with_retry(
                cursor,
                """
                INSERT OR REPLACE INTO metrics
                (
                    config_hash, log_prime_sse, primary_harmonic_error, missing_peak_penalty, noise_penalty,
                    sse_null_phase_scramble, sse_null_target_shuffle, pcs, pli, ic, c4_contrast, ablated_c4_contrast,
                    j_info_mean, grad_phase_var, max_amp_peak, clamp_fraction_mean, omega_sat_mean, collapse_event_count,
                    dominant_peak_k, secondary_peak_k, stage4_early_reject
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job.config_hash,
                    log_prime_sse,
                    primary_error,
                    _safe_float(spec.get("missing_peak_penalty"), default=0.0),
                    _safe_float(spec.get("noise_penalty"), default=0.0),
                    _safe_float(spec.get("sse_null_phase_scramble"), default=999.0),
                    _safe_float(spec.get("sse_null_target_shuffle"), default=999.0),
                    _safe_float(aletheia.get("pcs"), default=_safe_float(spec.get("pcs"), default=0.0)),
                    _safe_float(aletheia.get("pli"), default=0.0),
                    _safe_float(aletheia.get("ic"), default=1.0),
                    _safe_float(bridge.get("c4_interference_contrast"), default=0.0),
                    _safe_float(bridge.get("ablated_c4_contrast"), default=0.0),
                    _safe_float(aletheia.get("j_info_l2_mean"), default=0.0),
                    _safe_float(aletheia.get("grad_phase_var_mean"), default=0.0),
                    _safe_float(aletheia.get("max_amp_peak"), default=0.0),
                    _safe_float(aletheia.get("clamp_fraction_mean"), default=0.0),
                    _safe_float(aletheia.get("omega_sat_mean"), default=0.0),
                    int(spec.get("collapse_event_count", 0) or 0),
                    _safe_float(spec.get("dominant_peak_k"), default=0.0),
                    _safe_float(spec.get("secondary_peak_k"), default=0.0),
                    stage4_early_reject,
                ),
            )
            self._execute_with_retry(
                cursor,
                "UPDATE runs SET status='completed', fitness=?, origin=? WHERE config_hash=?",
                (fitness, job.origin, job.config_hash),
            )
            self._execute_with_retry(
                cursor,
                """
                INSERT OR REPLACE INTO results
                (
                    config_hash, generation, param_D, param_eta, param_rho_vac,
                    param_a_coupling, param_splash_coupling, param_splash_fraction,
                    log_prime_sse, fitness, parent_1, parent_2
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL)
                """,
                (
                    job.config_hash,
                    job.generation,
                    _safe_float(job.params.get("param_D")),
                    _safe_float(job.params.get("param_eta")),
                    _safe_float(job.params.get("param_rho_vac")),
                    _safe_float(job.params.get("param_a_coupling")),
                    _safe_float(job.params.get("param_splash_coupling"), default=0.5),
                    _safe_float(job.params.get("param_splash_fraction"), default=-0.5),
                    log_prime_sse,
                    fitness,
                ),
            )
            conn.commit()


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _setup_logging(verbose: bool = False) -> None:
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s")
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    if not logger.handlers:
        logger.addHandler(handler)


def _load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _strip_config_hash_for_hashing(params: Dict[str, Any]) -> Dict[str, Any]:
    clean = dict(params)
    clean.pop("config_hash", None)
    return clean


def _normalize_job_params(raw_params: Dict[str, Any], default_generation: int, default_origin: str) -> Dict[str, Any]:
    params = dict(raw_params)
    if "generation" not in params:
        params["generation"] = default_generation
    if "origin" not in params:
        params["origin"] = default_origin
    return params


def _resolve_or_create_params_file(
    params: Dict[str, Any],
    config_dir: str,
    hinted_path: Optional[str] = None,
) -> Tuple[str, str, Dict[str, Any]]:
    os.makedirs(config_dir, exist_ok=True)
    clean_for_hash = _strip_config_hash_for_hashing(params)
    config_hash = params.get("config_hash") or generate_canonical_hash(clean_for_hash)
    params["config_hash"] = config_hash

    if hinted_path:
        hinted_abs = os.path.abspath(hinted_path)
        if os.path.exists(hinted_abs):
            return config_hash, hinted_abs, params

    out_path = os.path.abspath(os.path.join(config_dir, f"config_{config_hash}.json"))
    if not os.path.exists(out_path):
        with open(out_path, "w", encoding="utf-8") as handle:
            json.dump(params, handle, indent=2)
    return config_hash, out_path, params


def _iter_manifest_rows(manifest_path: str) -> List[Any]:
    if manifest_path.endswith(".csv"):
        with open(manifest_path, "r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))

    data = _load_json(manifest_path)
    if isinstance(data, dict) and "jobs" in data and isinstance(data["jobs"], list):
        return data["jobs"]
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return [data]
    raise ValueError(f"Unsupported manifest format: {manifest_path}")


def _build_jobs_from_manifest(
    manifest_path: str,
    config_dir: str,
    default_generation: int,
    default_origin: str,
) -> List[BacklogJob]:
    rows = _iter_manifest_rows(manifest_path)
    jobs: List[BacklogJob] = []
    for row in rows:
        if isinstance(row, str):
            params_path = os.path.abspath(row)
            params = _load_json(params_path)
            params = _normalize_job_params(params, default_generation, default_origin)
            config_hash, resolved_path, params = _resolve_or_create_params_file(params, config_dir, params_path)
            jobs.append(
                BacklogJob(
                    config_hash=config_hash,
                    params_path=resolved_path,
                    params=params,
                    generation=int(params.get("generation", default_generation)),
                    origin=str(params.get("origin", default_origin)),
                )
            )
            continue

        if not isinstance(row, dict):
            logging.warning("[Backlog] Skipping unsupported manifest row type: %r", type(row))
            continue

        hinted_path = (
            row.get("params_path")
            or row.get("config_path")
            or row.get("path")
            or row.get("params_file")
        )

        if "params" in row and isinstance(row["params"], dict):
            params = row["params"]
        elif any(key.startswith("param_") for key in row.keys()) or "simulation" in row:
            params = dict(row)
        elif hinted_path:
            params_path = os.path.abspath(str(hinted_path))
            params = _load_json(params_path)
            params = _normalize_job_params(params, default_generation, default_origin)
            config_hash, resolved_path, params = _resolve_or_create_params_file(params, config_dir, params_path)
            jobs.append(
                BacklogJob(
                    config_hash=config_hash,
                    params_path=resolved_path,
                    params=params,
                    generation=int(params.get("generation", default_generation)),
                    origin=str(params.get("origin", default_origin)),
                )
            )
            continue
        else:
            logging.warning("[Backlog] Skipping manifest row with no params payload: %s", row)
            continue

        params = _normalize_job_params(params, default_generation, default_origin)
        config_hash, resolved_path, params = _resolve_or_create_params_file(
            params,
            config_dir,
            hinted_path=str(hinted_path) if hinted_path else None,
        )
        jobs.append(
            BacklogJob(
                config_hash=config_hash,
                params_path=resolved_path,
                params=params,
                generation=int(params.get("generation", default_generation)),
                origin=str(params.get("origin", default_origin)),
            )
        )
    return jobs


def _build_jobs_from_config_dir(
    config_dir: str,
    default_generation: int,
    default_origin: str,
) -> List[BacklogJob]:
    jobs: List[BacklogJob] = []
    for params_path in sorted(glob.glob(os.path.join(config_dir, "*.json"))):
        try:
            params = _load_json(params_path)
            if not isinstance(params, dict):
                continue
            params = _normalize_job_params(params, default_generation, default_origin)
            config_hash, resolved_path, params = _resolve_or_create_params_file(params, config_dir, params_path)
            jobs.append(
                BacklogJob(
                    config_hash=config_hash,
                    params_path=resolved_path,
                    params=params,
                    generation=int(params.get("generation", default_generation)),
                    origin=str(params.get("origin", default_origin)),
                )
            )
        except Exception as exc:
            logging.warning("[Backlog] Failed loading %s: %s", params_path, exc)
    return jobs


def _dedupe_jobs(jobs: List[BacklogJob]) -> List[BacklogJob]:
    seen: set[str] = set()
    out: List[BacklogJob] = []
    for job in jobs:
        if job.config_hash in seen:
            continue
        seen.add(job.config_hash)
        out.append(job)
    return out


def _load_backlog_jobs(args: argparse.Namespace) -> List[BacklogJob]:
    jobs: List[BacklogJob] = []

    if args.source in ("auto", "manifest") and args.manifest and os.path.exists(args.manifest):
        jobs.extend(
            _build_jobs_from_manifest(
                manifest_path=args.manifest,
                config_dir=args.config_dir,
                default_generation=args.generation,
                default_origin=args.origin,
            )
        )

    if args.source in ("auto", "directory"):
        jobs.extend(
            _build_jobs_from_config_dir(
                config_dir=args.config_dir,
                default_generation=args.generation,
                default_origin=args.origin,
            )
        )

    return _dedupe_jobs(jobs)

def _scan_runtime_flags(log_blob: str) -> Tuple[bool, bool]:
    if not log_blob:
        return False, False
    saw_killswitch = bool(KILLSWITCH_PATTERN.search(log_blob))
    saw_nan = bool(NAN_PATTERN.search(log_blob))
    return saw_killswitch, saw_nan


def _run_command(cmd: List[str], cwd: str, timeout_sec: int) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout_sec,
    )


def _inspect_h5_schema(h5_path: str) -> Dict[str, bool]:
    out = {
        "telemetry": False,
        "extended_telemetry": False,
    }
    try:
        with h5py.File(h5_path, "r") as handle:
            out["telemetry"] = "telemetry" in handle
            out["extended_telemetry"] = "extended_telemetry" in handle
    except Exception as exc:
        logging.warning("[Schema] Could not inspect HDF5 %s: %s", h5_path, exc)
    return out


def _find_provenance_file(provenance_dir: str, config_hash: str) -> Optional[str]:
    target_path = os.path.abspath(os.path.join(provenance_dir, f"provenance_{config_hash}.json"))
    if os.path.exists(target_path):
        return target_path
    return None


def _invoke_artifact_gc(data_dir: str, artifact_path: str, min_age_seconds: Optional[int] = None) -> None:
    if min_age_seconds is None:
        min_age_seconds = int(os.environ.get("ASTE_GC_MIN_AGE_SECONDS", "0"))

    artifact_abs = os.path.abspath(artifact_path)
    if not os.path.exists(artifact_abs):
        return

    if orchestrator_artifact_gc is not None:
        try:
            all_h5 = [
                os.path.abspath(os.path.join(data_dir, name))
                for name in os.listdir(data_dir)
                if name.endswith(".h5")
            ]
            protected = {path for path in all_h5 if path != artifact_abs}
            orchestrator_artifact_gc.purge_old_artifacts(
                data_dir=data_dir,
                protected_files=protected,
                current_generation=None,
                min_age_seconds=min_age_seconds,
                require_archived=False,
            )
            if not os.path.exists(artifact_abs):
                logging.info("[ArtifactGC] Purged %s via orchestrator.storage.artifact_gc", os.path.basename(artifact_abs))
                return
        except Exception as exc:
            logging.warning("[ArtifactGC] orchestrator purge failed (%s), falling back to direct remove.", exc)

    try:
        os.remove(artifact_abs)
        logging.info("[ArtifactGC] Purged %s", os.path.basename(artifact_abs))
    except OSError as exc:
        logging.warning("[ArtifactGC] Could not remove %s: %s", artifact_abs, exc)


def _known_live_workers(queue: Any, ttl_seconds: int) -> int:
    if hasattr(queue, "get_worker_heartbeats"):
        try:
            heartbeats = queue.get_worker_heartbeats() or {}
            now_ts = time.time()
            return len(
                [
                    wid
                    for wid, ts in heartbeats.items()
                    if isinstance(ts, (int, float)) and (now_ts - float(ts)) <= ttl_seconds
                ]
            )
        except Exception:
            return 0

    if not hasattr(queue, "get_stale_workers"):
        return 0

    try:
        stale = set(queue.get_stale_workers(ttl_seconds))
    except Exception:
        return 0

    try:
        client = getattr(queue, "_client", None)
        if client is not None:
            known = set(client.smembers(queue.WORKER_KNOWN_SET) or set())
            return len([wid for wid in known if wid not in stale])
    except Exception:
        pass

    known_fallback = getattr(queue, "_fallback_known_workers", set())
    return len([wid for wid in known_fallback if wid not in stale])


def _wait_for_worker_daemon(queue: Any, ttl_seconds: int, wait_timeout_sec: int, poll_interval: float) -> bool:
    deadline = time.time() + wait_timeout_sec
    while time.time() < deadline:
        live = _known_live_workers(queue, ttl_seconds)
        if live > 0:
            return True
        logging.warning("[Backlog] Waiting for worker_daemon heartbeat...")
        time.sleep(poll_interval)
    return False


def _process_job_direct(
    job: BacklogJob,
    args: argparse.Namespace,
    db: LedgerDB,
) -> Tuple[bool, bool]:
    output_h5 = os.path.abspath(os.path.join(args.data_dir, f"rho_history_{job.config_hash}.h5"))
    worker_cmd = [
        args.python_executable,
        args.worker_script,
        "--params",
        job.params_path,
        "--output",
        output_h5,
    ]
    logging.info("[Run %s] ETDRK4 start", job.config_hash[:12])
    worker_proc = _run_command(worker_cmd, cwd=args.repo_root, timeout_sec=args.worker_timeout_sec)
    worker_log = (worker_proc.stdout or "") + "\n" + (worker_proc.stderr or "")
    kill_flag, nan_flag = _scan_runtime_flags(worker_log)

    if worker_proc.returncode != 0 or kill_flag or nan_flag:
        logging.error(
            "[Run %s] Worker failed (rc=%s, killswitch=%s, nan=%s)",
            job.config_hash[:12],
            worker_proc.returncode,
            kill_flag,
            nan_flag,
        )
        db.mark_failed(job.config_hash)
        return False, kill_flag

    schema_flags = _inspect_h5_schema(output_h5)
    if not schema_flags["telemetry"]:
        logging.warning("[Run %s] Missing HDF5 group: telemetry", job.config_hash[:12])
    if not schema_flags["extended_telemetry"]:
        logging.warning("[Run %s] Missing HDF5 group: extended_telemetry", job.config_hash[:12])

    val_cmd = [
        args.python_executable,
        args.validation_script,
        "--input",
        output_h5,
        "--params",
        job.params_path,
        "--output_dir",
        args.provenance_dir,
    ]
    logging.info("[Run %s] Validation start", job.config_hash[:12])
    val_proc = _run_command(val_cmd, cwd=args.repo_root, timeout_sec=args.validation_timeout_sec)
    val_log = (val_proc.stdout or "") + "\n" + (val_proc.stderr or "")
    kill_flag_2, nan_flag_2 = _scan_runtime_flags(val_log)
    kill_flag = kill_flag or kill_flag_2

    if val_proc.returncode != 0 or kill_flag_2 or nan_flag_2:
        logging.error(
            "[Run %s] Validation failed (rc=%s, killswitch=%s, nan=%s)",
            job.config_hash[:12],
            val_proc.returncode,
            kill_flag_2,
            nan_flag_2,
        )
        db.mark_failed(job.config_hash)
        return False, kill_flag

    provenance_path = _find_provenance_file(args.provenance_dir, job.config_hash)
    if not provenance_path:
        logging.error("[Run %s] Validation completed but provenance file not found.", job.config_hash[:12])
        db.mark_failed(job.config_hash)
        return False, kill_flag

    db.write_success_from_provenance(job, provenance_path)
    _invoke_artifact_gc(args.data_dir, output_h5, min_age_seconds=args.gc_min_age)

    return True, kill_flag


def _process_job_daemon(
    job: BacklogJob,
    args: argparse.Namespace,
    db: LedgerDB,
    queue: Any,
    result_buffer: Dict[str, Dict[str, Any]],
) -> Tuple[bool, bool]:
    result_payload: Optional[Dict[str, Any]]
    if job.config_hash in result_buffer:
        result_payload = result_buffer.pop(job.config_hash)
    else:
        if not _wait_for_worker_daemon(
            queue=queue,
            ttl_seconds=args.worker_heartbeat_ttl_seconds,
            wait_timeout_sec=args.worker_wait_timeout_sec,
            poll_interval=args.poll_interval,
        ):
            logging.error("[Run %s] No live worker_daemon heartbeat detected.", job.config_hash[:12])
            db.mark_failed(job.config_hash)
            return False, False

        manifest = JobManifest(  # type: ignore[misc]
            job_id=job.config_hash[:16],
            config_hash=job.config_hash,
            generation=job.generation,
            seed=0,
            params=job.params,
            origin=job.origin,
        )
        queue.push_job(manifest.to_json())
        logging.info("[Run %s] Dispatched to worker_daemon queue.", job.config_hash[:12])

        deadline = time.time() + args.worker_timeout_sec
        result_payload = None
        while time.time() < deadline:
            pending_results = queue.get_results()
            if not pending_results:
                time.sleep(args.poll_interval)
                continue

            for payload in pending_results:
                if not isinstance(payload, dict):
                    continue
                cfg_hash = str(payload.get("config_hash", ""))
                if cfg_hash == job.config_hash:
                    result_payload = payload
                    break
                if cfg_hash:
                    result_buffer[cfg_hash] = payload

            if result_payload is not None:
                break

        if result_payload is None:
            logging.error("[Run %s] Timed out waiting for daemon result.", job.config_hash[:12])
            db.mark_failed(job.config_hash)
            return False, False

    status = str(result_payload.get("status", "FAIL")).upper()
    if status != "SUCCESS":
        logging.error("[Run %s] worker_daemon returned %s", job.config_hash[:12], status)
        db.mark_failed(job.config_hash)
        return False, False

    provenance_path = _find_provenance_file(args.provenance_dir, job.config_hash)
    artifact_path = os.path.abspath(
        os.path.join(args.repo_root, str(result_payload.get("artifact_url", f"simulation_data/rho_history_{job.config_hash}.h5")))
    )

    if not provenance_path and os.path.exists(artifact_path):
        logging.warning("[Run %s] Missing provenance from daemon path; running validation fallback.", job.config_hash[:12])
        val_cmd = [
            args.python_executable,
            args.validation_script,
            "--input",
            artifact_path,
            "--params",
            job.params_path,
            "--output_dir",
            args.provenance_dir,
        ]
        val_proc = _run_command(val_cmd, cwd=args.repo_root, timeout_sec=args.validation_timeout_sec)
        if val_proc.returncode != 0:
            db.mark_failed(job.config_hash)
            return False, False
        provenance_path = _find_provenance_file(args.provenance_dir, job.config_hash)

    if not provenance_path:
        db.mark_failed(job.config_hash)
        return False, False

    db.write_success_from_provenance(job, provenance_path)
    _invoke_artifact_gc(args.data_dir, artifact_path, min_age_seconds=args.gc_min_age)
    return True, False

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Strict backlog linear orchestrator")
    parser.add_argument("--repo-root", default=os.getcwd(), help="Repository root (default: cwd).")
    parser.add_argument("--config-dir", default="input_configs", help="Directory containing config JSON files.")
    parser.add_argument(
        "--manifest",
        default="backlog_queue.json",
        help="Optional backlog manifest path (prep_backlog output or custom JSON/CSV).",
    )
    parser.add_argument(
        "--source",
        choices=["auto", "manifest", "directory"],
        default="auto",
        help="Backlog source selection.",
    )
    parser.add_argument(
        "--execution-mode",
        choices=["direct", "daemon"],
        default="direct",
        help="direct=worker+validation subprocesses, daemon=queue->worker_daemon.",
    )
    parser.add_argument("--python-executable", default=sys.executable, help="Python interpreter for subprocess jobs.")
    parser.add_argument("--worker-script", default="worker_cupy.py", help="Worker script for direct mode.")
    parser.add_argument("--validation-script", default="validation_pipeline.py", help="Validation script path.")
    parser.add_argument("--data-dir", default="simulation_data", help="Directory for HDF5 artifacts.")
    parser.add_argument("--provenance-dir", default="provenance_reports", help="Directory for provenance artifacts.")
    parser.add_argument("--db-path", default="simulation_ledger.db", help="SQLite ledger path.")
    parser.add_argument("--generation", type=int, default=0, help="Generation value to write for backlog rows.")
    parser.add_argument("--origin", default="BACKLOG_STRICT", help="Origin label for strict backlog runs.")
    parser.add_argument("--worker-timeout-sec", type=int, default=7200, help="Max seconds for ETDRK4 or daemon result wait.")
    parser.add_argument("--validation-timeout-sec", type=int, default=3600, help="Max seconds for validation.")
    parser.add_argument("--poll-interval", type=float, default=1.0, help="Polling interval for daemon queue waits.")
    parser.add_argument("--worker-heartbeat-ttl-seconds", type=int, default=30, help="Staleness threshold for daemon heartbeats.")
    parser.add_argument("--worker-wait-timeout-sec", type=int, default=300, help="How long to wait for any live worker daemon.")
    parser.add_argument(
        "--gc-min-age",
        "--artifact-gc-min-age-seconds",
        dest="gc_min_age",
        type=int,
        default=int(os.environ.get("ASTE_GC_MIN_AGE_SECONDS", "0")),
        help="Minimum age (seconds) for artifact GC.",
    )
    parser.add_argument(
        "--sqlite-timeout-sec",
        type=float,
        default=float(os.environ.get("ASTE_SQLITE_TIMEOUT", "30.0")),
        help="SQLite connect timeout in seconds.",
    )
    parser.add_argument(
        "--sqlite-busy-timeout-ms",
        type=int,
        default=int(os.environ.get("ASTE_SQLITE_BUSY_TIMEOUT_MS", "30000")),
        help="SQLite PRAGMA busy_timeout in milliseconds.",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    args.repo_root = os.path.abspath(args.repo_root)
    args.config_dir = os.path.abspath(os.path.join(args.repo_root, args.config_dir))
    args.data_dir = os.path.abspath(os.path.join(args.repo_root, args.data_dir))
    args.provenance_dir = os.path.abspath(os.path.join(args.repo_root, args.provenance_dir))
    args.db_path = os.path.abspath(os.path.join(args.repo_root, args.db_path))
    args.manifest = os.path.abspath(os.path.join(args.repo_root, args.manifest))

    _setup_logging(args.verbose)
    os.makedirs(args.config_dir, exist_ok=True)
    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.provenance_dir, exist_ok=True)

    if args.execution_mode == "daemon" and not ORCHESTRATOR_QUEUE_AVAILABLE:
        raise RuntimeError(
            "execution-mode=daemon requires orchestrator.job_manifest and orchestrator.scheduling.queue_manager."
        )

    db = LedgerDB(
        args.db_path,
        sqlite_timeout_sec=args.sqlite_timeout_sec,
        sqlite_busy_timeout_ms=args.sqlite_busy_timeout_ms,
    )
    jobs = _load_backlog_jobs(args)
    if not jobs:
        logging.info("[Backlog] Queue empty. Nothing to run.")
        return

    logging.info(
        "[Backlog] Loaded %d jobs (source=%s, mode=%s).",
        len(jobs),
        args.source,
        args.execution_mode,
    )

    queue = (
        QueueManager(queue_file="backlog_queue.json", result_file="result_queue.json")
        if args.execution_mode == "daemon"
        else None
    )
    result_buffer: Dict[str, Dict[str, Any]] = {}

    started = dt.datetime.utcnow()
    successes = 0
    failures = 0
    killswitch_triggered = False

    for idx, job in enumerate(jobs, start=1):
        logging.info("[Backlog] (%d/%d) %s", idx, len(jobs), job.config_hash[:16])
        db.register_job(job)

        try:
            if args.execution_mode == "daemon":
                ok, kill_flag = _process_job_daemon(job, args, db, queue, result_buffer)
            else:
                ok, kill_flag = _process_job_direct(job, args, db)
        except subprocess.TimeoutExpired as exc:
            logging.error("[Run %s] Timeout: %s", job.config_hash[:12], exc)
            db.mark_failed(job.config_hash)
            ok = False
            kill_flag = False
        except Exception as exc:
            logging.exception("[Run %s] Unhandled error: %s", job.config_hash[:12], exc)
            db.mark_failed(job.config_hash)
            ok = False
            kill_flag = False

        if ok:
            successes += 1
        else:
            failures += 1

        if kill_flag:
            logging.critical("[Backlog] KillSwitch token detected. Stopping strict sweep.")
            killswitch_triggered = True
            break

    elapsed = (dt.datetime.utcnow() - started).total_seconds()
    logging.info(
        "[Backlog] Complete. success=%d failure=%d total=%d elapsed=%.1fs killswitch=%s",
        successes,
        failures,
        successes + failures,
        elapsed,
        killswitch_triggered,
    )


if __name__ == "__main__":
    main()
