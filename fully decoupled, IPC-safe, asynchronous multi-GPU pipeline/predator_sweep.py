#!/usr/bin/env python3
import os
import sys
import json
import argparse
import logging
import random
import time
import numpy as np
import sqlite3

from orchestrator.job_manifest import JobManifest
from orchestrator.scheduling.queue_manager import QueueManager

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

"""
Predator Mode: Deterministic Local Basin Exploitation
Refinement sweep for discovered resonance basins using a Truncated Normal kernel.
"""

# Tuning Constants
MICRO_POP_SIZE = 15
DRIFT_PCT = 0.01  # Hard 1% boundary
SQLITE_TIMEOUT = float(os.environ.get("ASTE_SQLITE_TIMEOUT", "30.0"))
SQLITE_BUSY_TIMEOUT_MS = int(os.environ.get("ASTE_SQLITE_BUSY_TIMEOUT_MS", "30000"))
DB_MAX_RETRIES = int(os.environ.get("ASTE_DB_MAX_RETRIES", "5"))


def _connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, timeout=SQLITE_TIMEOUT)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute(f"PRAGMA busy_timeout={SQLITE_BUSY_TIMEOUT_MS};")
    return conn


def _execute_with_retry(cursor: sqlite3.Cursor, query: str, params=()) -> None:
    for attempt in range(max(1, DB_MAX_RETRIES)):
        try:
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            return
        except sqlite3.OperationalError as exc:
            is_locked = "database is locked" in str(exc).lower()
            if not is_locked or attempt >= max(1, DB_MAX_RETRIES) - 1:
                raise
            sleep_s = (2 ** attempt) * 0.1 + random.uniform(0.0, 0.1)
            logging.warning(f"[PREDATOR_SWEEP] DB locked; retrying in {sleep_s:.2f}s")
            time.sleep(sleep_s)

def generate_micro_mutation(val: float) -> float:
    """Applies a strict Truncated Normal distribution bounded to a 1% drift."""
    if val == 0.0: return 0.0
    max_drift = abs(val) * DRIFT_PCT
    std_dev = max_drift / 3.0
    mutated = np.random.normal(loc=val, scale=std_dev)
    return float(np.clip(mutated, val - max_drift, val + max_drift))

def load_target_params(db_path: str, target_hash: str) -> dict:
    conn = _connect(db_path)
    try:
        cursor = conn.cursor()
        _execute_with_retry(
            cursor,
            """
            SELECT param_D, param_eta, param_rho_vac, param_a_coupling, param_splash_coupling, param_splash_fraction
            FROM parameters
            WHERE config_hash = ?
            LIMIT 1
            """,
            (target_hash,),
        )
        row = cursor.fetchone()
    finally:
        conn.close()
    if row is None:
        raise ValueError(f"Target hash not found in parameters table: {target_hash}")
    return {
        "param_D": float(row[0]) if row[0] is not None else 1.0,
        "param_eta": float(row[1]) if row[1] is not None else 0.65,
        "param_rho_vac": float(row[2]) if row[2] is not None else 1.0,
        "param_a_coupling": float(row[3]) if row[3] is not None else 0.1,
        "param_splash_coupling": float(row[4]) if row[4] is not None else 0.1,
        "param_splash_fraction": float(row[5]) if row[5] is not None else -0.5,
    }


def enqueue_mutation_batch(target_hash: str, base_params: dict, batch_size: int) -> int:
    queue_manager = QueueManager(queue_file="backlog_queue.json", result_file="result_queue.json")
    injected = 0

    for seed_idx in range(batch_size):
        candidate = dict(base_params)
        if "param_a_coupling" in candidate:
            candidate["param_a_coupling"] = generate_micro_mutation(float(candidate["param_a_coupling"]))
        if "param_splash_coupling" in candidate:
            candidate["param_splash_coupling"] = generate_micro_mutation(float(candidate["param_splash_coupling"]))

        candidate["origin"] = "PREDATOR_SWEEP"
        candidate["parent_1"] = target_hash
        candidate["parent_2"] = target_hash

        manifest = JobManifest.from_params(
            params=candidate,
            generation=-1,
            seed=seed_idx,
            origin="PREDATOR_SWEEP",
        )
        queue_manager.push_job(manifest.to_json())
        injected += 1

    return injected

def main():
    parser = argparse.ArgumentParser(description="Predator micro-sweep producer")
    parser.add_argument('--target_hash', type=str, required=True, help="Golden config_hash to exploit")
    parser.add_argument('--db', type=str, default="simulation_ledger.db", help="Path to simulation ledger DB")
    parser.add_argument('--batch-size', type=int, default=MICRO_POP_SIZE, help="Number of micro-mutations to enqueue")
    args = parser.parse_args()

    if not os.path.exists(args.db):
        logging.error(f"Ledger not found: {args.db}")
        return 1

    try:
        np.random.seed(int(abs(hash(args.target_hash)) % (2**32)))
        target_params = load_target_params(args.db, args.target_hash)
        injected = enqueue_mutation_batch(args.target_hash, target_params, int(args.batch_size))
        logging.info(f"🦅 [PREDATOR_SWEEP] Enqueued {injected} micro-mutations for {args.target_hash[:12]}")
        return 0
    except Exception as exc:
        logging.error(f"Predator producer failed: {exc}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
