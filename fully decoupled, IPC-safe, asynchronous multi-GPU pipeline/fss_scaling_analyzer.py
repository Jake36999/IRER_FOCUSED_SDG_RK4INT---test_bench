#!/usr/bin/env python3
import sqlite3
import pandas as pd
import numpy as np
import os
import logging
import argparse
import random
import time
from scipy.optimize import curve_fit, minimize

from orchestrator.job_manifest import JobManifest
from orchestrator.scheduling.queue_manager import QueueManager

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
SQLITE_TIMEOUT = float(os.environ.get("ASTE_SQLITE_TIMEOUT", "30.0"))
SQLITE_BUSY_TIMEOUT_MS = int(os.environ.get("ASTE_SQLITE_BUSY_TIMEOUT_MS", "30000"))
DB_MAX_RETRIES = int(os.environ.get("ASTE_DB_MAX_RETRIES", "5"))

"""
Mathematical Model: Local Error Manifold Approximation

We model the primary harmonic error ε as a second-order polynomial surface
over the reduced parameter subspace:

    a  = param_a_coupling
    s  = param_splash_coupling
    ℓ  = log(p), where p is the active prime target

The fitted surface is:

    ε(a, s, ℓ) =
        c0
      + c1·a + c2·s + c3·ℓ
      + c4·a² + c5·s² + c6·ℓ²
      + c7·a·s + c8·a·ℓ + c9·s·ℓ

This is a local quadratic (second-order Taylor) approximation of the
error manifold ε ∈ ℝ³ under the assumption that third-order partial
derivatives are negligible within the sampled basin.

Validity conditions:
- Fit is only epistemically valid locally.
- Requires sufficient sampling density (N ≥ 15).
- Requires bounded covariance (see confidence gate).
"""

def poly_surface(X, c0, c1, c2, c3, c4, c5, c6, c7, c8, c9):
    a, s, lp = X
    return (c0 + c1*a + c2*s + c3*lp + c4*(a**2) + c5*(s**2) + 
            c6*(lp**2) + c7*a*s + c8*a*lp + c9*s*lp)

def get_prime_targets(observed_lp):
    """
    Dynamically infers p_active and p_next from sequence.
    ℓ_current = argmin_ℓ | k_obs - ℓ |
    """
    PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
    # Find prime whose log is closest to the observed spectral alignment coordinate
    current_prime = min(PRIMES, key=lambda p: abs(np.log(p) - observed_lp))
    idx = PRIMES.index(current_prime)
    next_prime = PRIMES[idx + 1] if idx + 1 < len(PRIMES) else current_prime
    return current_prime, next_prime


def _connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, timeout=SQLITE_TIMEOUT)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute(f"PRAGMA busy_timeout={SQLITE_BUSY_TIMEOUT_MS};")
    return conn


def _read_sql_with_retry(query: str, conn: sqlite3.Connection, params=()) -> pd.DataFrame:
    for attempt in range(max(1, DB_MAX_RETRIES)):
        try:
            return pd.read_sql_query(query, conn, params=params)
        except sqlite3.OperationalError as exc:
            is_locked = "database is locked" in str(exc).lower()
            if not is_locked or attempt >= max(1, DB_MAX_RETRIES) - 1:
                raise
            sleep_s = (2 ** attempt) * 0.1 + random.uniform(0.0, 0.1)
            logging.warning(f"[FSS] DB locked; retrying read in {sleep_s:.2f}s")
            time.sleep(sleep_s)
    return pd.DataFrame()

def _fetch_elites(db_path: str, limit: int = 15) -> pd.DataFrame:
    conn = _connect(db_path)

    # Fetch elite runs including the dominant_peak_k coordinate
    query = """
        SELECT p.param_a_coupling, p.param_splash_coupling, m.primary_harmonic_error, m.dominant_peak_k
        FROM runs r
        JOIN parameters p ON r.config_hash = p.config_hash
        JOIN metrics m ON r.config_hash = m.config_hash
        WHERE m.primary_harmonic_error < 1.0
        ORDER BY r.fitness ASC, r.generation DESC
        LIMIT ?
    """
    try:
        df = _read_sql_with_retry(query, conn, params=(int(limit),))
    except Exception as e:
        logging.error(f"Database query failed. Ensure 'dominant_peak_k' column exists: {e}")
        conn.close()
        return pd.DataFrame()
    conn.close()
    return df


def _predict_candidate(df: pd.DataFrame) -> dict:
    if len(df) < 15:
        raise ValueError("Insufficient data for stable 3D surface fit (need >= 15 elite samples).")

    # Dynamic prime inference
    best_row = df.loc[df['primary_harmonic_error'].idxmin()]
    observed_lp = float(best_row['dominant_peak_k'])
    current_prime, next_prime = get_prime_targets(observed_lp)

    a_vals = df['param_a_coupling'].values
    s_vals = df['param_splash_coupling'].values
    lp_vals = np.full(len(df), np.log(current_prime))
    sse_vals = df['primary_harmonic_error'].values

    popt, pcov = curve_fit(poly_surface, (a_vals, s_vals, lp_vals), sse_vals, maxfev=10000)
    param_uncertainty = np.sqrt(np.diag(pcov))
    confidence_score = 1.0 / (1.0 + np.mean(param_uncertainty))

    if confidence_score < 0.65:
        raise ValueError(f"Deduction unreliable (confidence={confidence_score:.3f})")

    logging.info(
        f"Surface Fit Secure (C={confidence_score:.3f}). "
        f"Current Prime: {current_prime}. Predicting Prime {next_prime}..."
    )

    next_lp = np.log(next_prime)
    res = minimize(
        lambda p: poly_surface((p[0], p[1], next_lp), *popt),
        [np.mean(a_vals), np.mean(s_vals)],
        bounds=[(0.1, 1.0), (0.0, 1.0)],
    )

    return {
        "param_D": 1.0,
        "param_eta": 0.65,
        "param_rho_vac": 1.0,
        "param_a_coupling": float(res.x[0]),
        "param_splash_coupling": float(res.x[1]),
        "param_splash_fraction": -0.5,
        "origin": "FSS_PREDICTOR",
        "parent_1": "FSS_PREDICTOR",
        "parent_2": "FSS_PREDICTOR",
        "confidence": float(confidence_score),
    }


def enqueue_fss_candidate(candidate: dict, generation: int) -> str:
    queue_manager = QueueManager(queue_file="backlog_queue.json", result_file="result_queue.json")
    params = dict(candidate)
    params["origin"] = "FSS_PREDICTOR"
    params["generation"] = generation
    manifest = JobManifest.from_params(
        params=params,
        generation=generation,
        seed=0,
        origin="FSS_PREDICTOR",
    )
    queue_manager.push_job(manifest.to_json())
    return manifest.config_hash


def main():
    parser = argparse.ArgumentParser(description="FSS manifold predictor producer")
    parser.add_argument("--generation", type=int, required=True, help="Target generation for injected candidate")
    parser.add_argument("--db", type=str, default="simulation_ledger.db", help="Path to simulation ledger DB")
    args = parser.parse_args()

    db_path = args.db
    if not os.path.exists(db_path):
        logging.warning(f"Ledger not found: {db_path}")
        return

    try:
        df = _fetch_elites(db_path, limit=15)
        if df.empty:
            logging.warning("No elite records available for FSS prediction.")
            return
        candidate = _predict_candidate(df)
        injected_hash = enqueue_fss_candidate(candidate, generation=int(args.generation))
        logging.info(f"FSS predictor injected candidate {injected_hash[:12]} for generation {args.generation}")
    except Exception as e:
        logging.error(f"Surface fit failed: {e}")

if __name__ == "__main__": 
    main()