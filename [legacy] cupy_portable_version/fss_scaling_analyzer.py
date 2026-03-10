#!/usr/bin/env python3
import sqlite3
import pandas as pd
import numpy as np
import json
import os
import logging
from scipy.optimize import curve_fit, minimize

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

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

def main():
    db_path = "simulation_ledger.db"
    if not os.path.exists(db_path): return

    conn = sqlite3.connect(db_path)
    # Fetch elite runs including the dominant_peak_k coordinate
    query = """
        SELECT p.param_a_coupling, p.param_splash_coupling, m.primary_harmonic_error, m.dominant_peak_k
        FROM runs r
        JOIN parameters p ON r.config_hash = p.config_hash
        JOIN metrics m ON r.config_hash = m.config_hash
        WHERE m.primary_harmonic_error < 1.0
        ORDER BY r.generation DESC LIMIT 100
    """
    try:
        df = pd.read_sql_query(query, conn)
    except Exception as e:
        logging.error(f"Database query failed. Ensure 'dominant_peak_k' column exists: {e}")
        conn.close()
        return
    conn.close()

    if len(df) < 15:
        logging.warning("Insufficient data for stable 3D surface fit (need >= 15 elite samples).")
        return

    # --- DYNAMIC PRIME INFERENCE (FIXED) ---
    # Anchor the manifold slice at the deepest harmonic basin using coordinate data
    best_row = df.loc[df['primary_harmonic_error'].idxmin()]
    observed_lp = float(best_row['dominant_peak_k']) 
    current_prime, next_prime = get_prime_targets(observed_lp)
    # ----------------------------------------

    a_vals = df['param_a_coupling'].values
    s_vals = df['param_splash_coupling'].values
    lp_vals = np.full(len(df), np.log(current_prime))
    sse_vals = df['primary_harmonic_error'].values

    try:
        popt, pcov = curve_fit(poly_surface, (a_vals, s_vals, lp_vals), sse_vals, maxfev=10000)
        
        # [cite_start]CONFIDENCE GATE [cite: 43, 44, 246]
        param_uncertainty = np.sqrt(np.diag(pcov))
        confidence_score = 1.0 / (1.0 + np.mean(param_uncertainty))

        if confidence_score < 0.65:
            logging.warning(f"Deduction Unreliable (Confidence: {confidence_score:.3f}). Aborting.")
            return

        logging.info(f"Surface Fit Secure (C={confidence_score:.3f}). Current Prime: {current_prime}. Predicting Prime {next_prime}...")

        # Solve minimization for the NEXT prime target (Extrapolation)
        next_lp = np.log(next_prime)
        res = minimize(lambda p: poly_surface((p[0], p[1], next_lp), *popt), 
                       [np.mean(a_vals), np.mean(s_vals)], 
                       bounds=[(0.1, 1.0), (0.0, 1.0)])
        
        probe = {
            "param_D": 1.0, "param_eta": 0.65, "param_rho_vac": 1.0,
            "param_a_coupling": float(res.x[0]),
            "param_splash_coupling": float(res.x[1]),
            "param_splash_fraction": -0.5,
            "origin": "PREDICTOR_ENGINE",
            "parent_1": "PREDICTOR_ENGINE",
            "parent_2": "PREDICTOR_ENGINE",
            "confidence": float(confidence_score)
        }

        with open("scaling_probe.json", "w") as f:
            json.dump(probe, f, indent=2)
        logging.info(f"Probe emitted for P={next_prime}: a={res.x[0]:.4f}, splash={res.x[1]:.4f}")

    except Exception as e:
        logging.error(f"Surface fit failed: {e}")

if __name__ == "__main__": 
    main()