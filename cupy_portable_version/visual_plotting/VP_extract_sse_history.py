#!/usr/bin/env python3
"""
extract_sse_history.py
Goal: Extracts the spectral SSE for each frame in a rho_history HDF5 file using the V13 Quantule Profiler.
Outputs sse_history.npy for use with render_single_run.py.
"""
import sys
import os
import h5py
import numpy as np

# --- PATH ENFORCEMENT ---
# Force Python to look in the root directory for the REAL quantulemapper_real.py
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, root_dir)

import quantulemapper_real as qm

def extract_sse(rho_path, out_path):
    print(f"Opening {rho_path} for SSE extraction...")
    with h5py.File(rho_path, 'r') as f:
        rho_history = f['rho_history'][:]

    T = rho_history.shape[0]
    sse_history = np.zeros(T)
    
    print(f"Processing {T} frames...")
    for t in range(T):
        # Pass the 3D frame directly into our V13 hardened Bipartite Matcher
        frame = rho_history[t]
        try:
            results = qm.prime_log_sse(frame)
            sse_history[t] = results.get("log_prime_sse", 999.0)
        except Exception as e:
            print(f"  Frame {t} failed: {e}")
            sse_history[t] = 999.0
            
        if t % 10 == 0 or t == T - 1:
            print(f"  [{t}/{T-1}] SSE: {sse_history[t]:.4f}")

    np.save(out_path, sse_history)
    print(f"SUCCESS: Saved SSE timeline to {out_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to rho_history.h5")
    parser.add_argument("--output", required=True, help="Path to output sse_history.npy")
    args = parser.parse_args()
    
    extract_sse(args.input, args.output)