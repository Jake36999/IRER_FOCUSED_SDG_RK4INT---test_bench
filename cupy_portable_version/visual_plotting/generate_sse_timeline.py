#!/usr/bin/env python3

"""
generate_sse_timeline.py
Goal: Reads the true golden HDF5 artifact and computes the SSE for every time step,
      creating a temporal map of the attractor basin.
"""

import h5py  # type: ignore
import numpy as np
import sys
import os

# Import your real scientific analysis pipeline
try:
    import quantulemapper_real as qm
except ImportError:
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    try:
        import quantulemapper_real as qm
    except ImportError:
        print("FATAL: Could not import 'quantulemapper_real.py'.", file=sys.stderr)
        print("Ensure it is in the workspace root or visual_plotting directory.", file=sys.stderr)
        sys.exit(1)

def extract_sse_timeline(h5_path: str, output_npy_path: str):
    print(f"Loading golden simulation history from: {h5_path}")
    
    if not os.path.exists(h5_path):
        print(f"Error: Could not find {h5_path}. Did the worker save it here?", file=sys.stderr)
        return

    with h5py.File(h5_path, 'r') as f:
        rho_history = f['rho_history'][:]

    T = rho_history.shape[0]
    sse_history = np.zeros(T)
    
    print(f"Processing {T} time steps for dynamic SSE tracking...")
    
    for t in range(T):
        frame = rho_history[t]
        try:
            # 1. Multi-ray FFT (detrended, windowed)
            k_main, power_main = qm._multi_ray_fft(frame)
            
            # 2. Find the dominant spatial frequencies (peaks)
            peaks_k, _ = qm._find_peaks(k_main, power_main)
            
            # 3. Calculate Real SSE against Prime-Log Targets
            match_result = qm.prime_log_sse(peaks_k, qm.LOG_PRIME_TARGETS)
            
            sse_history[t] = match_result.sse
            
        except Exception as e:
            # If a frame has no valid rays or completely fails, log and assign a high penalty
            print(f"  Step {t}: Failed to compute SSE ({e}). Setting to 999.0")
            sse_history[t] = 999.0
        
        # Simple progress output
        if t % 10 == 0 or t == T - 1:
            print(f"  [{t}/{T-1}] SSE: {sse_history[t]:.4f}")

    # Save the numpy array for the animator to use later
    np.save(output_npy_path, sse_history)
    print(f"\nSuccess! Saved temporal SSE curve to {output_npy_path}")

if __name__ == "__main__":
    # Adjust these paths for the high SSE run
    input_file = "simulation_data/rho_history_high_sse.h5"
    output_file = "simulation_data/sse_history_high_sse.npy"
    
    extract_sse_timeline(input_file, output_file)