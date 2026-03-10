#!/usr/bin/env python3
"""
extract_topology_metrics.py (Root Directory Version)
Used by the Orchestrator to extract TDA metrics from the FINAL frame of an HDF5 run.
"""
import sys
import numpy as np
import argparse
import h5py

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    try:
        # 1. Load the HDF5 file and grab the final stabilized frame
        with h5py.File(args.input, 'r') as f:
            if 'final_rho' in f:
                rho_field = f['final_rho'][:]
            elif 'rho_history' in f:
                rho_field = f['rho_history'][-1]
            else:
                raise KeyError("No valid density data found.")
            
        # Dimensionality Firewall
        if rho_field.ndim == 4:
            rho_field = rho_field[-1]

        # 2. Extract Topology (Persistent Homology)
        from ripser import ripser
        
        threshold = np.mean(rho_field) + np.std(rho_field)
        binary_mask = rho_field > threshold
        coords = np.argwhere(binary_mask)

        # Mathematical Hardening: Prevent Ripser OOM Hangs
        MAX_TDA_POINTS = 1000
        if len(coords) > MAX_TDA_POINTS:
            indices = np.random.choice(len(coords), size=MAX_TDA_POINTS, replace=False)
            coords = coords[indices]

        if len(coords) > 0:
            result = ripser(coords, maxdim=2)
            # Count the topological features (H0=fragments, H1=loops, H2=voids)
            h0_count = len(result['dgms'][0]) if len(result['dgms']) > 0 else 0
            h1_count = len(result['dgms'][1]) if len(result['dgms']) > 1 else 0
            h2_count = len(result['dgms'][2]) if len(result['dgms']) > 2 else 0
        else:
            h0_count, h1_count, h2_count = 0, 0, 0

        # 3. Save the .npz output for the Orchestrator
        np.savez(args.output, h0=h0_count, h1=h1_count, h2=h2_count)
        print(f"Successfully extracted topology to {args.output}")

    except Exception as e:
        print(f"Failed to extract topology: {e}")
        sys.exit(1)