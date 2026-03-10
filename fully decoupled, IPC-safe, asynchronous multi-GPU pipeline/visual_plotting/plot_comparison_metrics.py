#!/usr/bin/env python3

"""
plot_comparison_metrics.py
Goal: Generates a high-detail 4-panel figure comparing SSE, Coverage, Volume, and Area.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys

def plot_run_comparisons(sse_a_path, sse_b_path, metrics_npz_path, output_png="run_comparison_graphs.png"):
    if not os.path.exists(metrics_npz_path):
        print("Error: Metrics file missing. Please run 'compare_basin_runs.py' first.")
        sys.exit(1)

    print("Loading detailed data for visualization...")
    sse_a = np.load(sse_a_path)
    sse_b = np.load(sse_b_path)
    
    metrics = np.load(metrics_npz_path)
    cov_a, cov_b = metrics['cov_a'], metrics['cov_b']
    vol_a, vol_b = metrics['vol_a'], metrics['vol_b']
    area_a, area_b = metrics['area_a'], metrics['area_b']

    T = min(len(sse_a), len(sse_b), len(cov_a), len(cov_b))
    time_steps = np.arange(T)

    # Slice to match shortest array length
    sse_a, sse_b = sse_a[:T], sse_b[:T]
    cov_a, cov_b = cov_a[:T], cov_b[:T]
    vol_a, vol_b = vol_a[:T], vol_b[:T]
    area_a, area_b = area_a[:T], area_b[:T]

    # Create 2x2 grid
    fig, axs = plt.subplots(2, 2, figsize=(16, 10), sharex=True)
    fig.suptitle('Automated Structural Run Comparison: Golden vs Test', fontsize=18, fontweight='bold')

    # Utility function to standardize plot styling
    def format_plot(ax, data_a, data_b, title, ylabel, show_delta=False):
        ax.plot(time_steps, data_a, color='blue', linewidth=2, label='Run A')
        ax.plot(time_steps, data_b, color='green', linewidth=2, linestyle='--', label='Run B')
        if show_delta:
            delta = data_b - data_a
            ax.bar(time_steps, delta, color=['red' if d < 0 else 'limegreen' for d in delta], alpha=0.3, label='Delta (B-A)')
        ax.set_title(title, fontsize=14)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.grid(True, linestyle=':', alpha=0.7)
        ax.legend(loc='best')

    # Top Left: SSE
    format_plot(axs[0, 0], sse_a, sse_b, '1D Spectral SSE (Convergence)', 'Spectral Error')
    
    # Top Right: Volume
    format_plot(axs[0, 1], vol_a, vol_b, '3D Isosurface Volume', 'Voxels^3')
    
    # Bottom Left: Screen Coverage
    format_plot(axs[1, 0], cov_a, cov_b, '2D Rendered Screen Coverage', 'Coverage %', show_delta=True)
    axs[1, 0].set_xlabel('Time Step', fontsize=12)
    
    # Bottom Right: Surface Area
    format_plot(axs[1, 1], area_a, area_b, '3D Surface Area', 'Voxels^2')
    axs[1, 1].set_xlabel('Time Step', fontsize=12)

    plt.tight_layout()
    plt.savefig(output_png, dpi=300)
    print(f"\nSUCCESS! 4-Panel comparison plot saved to: {output_png}")

if __name__ == "__main__":
    SSE_A = "simulation_data/sse_history_golden.npy"
    SSE_B = "simulation_data/sse_history_high_sse.npy" 
    METRICS_NPZ = "basin_comparison_metrics.npz" 
    OUTPUT_IMAGE = "run_comparison_graphs.png"

    plot_run_comparisons(SSE_A, SSE_B, METRICS_NPZ, OUTPUT_IMAGE)