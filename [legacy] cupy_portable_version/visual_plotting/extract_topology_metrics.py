#!/usr/bin/env python3
"""
extract_topology_metrics.py (Upgraded to Unified Dynamics Analyzer)
Goal: Extracts and plots both Topological metrics (Betti, Area, Volume) 
      AND continuous Field Dynamics (Energy, Max Density, Variance) 
      from the 4D simulation history in a single pass.
"""
import sys
import os
import numpy as np
import h5py # type: ignore
import matplotlib.pyplot as plt
from skimage.measure import marching_cubes, label

try:
    import pyvista as pv
    HAS_PYVISTA = True
except ImportError as e:
    print(f"Warning: pyvista import failed. Skipping Volume/Area extraction. Dashboard will still generate.")
    HAS_PYVISTA = False

def compute_and_plot_metrics(rho_path, threshold=1.6367, output_prefix=None):
    print(f"Loading artifact: {rho_path}")
    with h5py.File(rho_path, "r") as f:
        rho_history = f["rho_history"][:]
        
    T = rho_history.shape[0]
    
    # Topology Metrics
    n_components = np.zeros(T, dtype=int)
    surface_area = np.zeros(T)
    volume = np.zeros(T)
    
    # Field Dynamics Metrics
    total_energy = np.zeros(T)
    max_density = np.zeros(T)
    variance = np.zeros(T)
    
    print(f"Extracting unified metrics across {T} frames...")
    for t in range(T):
        rho = rho_history[t]
        
        # 1. Field Dynamics
        total_energy[t] = np.sum(rho)
        max_density[t] = np.max(rho)
        variance[t] = np.var(rho)
        
        # 2. Topology (Connected components / Betti 0)
        mask = rho > threshold
        labeled, num = label(mask, return_num=True, connectivity=1)
        n_components[t] = num
        
        # 3. 3D Mesh properties
        if HAS_PYVISTA:
            try:
                verts, faces, normals, values = marching_cubes(rho, level=threshold)
                faces = np.hstack([np.full((faces.shape[0], 1), 3), faces]).astype(np.int64)
                mesh = pv.PolyData(verts, faces)
                surface_area[t] = mesh.area
                volume[t] = mesh.volume
            except Exception:
                surface_area[t] = 0.0
                volume[t] = 0.0
        else:
            surface_area[t] = 0.0
            volume[t] = 0.0
            
        if t % 25 == 0 or t == T - 1:
            print(f"  Frame {t}/{T-1}: Energy={total_energy[t]:.1f}, MaxDen={max_density[t]:.2f}, Fragments={n_components[t]}")

    if output_prefix:
        # Save Raw Data for downstream use
        npz_out = f"{output_prefix}_metrics.npz"
        np.savez(npz_out, 
                 n_components=n_components, surface_area=surface_area, volume=volume,
                 total_energy=total_energy, max_density=max_density, variance=variance)
        print(f"\nSaved raw metric data to {npz_out}")
        
        # Plot Visual Dashboard
        png_out = f"{output_prefix}_dashboard.png"
        fig, axs = plt.subplots(4, 1, figsize=(12, 16), sharex=True)

        # Plot 1: Energy (Log Scale)
        axs[0].plot(total_energy + 1e-12, color='blue', lw=2)
        axs[0].set_yscale('log')
        axs[0].set_title('Total Resonance Energy (Log Scale $\Sigma \\rho$)', fontweight='bold')
        axs[0].grid(True, linestyle='--', alpha=0.6)

        # Plot 2: Density (Log Scale)
        axs[1].plot(max_density + 1e-12, color='red', lw=2)
        axs[1].set_yscale('log')
        axs[1].set_title('Maximum Local Density (Log Scale $max(\\rho)$)', fontweight='bold')
        axs[1].grid(True, linestyle='--', alpha=0.6)

        # Plot 3: Variance (Log Scale)
        axs[2].plot(variance + 1e-12, color='green', lw=2)
        axs[2].set_yscale('log')
        axs[2].set_title('Spatial Variance (Log Scale $\sigma^2$)', fontweight='bold')
        axs[2].grid(True, linestyle='--', alpha=0.6)

        # Plot 4: Topology (Linear Scale)
        axs[3].plot(n_components, color='purple', lw=2)
        axs[3].set_title(f'Topological Fragments (Betti $H_0$ at T={threshold})', fontweight='bold')
        axs[3].set_xlabel('Time Step (Saved Frames)')
        axs[3].grid(True, linestyle='--', alpha=0.6)

        plt.tight_layout()
        plt.savefig(png_out, dpi=300)
        print(f"Saved visual dashboard to {png_out}")
        
    return n_components, surface_area, volume, total_energy, max_density, variance

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Unified metrics extractor and grapher.")
    parser.add_argument("--input", default="C:\\Users\\jakem\\Documents\\golden_hunter_low_SSE\\simulation_data\\rho_history_golden_test.h5", help="Path to rho_history HDF5")
    parser.add_argument("--threshold", type=float, default=1.6367, help="Isosurface threshold")
    parser.add_argument("--output_prefix", default="visual_plotting/unified_run", help="Prefix for outputs")
    args = parser.parse_args()
    
    compute_and_plot_metrics(args.input, args.threshold, args.output_prefix)