#!/usr/bin/env python3
"""
dual_evolution_comparison.py
Goal: Render a 2x2 panel animation for visual comparison of two runs (3D isosurface + SSE for each),
      and compute automated comparison metrics per frame.
"""
import sys
import os
import numpy as np
import h5py  # type: ignore
import pyvista as pv
from skimage.measure import marching_cubes

# --- Helper to load data ---
def load_run(rho_path, sse_path):
    with h5py.File(rho_path, "r") as f:
        rho_history = f["rho_history"][:]
    sse_history = np.load(sse_path)
    return rho_history, sse_history

# --- Main function ---
def render_dual_evolution(
    rho1_path, sse1_path, rho2_path, sse2_path, output_gif,
    threshold=1.003, lock_threshold=0.5
):
    rho1, sse1 = load_run(rho1_path, sse1_path)
    rho2, sse2 = load_run(rho2_path, sse2_path)
    T = min(rho1.shape[0], rho2.shape[0])
    # Setup 2x2 PyVista plotter
    plotter = pv.Plotter(shape=(2, 2), window_size=(1600, 1600), off_screen=True)
    plotter.open_gif(output_gif)
    # Prepare metrics
    proj_area1, proj_area2 = [], []
    sse_diff = []
    # --- Precompute chart data ---
    time_steps = np.arange(T)
    # --- Animation loop ---
    for t in range(T):
        # --- Top Left: Run 1 3D ---
        plotter.subplot(0, 0)
        plotter.clear_actors()
        try:
            verts, faces, normals, values = marching_cubes(rho1[t], level=threshold)
            faces = np.hstack([np.full((faces.shape[0], 1), 3), faces]).astype(np.int64)
            mesh = pv.PolyData(verts, faces)
            plotter.add_mesh(mesh, color="cyan", opacity=0.8, show_edges=True, edge_color="blue")
            bounds = np.array([0, rho1.shape[1], 0, rho1.shape[2], 0, rho1.shape[3]])
            plotter.reset_camera(bounds=bounds)
            img1 = plotter.screenshot(return_img=True)
            if img1 is not None:
                object_pixels = np.sum(np.any(img1 > 10, axis=2))
                total_pixels = img1.shape[0] * img1.shape[1]
                proj_area1.append(object_pixels / total_pixels * 100)
            else:
                proj_area1.append(np.nan)
        except Exception:
            proj_area1.append(np.nan)
        plotter.add_text(f"Run 1\nTime: {t}\nSSE: {sse1[t]:.3f}", position="upper_left", font_size=14, color="white", name="hud1")
        # --- Top Right: Run 1 SSE ---
        plotter.subplot(0, 1)
        plotter.clear_actors()
        chart1 = pv.Chart2D()
        chart1.line(time_steps, sse1, color='b', width=2.0, label="SSE")
        chart1.line(np.array([0, T]), np.array([lock_threshold, lock_threshold]), color='r', style='--', label="Lock Threshold")
        chart1.scatter(np.array([t]), np.array([sse1[t]]), color='darkorange', size=15)
        chart1.x_axis.title = "Time Step"
        chart1.y_axis.title = "Spectral SSE"
        plotter.add_chart(chart1)
        # --- Bottom Left: Run 2 3D ---
        plotter.subplot(1, 0)
        plotter.clear_actors()
        try:
            verts, faces, normals, values = marching_cubes(rho2[t], level=threshold)
            faces = np.hstack([np.full((faces.shape[0], 1), 3), faces]).astype(np.int64)
            mesh = pv.PolyData(verts, faces)
            plotter.add_mesh(mesh, color="magenta", opacity=0.8, show_edges=True, edge_color="purple")
            bounds = np.array([0, rho2.shape[1], 0, rho2.shape[2], 0, rho2.shape[3]])
            plotter.reset_camera(bounds=bounds)
            img2 = plotter.screenshot(return_img=True)
            if img2 is not None:
                object_pixels = np.sum(np.any(img2 > 10, axis=2))
                total_pixels = img2.shape[0] * img2.shape[1]
                proj_area2.append(object_pixels / total_pixels * 100)
            else:
                proj_area2.append(np.nan)
        except Exception:
            proj_area2.append(np.nan)
        plotter.add_text(f"Run 2\nTime: {t}\nSSE: {sse2[t]:.3f}", position="upper_left", font_size=14, color="white", name="hud2")
        # --- Bottom Right: Run 2 SSE ---
        plotter.subplot(1, 1)
        plotter.clear_actors()
        chart2 = pv.Chart2D()
        chart2.line(time_steps, sse2, color='b', width=2.0, label="SSE")
        chart2.line(np.array([0, T]), np.array([lock_threshold, lock_threshold]), color='r', style='--', label="Lock Threshold")
        chart2.scatter(np.array([t]), np.array([sse2[t]]), color='darkorange', size=15)
        chart2.x_axis.title = "Time Step"
        chart2.y_axis.title = "Spectral SSE"
        plotter.add_chart(chart2)
        # --- Automated Metrics ---
        sse_diff.append(abs(sse1[t] - sse2[t]))
        # --- Write frame ---
        plotter.write_frame()
        print(f"Frame {t}/{T-1} rendered.")
    plotter.close()
    # Save metrics
    np.savez(output_gif.replace('.gif', '_comparison_metrics.npz'),
             proj_area1=np.array(proj_area1), proj_area2=np.array(proj_area2), sse_diff=np.array(sse_diff))
    print(f"Saved comparison metrics to {output_gif.replace('.gif', '_comparison_metrics.npz')}")
    print(f"SUCCESS! Dual evolution animation saved to: {output_gif}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Dual evolution animation and metrics.")
    parser.add_argument('--rho1', required=True, help='Run 1 rho_history.h5')
    parser.add_argument('--sse1', required=True, help='Run 1 sse_history.npy')
    parser.add_argument('--rho2', required=True, help='Run 2 rho_history.h5')
    parser.add_argument('--sse2', required=True, help='Run 2 sse_history.npy')
    parser.add_argument('--output', default='dual_evolution_comparison.gif', help='Output GIF/MP4 file')
    parser.add_argument('--threshold', type=float, default=1.003)
    parser.add_argument('--lock_threshold', type=float, default=0.5)
    args = parser.parse_args()
    render_dual_evolution(args.rho1, args.sse1, args.rho2, args.sse2, args.output, args.threshold, args.lock_threshold)
