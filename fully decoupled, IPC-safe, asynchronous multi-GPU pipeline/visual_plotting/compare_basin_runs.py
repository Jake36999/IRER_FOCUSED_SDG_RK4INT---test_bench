#!/usr/bin/env python3

"""
compare_basin_runs.py
Goal: Renders a 3-panel animation comparing two runs side-by-side.
      Extracts Coverage, Volume, and Surface Area for graphing.
"""

import sys
import os
import numpy as np
import h5py  # type: ignore

try:
    import pyvista as pv
    from skimage.measure import marching_cubes
except ImportError:
    print("Error: Required libraries missing.")
    sys.exit(1)

def extract_mesh_and_metrics(rho, threshold):
    """Extracts mesh, maps Z-elevation, and calculates physical properties."""
    try:
        verts, faces, normals, values = marching_cubes(rho, level=threshold)
        faces = np.hstack([np.full((faces.shape[0], 1), 3), faces]).astype(np.int64)
        mesh = pv.PolyData(verts, faces)
        mesh["Elevation"] = verts[:, 2]
        return mesh, mesh.volume, mesh.area
    except ValueError:
        return None, 0.0, 0.0

def render_comparison_animation(rho_a, sse_a, rho_b, sse_b, output_gif, threshold=1.003):
    print("Loading datasets for comparison...")
    with h5py.File(rho_a, "r") as f: history_a = f["rho_history"][:]
    with h5py.File(rho_b, "r") as f: history_b = f["rho_history"][:]
    
    sse_history_a = np.load(sse_a)
    sse_history_b = np.load(sse_b)
    T = min(history_a.shape[0], history_b.shape[0])
    
    # Setup 1x3 Grid
    plotter = pv.Plotter(shape=(1, 3), window_size=(2400, 800), off_screen=True)
    plotter.open_gif(output_gif)
    
    # Initialize Panel Backgrounds explicitly! (no all_axes argument)
    plotter.subplot(0, 0)
    plotter.set_background("black")
    plotter.subplot(0, 1)
    plotter.set_background("black")
    plotter.subplot(0, 2)
    plotter.set_background("white")

    # Setup Chart Panel
    chart = pv.Chart2D()
    time_steps = np.arange(T)
    chart.line(time_steps, sse_history_a[:T], color='b', width=2.0, label="Run A")
    chart.line(time_steps, sse_history_b[:T], color='g', width=2.0, label="Run B")
    tracker_a = chart.scatter(np.array([0]), np.array([sse_history_a[0]]), color='blue', size=15)
    tracker_b = chart.scatter(np.array([0]), np.array([sse_history_b[0]]), color='green', size=15)
    chart.x_axis.title = "Time Step"
    chart.y_axis.title = "Spectral SSE"
    plotter.add_chart(chart)

    # Metric Storage Lists
    metrics = {"cov_a": [], "cov_b": [], "vol_a": [], "vol_b": [], "area_a": [], "area_b": []}

    print(f"Generating comparison animation ({T} frames)...")
    for t in range(T):
        # Update Chart Trackers
        plotter.subplot(0, 2)
        tracker_a.update(np.array([t]), np.array([sse_history_a[t]]))
        tracker_b.update(np.array([t]), np.array([sse_history_b[t]]))

        # Update Run A
        plotter.subplot(0, 0)
        plotter.clear_actors()
        mesh_a, vol_a, area_a = extract_mesh_and_metrics(history_a[t], threshold)
        if mesh_a:
            plotter.add_mesh(mesh_a, scalars="Elevation", cmap="plasma", show_edges=True, edge_color="white", opacity=0.9)
            bounds = np.array([0, history_a[t].shape[0], 0, history_a[t].shape[1], 0, history_a[t].shape[2]])
            plotter.reset_camera(bounds=bounds)
        
        # Update Run B
        plotter.subplot(0, 1)
        plotter.clear_actors()
        mesh_b, vol_b, area_b = extract_mesh_and_metrics(history_b[t], threshold)
        if mesh_b:
            plotter.add_mesh(mesh_b, scalars="Elevation", cmap="viridis", show_edges=True, edge_color="white", opacity=0.9)
            bounds = np.array([0, history_b[t].shape[0], 0, history_b[t].shape[1], 0, history_b[t].shape[2]])
            plotter.reset_camera(bounds=bounds)

        # Force render to align subplots before taking a screenshot
        plotter.render()

        # Capture 2D Panes
        img = plotter.screenshot(return_img=True)
        img_a = img[:, :800]
        img_b = img[:, 800:1600]
        
        # Now that backgrounds are black, > 10 correctly maps the mesh!
        cov_a = (np.sum(np.any(img_a > 10, axis=2)) / (800 * 800)) * 100
        cov_b = (np.sum(np.any(img_b > 10, axis=2)) / (800 * 800)) * 100
        
        # Save metrics
        metrics["cov_a"].append(cov_a); metrics["cov_b"].append(cov_b)
        metrics["vol_a"].append(vol_a); metrics["vol_b"].append(vol_b)
        metrics["area_a"].append(area_a); metrics["area_b"].append(area_b)

        # Update HUD Text
        plotter.subplot(0, 0)
        plotter.add_text(f"RUN A\nCov: {cov_a:.1f}%\nVol: {vol_a:.0f}\nSSE: {sse_history_a[t]:.3f}", position="upper_left", font_size=12, color="white", name="hud_a")
        
        plotter.subplot(0, 1)
        plotter.add_text(f"RUN B\nCov: {cov_b:.1f}%\nVol: {vol_b:.0f}\nSSE: {sse_history_b[t]:.3f}", position="upper_left", font_size=12, color="white", name="hud_b")

        plotter.write_frame()
        print(f"  Frame {t}/{T-1} rendered.")

    plotter.close()
    
    # Save the expanded metrics payload
    np.savez(output_gif.replace('.gif', '_metrics.npz'), **metrics)
    print(f"\nSUCCESS! Comparison animation and metrics saved.")

if __name__ == "__main__":
    run_a_rho = "simulation_data/rho_history_true_golden.h5"
    run_a_sse = "simulation_data/sse_history_golden.npy"
    run_b_rho = "simulation_data/rho_history_high_sse.h5"
    run_b_sse = "simulation_data/sse_history_high_sse.npy"
    out_gif = "basin_comparison.gif"
    render_comparison_animation(run_a_rho, run_a_sse, run_b_rho, run_b_sse, out_gif)