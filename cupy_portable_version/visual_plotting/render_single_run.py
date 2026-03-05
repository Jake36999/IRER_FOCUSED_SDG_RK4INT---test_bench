#!/usr/bin/env python3

"""
render_single_run.py
Goal: Render a high-quality 3D animation of a single run and extract its metrics.
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
    try:
        verts, faces, normals, values = marching_cubes(rho, level=threshold)
        faces = np.hstack([np.full((faces.shape[0], 1), 3), faces]).astype(np.int64)
        mesh = pv.PolyData(verts, faces)
        mesh["Elevation"] = verts[:, 2] # Z-elevation shader
        return mesh, mesh.volume, mesh.area
    except ValueError:
        return None, 0.0, 0.0

def render_run(rho_path, sse_path, run_name, output_gif, threshold=1.5):
    print(f"Rendering {run_name}...")
    with h5py.File(rho_path, "r") as f:
        history = f["rho_history"][:]
    sse_history = np.load(sse_path)
    T = history.shape[0]

    # Ensure 250 frames for rendering
    target_frames = 250
    if T != target_frames:
        indices = np.linspace(0, T - 1, target_frames).astype(int)
        history = history[indices]
        if len(sse_history) == T:
            sse_history = sse_history[indices]
        T = target_frames

    plotter = pv.Plotter(window_size=(800, 800), off_screen=True)
    plotter.set_background("black")
    plotter.open_gif(output_gif)

    metrics = {"coverage": [], "volume": [], "area": []}

    for t in range(T):
        plotter.clear_actors()
        mesh, vol, area = extract_mesh_and_metrics(history[t], threshold)
        
        if mesh:
            plotter.add_mesh(mesh, scalars="Elevation", cmap="plasma", show_edges=True, edge_color="white", opacity=0.9)
            bounds = np.array([0, history[t].shape[0], 0, history[t].shape[1], 0, history[t].shape[2]])
            plotter.reset_camera(bounds=bounds)

        plotter.render()
        
        # Calculate 2D Screen Coverage
        img = plotter.screenshot(return_img=True)
        cov = (np.sum(np.any(img > 10, axis=2)) / (800 * 800)) * 100
        
        metrics["coverage"].append(cov)
        metrics["volume"].append(vol)
        metrics["area"].append(area)

        # HUD
        plotter.add_text(f"{run_name}\nTime: {t}\nCov: {cov:.1f}%\nVol: {vol:.0f}\nSSE: {sse_history[t]:.3f}", 
                         position="upper_left", font_size=12, color="white", name="hud")
        
        plotter.write_frame()
        print(f"  Frame {t}/{T-1} rendered.")

    plotter.close()
    
    # Save metrics
    np.savez(output_gif.replace('.gif', '_metrics.npz'), **metrics)
    print(f"SUCCESS! Render and metrics saved for {run_name}.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--rho', required=True, help='Path to rho_history.h5')
    parser.add_argument('--sse', required=True, help='Path to sse_history.npy')
    parser.add_argument('--name', default='Simulation Run', help='Name overlay')
    parser.add_argument('--out', default='render.gif', help='Output GIF name')
    args = parser.parse_args()
    
    render_run(args.rho, args.sse, args.name, args.out)