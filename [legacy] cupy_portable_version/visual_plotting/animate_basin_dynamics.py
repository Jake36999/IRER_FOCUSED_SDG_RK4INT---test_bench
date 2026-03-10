#!/usr/bin/env python3

"""
animate_basin_dynamics.py
Goal: Renders a dual-panel animation showing 3D structural physics (isosurface)
      syncing exactly with the 1D Spectral Sum of Errors (SSE) over time.
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
    print("Please install via: pip install pyvista scikit-image h5py numpy")
    sys.exit(1)

def render_dual_panel_animation(rho_path, sse_path, output_gif, threshold=1.003, lock_threshold=0.5):
    print(f"Loading 4D density field from: {rho_path}")
    print(f"Loading 1D SSE timeline from: {sse_path}")
    
    if not os.path.exists(rho_path) or not os.path.exists(sse_path):
        print("Error: Input files not found. Did you run Phase 1?")
        sys.exit(1)

    with h5py.File(rho_path, "r") as f:
        rho_history = f["rho_history"][:]
    
    sse_history = np.load(sse_path)
    T = rho_history.shape[0]

    # Initialize PyVista Plotter with 1 row, 2 columns
    plotter = pv.Plotter(shape=(1, 2), window_size=(1600, 800), off_screen=True)
    plotter.open_gif(output_gif)

    # --- SETUP RIGHT PANEL (2D SSE Chart) ---
    plotter.subplot(0, 1)
    plotter.set_background("white")
    chart = pv.Chart2D()
    
    # Plot the full SSE curve in blue
    time_steps = np.arange(T)
    chart.line(time_steps, sse_history, color='b', width=2.0, label="Dynamic SSE")
    
    # Plot the lock threshold line in red (dashed style)
    chart.line(np.array([0, T]), np.array([lock_threshold, lock_threshold]), color='r', style='--', label="Lock Threshold")
    
    # Initialize the moving tracker dot
    tracker_dot = chart.scatter(np.array([0]), np.array([sse_history[0]]), color='darkorange', size=15)
    
    chart.x_axis.title = "Time Step"
    chart.y_axis.title = "Spectral SSE (Log Scale)"
    
    # Optional: Log scale the Y axis if SSE starts at 999.0
    if np.max(sse_history) > 100:
        # PyVista Chart2D does not support 'log' behavior, so we log-transform the data manually
        sse_history_plot = np.log10(np.clip(sse_history, 1e-3, None))
        chart.clear()
        chart.line(time_steps, sse_history_plot, color='b', width=2.0, label="Dynamic SSE (log10)")
        chart.line(np.array([0, T]), np.log10(np.array([lock_threshold, lock_threshold])), color='r', style='--', label="Lock Threshold (log10)")
        tracker_dot = chart.scatter(np.array([0]), np.array([sse_history_plot[0]]), color='darkorange', size=15)
        chart.y_axis.title = "Spectral SSE (log10)"
    else:
        sse_history_plot = sse_history
    # ...existing code...
    
    plotter.add_chart(chart)

    # --- SETUP LEFT PANEL (3D Isosurface) ---
    plotter.subplot(0, 0)
    plotter.set_background("black")
    
    print(f"Generating animation ({T} frames)...")
    
    projected_areas = []  # Store 2D projected area percentage per frame
    for t in range(T):
        # Update Right Panel Tracker
        plotter.subplot(0, 1)
        tracker_dot.update(np.array([t]), np.array([sse_history_plot[t]]))

        # Update Left Panel 3D Geometry
        plotter.subplot(0, 0)
        plotter.clear_actors() # Clear the previous frame's mesh

        rho = rho_history[t]
        try:
            # Extract isosurface
            verts, faces, normals, values = marching_cubes(rho, level=threshold)
            # PyVista requires a specific face array format: [n_points, p1, p2, p3]
            faces = np.hstack([np.full((faces.shape[0], 1), 3), faces]).astype(np.int64)
            mesh = pv.PolyData(verts, faces)

            # Add mesh to the scene
            plotter.add_mesh(mesh, color="cyan", opacity=0.8, show_edges=True, edge_color="blue")

            # Set fixed camera view so the box doesn't jump around
            bounds = np.array([0, rho.shape[0], 0, rho.shape[1], 0, rho.shape[2]])
            plotter.reset_camera(bounds=bounds)

            # --- 2D Projected Area Measurement ---
            img = plotter.screenshot(return_img=True)
            if img is not None:
                # Count non-black pixels (object coverage)
                object_pixels = np.sum(np.any(img > 10, axis=2))
                total_pixels = img.shape[0] * img.shape[1]
                coverage = object_pixels / total_pixels * 100
                projected_areas.append(coverage)
            else:
                projected_areas.append(np.nan)

        except ValueError:
            # If no structure crosses the threshold yet, marching_cubes raises a ValueError
            projected_areas.append(np.nan)
            pass

        # Add dynamic text overlay
        if np.max(sse_history) > 100:
            sse_val = sse_history[t]
            status = "UNLOCKED" if sse_val > lock_threshold else "LOCKED!"
            text_color = "red" if status == "UNLOCKED" else "green"
            plotter.add_text(f"Time: {t} \nSSE: {sse_val:.3f} \nStatus: {status}", 
                             position="upper_left", font_size=14, color=text_color, name="hud")
        else:
            sse_val = sse_history[t]
            status = "UNLOCKED" if sse_val > lock_threshold else "LOCKED!"
            text_color = "red" if status == "UNLOCKED" else "green"
            plotter.add_text(f"Time: {t} \nSSE: {sse_val:.3f} \nStatus: {status}", 
                             position="upper_left", font_size=14, color=text_color, name="hud")

        # Write the combined dual-panel frame
        plotter.write_frame()
        print(f"  Frame {t}/{T-1} rendered.")

    plotter.close()
    print(f"\nSUCCESS! Dual-panel animation saved to: {output_gif}")

    # After animation, save projected area array for analysis
    np.save(output_gif.replace('.gif', '_projected_area.npy'), np.array(projected_areas))
    print(f"Saved per-frame projected area to {output_gif.replace('.gif', '_projected_area.npy')}")

if __name__ == "__main__":
    # Ensure variables match the outputs from Phase 1
    rho_input = "simulation_data/rho_history_true_golden.h5"
    sse_input = "simulation_data/sse_history_golden.npy"
    gif_output = "basin_evolution.gif"
    
    render_dual_panel_animation(rho_input, sse_input, gif_output)