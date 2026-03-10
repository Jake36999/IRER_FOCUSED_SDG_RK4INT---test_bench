#!/usr/bin/env python3

import sys
import argparse
import h5py  # type: ignore
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # type: ignore[import]
from typing import Optional

# Graceful import check for scikit-image
try:
    from skimage import measure
except ImportError:
    print("Error: scikit-image is required for marching cubes.")
    print("Please install it by running: pip install scikit-image")
    sys.exit(1)

def plot_final_isosurface(h5_path: str, threshold: float = 1.003, save_path: Optional[str] = None):
    """
    Render a 3D isosurface from the final frame of rho_history using marching cubes.
    """
    print(f"Loading data from: {h5_path}")
    try:
        with h5py.File(h5_path, 'r') as f:
            rho_history = f['rho_history'][:]
            final_frame = rho_history[-1]  # Extract the last 3D state
    except FileNotFoundError:
        print(f"Error: Could not find file at {h5_path}")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading HDF5 file: {e}")
        sys.exit(1)

    print(f"Extracting isosurface at threshold {threshold}...")
    try:
        # Marching cubes isosurface extraction
        verts, faces, normals, values = measure.marching_cubes(final_frame, level=threshold)
    except ValueError as e:
        print(f"Error extracting isosurface (threshold {threshold} might be out of data bounds): {e}")
        sys.exit(1)

    # 3D Plotting
    print("Rendering 3D plot...")
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create the mesh
    mesh = Poly3DCollection(verts[faces], alpha=0.7)
    mesh.set_facecolor('cyan')
    mesh.set_edgecolor('black') # Adds definition to the polygonal edges
    mesh.set_linewidth(0.2)
    
    ax.add_collection3d(mesh)
    
    # Set limits based on grid size
    ax.set_xlim(0, final_frame.shape[0])
    ax.set_ylim(0, final_frame.shape[1])
    ax.set_zlim(0, final_frame.shape[2])
    
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')
    plt.title(f'3D Isosurface: \u03c1 = {threshold}')
    plt.tight_layout()

    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Successfully saved 3D render to: {save_path}")
    else:
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render 3D isosurfaces from ASTE simulation data.")
    
    parser.add_argument(
        "--input", 
        type=str, 
        default="simulation_data/rho_history_true_golden.h5", 
        help="Path to the input HDF5 artifact."
    )
    parser.add_argument(
        "--threshold", 
        type=float, 
        default=1.003, 
        help="Density (\u03c1) threshold for the marching cubes algorithm."
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default=None, 
        help="Optional: Path to save the rendered image (e.g., render_3d.png)."
    )

    args = parser.parse_args()

    plot_final_isosurface(
        h5_path=args.input, 
        threshold=args.threshold, 
        save_path=args.output
    )