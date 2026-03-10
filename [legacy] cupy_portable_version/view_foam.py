import numpy as np
import h5py
import pyvista as pv
from skimage.measure import marching_cubes

print("Loading Golden Run...")
with h5py.File("simulation_data/golden_run_recreation.h5", "r") as f:
    rho = f["final_rho"][:]

print("Extracting 3D Surface...")
threshold = np.mean(rho) + np.std(rho)
verts, faces, normals, values = marching_cubes(rho, level=threshold)

# Format for PyVista
faces = np.hstack([np.full((faces.shape[0], 1), 3), faces]).astype(np.int64)
mesh = pv.PolyData(verts, faces)
mesh["Elevation"] = verts[:, 2] # Color by Z-height

print("Rendering...")
plotter = pv.Plotter()
plotter.add_mesh(mesh, scalars="Elevation", cmap="plasma", show_edges=False, smooth_shading=True)
plotter.add_text("Golden Run: ln(2) Quantum Foam\n(1000 droplets, 425 loops, 93 voids)", color="white", font_size=12)
plotter.set_background("black")
plotter.show()