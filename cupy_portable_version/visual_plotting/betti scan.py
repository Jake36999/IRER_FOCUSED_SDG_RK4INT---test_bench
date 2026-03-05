import numpy as np
import matplotlib.pyplot as plt
import os

# Path setup
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, 'stabilized_render_metrics.npz')

# Load data
data = np.load(file_path)
n_components = data['n_components']
time_steps = np.arange(len(n_components))

plt.figure(figsize=(10, 5))
plt.plot(time_steps, n_components, color='purple')

# Apply the scaling you requested: 
# Linear between 0-10, logarithmic for values above 10
plt.yscale('symlog', linthresh=10)

plt.title('Topological Fragments (Betti $H_0$ at T=1.6367)')
plt.xlabel('Time Step (Saved Frames)')
plt.ylabel('Fragments')
plt.grid(True, which="both", ls="--", alpha=0.5)

# Optional: Add a line to clearly show where the 10 threshold is
plt.axhline(y=10, color='red', linestyle='--', alpha=0.3, label='Scaling Threshold (10)')

plt.tight_layout()
plt.show()