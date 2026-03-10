#!/usr/bin/env python3
"""
plot_phase_transition.py
Goal: Formally defines the 'Dial Between Worlds' by measuring topological 
      fragmentation (Connected Components / Betti 0) across coherence thresholds.
"""
import sys
import numpy as np
import h5py # type: ignore
import matplotlib.pyplot as plt
from skimage.measure import label

def plot_phase_transition(h5_path):
    print(f"Loading high-res artifact: {h5_path}")
    with h5py.File(h5_path, 'r') as f:
        # Grab the final stabilized frame of the simulation
        final_frame = f['rho_history'][-1]
    
    thresholds = np.linspace(1.0, 2.2, 50)
    components = []
    
    print("Sweeping Coherence Gate Thresholds...")
    for t in thresholds:
        mask = final_frame > t
        _, num_features = label(mask, return_num=True, connectivity=1)
        components.append(num_features)
        
    # Plotting the Phase Transition
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, components, lw=3, color='crimson', marker='o', markersize=4)
    
    # Annotate the Phases based on your visual discovery
    plt.axvspan(1.0, 1.2, color='blue', alpha=0.1, label='Phase I: Supercritical Foam (Sponge)')
    plt.axvspan(1.2, 1.6, color='purple', alpha=0.1, label='Phase II: Coherent Manifold (Geometry)')
    plt.axvspan(1.6, 2.2, color='orange', alpha=0.1, label='Phase III: Localized Particles (Islands)')
    
    plt.title("Topological Phase Transition in S-NCGL Field", fontsize=14, fontweight='bold')
    plt.xlabel("Coherence Gate (Threshold)", fontsize=12)
    plt.ylabel("Fragment Count (Betti $H_0$)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='upper left')
    
    # Extract first 6 chars of the SHA-like number from the input filename
    import os
    base = os.path.basename(h5_path)
    # Find the first 6 hex digits after 'rho_history_'
    import re
    match = re.search(r"rho_history_([0-9a-fA-F]{6})", base)
    suffix = match.group(1) if match else "unknown"
    out_path = f"visual_plotting/phase_transition_curve_{suffix}.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    # Save plot data to JSON
    import json
    json_path = out_path.replace('.png', '_data.json')
    plot_data = {
        "thresholds": [float(x) for x in thresholds],
        "components": [int(x) for x in components]
    }
    with open(json_path, 'w') as jf:
        json.dump(plot_data, jf, indent=2)
    print(f"SUCCESS: Phase transition plotted and saved to {out_path}")
    print(f"Plot data saved to {json_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Path to high-res rho_history.h5')
    args = parser.parse_args()
    
    plot_phase_transition(args.input)