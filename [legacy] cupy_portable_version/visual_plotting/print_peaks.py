import numpy as np
import sys
import os
import json

if len(sys.argv) < 2:
	print("Usage: python print_peaks.py <metrics_file.npz or .json>")
	sys.exit(1)

file_path = sys.argv[1]
thresholds = None
betti_counts = None

if file_path.endswith('.npz'):
	metrics = np.load(file_path)
	thresholds = metrics['thresholds'] if 'thresholds' in metrics else None
	betti_counts = metrics['components'] if 'components' in metrics else None
elif file_path.endswith('.json'):
	with open(file_path, 'r') as jf:
		data = json.load(jf)
		thresholds = np.array(data.get('thresholds')) if 'thresholds' in data else None
		betti_counts = np.array(data.get('components')) if 'components' in data else None
else:
	print("Error: Unsupported file type. Use .npz or .json.")
	sys.exit(2)

if thresholds is None or betti_counts is None:
	print("Error: metrics file must contain 'thresholds' and 'components' arrays.")
	sys.exit(2)

peak_idx = np.argmax(betti_counts)
peak_threshold = thresholds[peak_idx]
peak_betti = betti_counts[peak_idx]

print(f"Peak threshold: {peak_threshold:.4f}, Peak Betti count: {peak_betti}")