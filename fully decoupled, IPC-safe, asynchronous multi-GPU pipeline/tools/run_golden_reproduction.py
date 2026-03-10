#!/usr/bin/env python3
"""
run_golden_reproduction.py
Runs the IRER physics engine with the true golden parameters to verify reproduction of the prime-log attractor state.
"""
import os
import sys
import subprocess

GOLDEN_CONFIG = "configs/config_true_golden.json"
WORKER_SCRIPT = "worker_unified.py"
VALIDATOR_SCRIPT = "validation_pipeline.py"
PROVENANCE_DIR = "provenance_reports"

# Output file for simulation
RHO_HISTORY_PATH = "simulation_data/rho_history_true_golden.h5"

# 1. Run the simulation (worker)
worker_command = [
    sys.executable, WORKER_SCRIPT,
    "--params", GOLDEN_CONFIG,
    "--output", RHO_HISTORY_PATH
]
print(f"[Reproduction] Running worker: {' '.join(worker_command)}")
worker_result = subprocess.run(worker_command, capture_output=True, text=True)
if worker_result.returncode != 0:
    print("[Reproduction] Worker failed:")
    print(worker_result.stdout)
    print(worker_result.stderr)
    sys.exit(1)
print("[Reproduction] Worker completed successfully.")

# 2. Run the validator (judge)
validator_command = [
    sys.executable, VALIDATOR_SCRIPT,
    "--input", RHO_HISTORY_PATH,
    "--params", GOLDEN_CONFIG,
    "--output_dir", PROVENANCE_DIR
]
print(f"[Reproduction] Running validator: {' '.join(validator_command)}")
validator_result = subprocess.run(validator_command, capture_output=True, text=True)
if validator_result.returncode != 0:
    print("[Reproduction] Validator failed:")
    print(validator_result.stdout)
    print(validator_result.stderr)
    sys.exit(1)
print("[Reproduction] Validator completed successfully.")

print("\n[Reproduction] ✅ ALL ASSERTIONS PASSED! If you see this, the Golden State has been reproduced.")
