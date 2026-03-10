"""
tests/test_resilience.py
MANDATE: Inject 'Poison' config (Alpha=1e5) and assert graceful failure.
"""
import subprocess
import sys
import json
import os
import time
from pathlib import Path

# Configuration
WORKER_SCRIPT = "worker_v13_engine.py"
POISON_UUID = "poison_pill_test_001"
CONFIG_PATH = f"configs/{POISON_UUID}.json"
ARTIFACT_PATH = f"temp_artifacts/summary_{POISON_UUID}.json"

def generate_poison_config():
    """Creates a config guaranteed to explode the solver instantly."""
    config = {
        "job_uuid": POISON_UUID,
        "param_D": 1.0,
        "param_epsilon": 0.1,
        "param_alpha": 100000.0, # <--- THE POISON (Immediate Divergence)
        "grid_size": 64,
        "steps": 10
    }
    os.makedirs("configs", exist_ok=True)
    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f)
    print(f"[Setup] Poison config written to {CONFIG_PATH}")

def run_worker():
    """Runs the worker and captures the exit code."""
    print(f"[Action] Executing Worker with Poison...")
    cmd = [sys.executable, WORKER_SCRIPT, "--job_uuid", POISON_UUID, "--config_path", CONFIG_PATH]
    
    # We expect this to FAIL, so check=False
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    print(f"[Result] Worker Exit Code: {result.returncode}")
    print(f"[Result] Stdout tail: {result.stdout[-200:]}")
    return result

def verify_survival():
    """Asserts that the system caught the error."""
    # 1. Check Exit Code (Should be non-zero if we rely on sys.exit(3))
    # OR 0 if we handled it gracefully and wrote a "DIVERGED" summary.
    
    # 2. Check Artifact
    if not os.path.exists(ARTIFACT_PATH):
        print("❌ FAILED: No summary artifact generated. Worker died silently.")
        return False
        
    with open(ARTIFACT_PATH, "r") as f:
        data = json.load(f)
        
    status = data.get("status")
    print(f"[Artifact] Status reported: {status}")
    
    if status in ["DIVERGED", "UNSTABLE", "FAILED"]:
        print("✅ SUCCESS: Divergence correctly caught and reported.")
        return True
    else:
        print(f"❌ FAILED: Incorrect status '{status}' for poison pill.")
        return False

def cleanup():
    if os.path.exists(CONFIG_PATH): os.remove(CONFIG_PATH)
    if os.path.exists(ARTIFACT_PATH): os.remove(ARTIFACT_PATH)

if __name__ == "__main__":
    try:
        generate_poison_config()
        run_worker()
        success = verify_survival()
        if not success:
            sys.exit(1)
    finally:
        cleanup()