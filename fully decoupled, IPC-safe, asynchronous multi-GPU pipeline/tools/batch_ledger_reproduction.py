import csv
import json
import os
import subprocess
import sys
import glob

# --- Configuration ---
LEDGER_FILE = "Simulation_ledgers/simulation_ledger (6).csv"  # Target the main ledger
CONFIG_DIR = "configs/reproduction_batch"
DATA_DIR = "simulation_data/reproduction_batch"
PROV_DIR = "provenance_reports/reproduction_batch"
OUTPUT_REPORT = "reproduction_comparison_report.csv"

def setup_directories():
    os.makedirs(CONFIG_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(PROV_DIR, exist_ok=True)

def get_latest_provenance(config_hash):
    """Finds the generated provenance file for a given hash."""
    search_pattern = os.path.join(PROV_DIR, f"provenance_{config_hash}.json")
    files = glob.glob(search_pattern)
    if not files:
        return None
    return max(files, key=os.path.getctime)

def main():
    setup_directories()
    
    print("--- 🧬 ASTE LEDGER REPRODUCTION ENGINE ---")
    
    if not os.path.exists(LEDGER_FILE):
        print(f"❌ Could not find ledger: {LEDGER_FILE}")
        sys.exit(1)

    # 1. Load and filter the historical ledger
    runs_to_test = []
    with open(LEDGER_FILE, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                hist_sse = float(row.get('log_prime_sse', 999.0))
                # Filter: Only reproduce runs that achieved a sub-1.0 SSE (The Golden Tier)
                if hist_sse < 1.0:
                    runs_to_test.append(row)
            except ValueError:
                continue

    print(f"🔍 Found {len(runs_to_test)} historical 'Golden' runs (SSE < 1.0) to reproduce.")
    
    comparison_results = []

    # 2. Execution Loop
    for idx, run in enumerate(runs_to_test):
        orig_hash = run.get('config_hash', f'unknown_{idx}')
        hist_sse = float(run['log_prime_sse'])
        
        print(f"\n[{idx+1}/{len(runs_to_test)}] Reproducing Run: {orig_hash[:10]} (Historical SSE: {hist_sse:.6f})")

        # Reconstruct the physics parameters exactly as they were in the ledger
        # Note: If you want to test the *new* splash physics on old parameters, 
        # you would inject "param_splash_coupling": 0.15 here.
        config_payload = {
            "param_D": float(run['param_D']),
            "param_eta": float(run['param_eta']),
            "param_rho_vac": float(run['param_rho_vac']),
            "param_a_coupling": float(run['param_a_coupling']),
            "global_seed": 42, # Lock the seed for deterministic comparison
            "simulation": {
                "N_grid": 16,
                "L_domain": 10.0,
                "T_steps": 50,
                "dt": 0.01
            }
        }

        # Save the temporary config
        config_path = os.path.join(CONFIG_DIR, f"config_{orig_hash}.json")
        with open(config_path, 'w') as f:
            json.dump(config_payload, f, indent=2)

        data_path = os.path.join(DATA_DIR, f"rho_{orig_hash}.h5")

        # Execute Worker
        print("   -> Running S-NCGL Physics Engine...")
        worker_cmd = [sys.executable, "worker_unified.py", "--params", config_path, "--output", data_path]
        subprocess.run(worker_cmd, capture_output=True, text=True)

        # Execute Validator
        print("   -> Running CEPP v2.0 Validation...")
        val_cmd = [sys.executable, "validation_pipeline.py", "--input", data_path, "--params", config_path, "--output_dir", PROV_DIR]
        subprocess.run(val_cmd, capture_output=True, text=True)

        # 3. Data Extraction & Comparison
        prov_file = get_latest_provenance(orig_hash)
        new_sse = 999.0
        new_null_a = 999.0
        
        if prov_file:
            with open(prov_file, 'r') as f:
                prov_data = json.load(f)
                try:
                    spec = prov_data.get('spectral_fidelity', {})
                    new_sse = float(spec.get('log_prime_sse', 999.0))
                    new_null_a = float(spec.get('sse_null_phase_scramble', 999.0))
                except Exception as e:
                    print(f"   ⚠️ Could not parse provenance: {e}")

        sse_delta = abs(hist_sse - new_sse)
        status = "MATCH" if sse_delta < 0.01 else "DIVERGED"
        
        print(f"   📊 Result: Old SSE = {hist_sse:.6f} | New SSE = {new_sse:.6f} | Status = {status}")

        comparison_results.append({
            "config_hash": orig_hash,
            "param_D": config_payload["param_D"],
            "historical_sse": hist_sse,
            "reproduced_sse": new_sse,
            "reproduced_null_a": new_null_a,
            "delta": sse_delta,
            "status": status
        })

    # 4. Generate the Comparison Report
    if comparison_results:
        keys = comparison_results[0].keys()
        with open(OUTPUT_REPORT, 'w', newline='') as f:
            dict_writer = csv.DictWriter(f, fieldnames=keys)
            dict_writer.writeheader()
            dict_writer.writerows(comparison_results)
        print(f"\n✅ Batch complete. Comparison report saved to: {OUTPUT_REPORT}")

if __name__ == "__main__":
    main()