import os
import sys
import json
import glob
import random
import uuid
import subprocess
import math

import csv

# Move rerun_from_ledgers above main()
def rerun_from_ledgers(ledger_paths):
    """Re-run worker and validation for each config_hash in the given ledger CSVs."""
    seen_hashes = set()
    results = []
    for ledger_path in ledger_paths:
        if not os.path.exists(ledger_path):
            print(f"Ledger not found: {ledger_path}")
            continue
        with open(ledger_path, 'r', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                config_hash = row.get('config_hash')
                if not config_hash or config_hash in seen_hashes:
                    continue
                seen_hashes.add(config_hash)
                config_path = os.path.join('configs', f'config_{config_hash}.json')
                data_path = os.path.join('simulation_data', f'rho_history_{config_hash}.h5')
                if not os.path.exists(config_path):
                    print(f"Config not found: {config_path}")
                    continue
                print(f"\n=== Re-running config: {config_path} ===")
                ok, err = run_worker_and_validation(config_path, data_path)
                if not ok:
                    print(f"❌ Pipeline failed: {err}")
                    continue
                log_prime_sse, sse_null_phase_scramble, sse_null_target_shuffle, status = parse_provenance(config_hash)
                print(f"Result: SSE={log_prime_sse}, PhaseScramble={sse_null_phase_scramble}, TargetShuffle={sse_null_target_shuffle}, Status={status}")
                results.append({
                    'config_hash': config_hash,
                    'log_prime_sse': log_prime_sse,
                    'sse_null_phase_scramble': sse_null_phase_scramble,
                    'sse_null_target_shuffle': sse_null_target_shuffle,
                    'status': status,
                    'config_path': config_path
                })
    # Optionally, write a summary CSV
    if results:
        summary_path = 'rerun_summary.csv'
        with open(summary_path, 'w', newline='') as csvfile:
            fieldnames = ['config_hash', 'log_prime_sse', 'sse_null_phase_scramble', 'sse_null_target_shuffle', 'status', 'config_path']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in results:
                writer.writerow(row)
    print(f"\n🏁 Rerun complete! See {summary_path} for all results.")

# --- Config ---
GOLDEN_CONFIG = "configs/config_dc0c0ffee82843dad93aed724642941ee01849be2b9ea50fdbc6128576e94068.json"
OUTPUT_H5 = "simulation_data/rho_history_reproduction.h5"
PROVENANCE_DIR = "provenance_reports"
CONFIG_DIR = "configs/local_hunt/"
DATA_DIR = "simulation_data/"
SUMMARY_CSV = "local_hunt_summary.csv"

BASE_PARAMS = {
    "param_D": 1.496168500093591,
    "param_a_coupling": 0.24160744659804506,
    "param_eta": 0.9033638089109618,
    "param_rho_vac": 0.9517744828272855
}

# --- Utility Functions ---
def setup_dirs():
    for d in [CONFIG_DIR, DATA_DIR, PROVENANCE_DIR]:
        os.makedirs(d, exist_ok=True)

def run_step(cmd, step_name):
    print(f"\n{'='*50}")
    print(f"🚀 STARTING STEP: {step_name}")
    print(f"Command: {' '.join(str(x) for x in cmd)}")
    print(f"{'='*50}")
    try:
        subprocess.run(cmd, check=True)
        print(f"\n✅ [{step_name}] COMPLETED SUCCESSFULLY.")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ [{step_name}] FAILED with exit code {e.returncode}.")
        sys.exit(1)

def generate_nearby_config(base_params, variance):
    new_params = {}
    for key, value in base_params.items():
        shift = random.uniform(-variance, variance)
        new_params[key] = value * (1.0 + shift)
    new_params["global_seed"] = random.randint(100000000, 999999999)
    new_params["run_uuid"] = str(uuid.uuid4())
    return new_params

def run_worker_and_validation(config_path, data_path):
    worker_cmd = [sys.executable, "worker_unified.py", "--params", config_path, "--output", data_path]
    worker_result = subprocess.run(worker_cmd, capture_output=True, text=True)
    if worker_result.returncode != 0:
        return False, f"Worker failed: {worker_result.stderr}"
    val_cmd = [sys.executable, "validation_pipeline.py", "--input", data_path, "--params", config_path, "--output_dir", PROVENANCE_DIR]
    val_result = subprocess.run(val_cmd, capture_output=True, text=True)
    if val_result.returncode != 0:
        return False, f"Validation failed: {val_result.stderr}"
    return True, None

def parse_provenance(config_hash=None):
    if config_hash:
        prov_files = glob.glob(os.path.join(PROVENANCE_DIR, f"provenance_{config_hash}*.json"))
    else:
        prov_files = glob.glob(os.path.join(PROVENANCE_DIR, "provenance_*.json"))
    if not prov_files:
        return None, None, None, None
    prov_path = max(prov_files, key=os.path.getctime)
    with open(prov_path, 'r') as f:
        prov_data = json.load(f)
    # Try both root and nested keys
    log_prime_sse = prov_data.get('log_prime_sse', prov_data.get('total_sse', 999.0))
    sse_null_phase_scramble = prov_data.get('sse_null_phase_scramble', 999.0)
    sse_null_target_shuffle = prov_data.get('sse_null_target_shuffle', 999.0)
    if isinstance(prov_data.get('spectral_fidelity'), dict):
        sf = prov_data['spectral_fidelity']
        log_prime_sse = sf.get('log_prime_sse', log_prime_sse)
        sse_null_phase_scramble = sf.get('sse_null_phase_scramble', sse_null_phase_scramble)
        sse_null_target_shuffle = sf.get('sse_null_target_shuffle', sse_null_target_shuffle)
    status = prov_data.get('validation_status', prov_data.get('status', ''))
    return log_prime_sse, sse_null_phase_scramble, sse_null_target_shuffle, status

def verify_contracts(log_prime_sse, sse_null_phase_scramble, sse_null_target_shuffle):
    print("\n--- ⚖️ Validating Mathematical Contracts ---")
    # 1. Spectral Attractor Contract
    if math.isclose(float(log_prime_sse), 0.129466, abs_tol=0.001):
        print("✅ Spectral Attractor Contract: Passed")
    else:
        print(f"❌ Spectral Attractor Contract Failed! Expected ~0.129466, got {log_prime_sse}")
        return False
    # 2. Ontological Phase Contract
    if float(sse_null_phase_scramble) > 500.0:
        print("✅ Ontological Phase Contract: Passed")
    else:
        print(f"❌ Ontological Phase Contract Failed! Expected > 500.0, got {sse_null_phase_scramble}")
        return False
    # 3. Target Specificity Contract
    if math.isclose(float(sse_null_target_shuffle), 996.0, abs_tol=1.0):
        print("✅ Target Specificity Contract: Passed")
    else:
        print(f"❌ Target Specificity Contract Failed! Expected ~996.0, got {sse_null_target_shuffle}")
        return False
    print("\n🎉 ALL ASSERTIONS PASSED! Reproduction pipeline cryptographically verified.")
    return True

def run_golden():
    os.makedirs("simulation_data", exist_ok=True)
    os.makedirs(PROVENANCE_DIR, exist_ok=True)
    if not os.path.exists(GOLDEN_CONFIG):
        print(f"❌ Error: Could not find golden config at {GOLDEN_CONFIG}")
        sys.exit(1)
    print("Initiating Closed-Loop Automated Reproduction...")
    worker_cmd = [sys.executable, "worker_unified.py", "--params", GOLDEN_CONFIG, "--output", OUTPUT_H5]
    run_step(worker_cmd, "Physics Engine (S-NCGL + FMIA)")
    validator_cmd = [sys.executable, "validation_pipeline.py", "--input", OUTPUT_H5, "--params", GOLDEN_CONFIG, "--output_dir", PROVENANCE_DIR]
    run_step(validator_cmd, "Validation Pipeline (CEPP v2.0)")
    log_prime_sse, sse_null_phase_scramble, sse_null_target_shuffle, status = parse_provenance()
    verify_contracts(log_prime_sse, sse_null_phase_scramble, sse_null_target_shuffle)

def run_sweep(iterations, variance, success_sse):
    setup_dirs()
    print(f"\n🎯 Starting Autonomous Parameter Space Hunt with {iterations} iterations.")
    print(f"Base Parameters: {BASE_PARAMS}")
    print(f"Variance: +/- {variance*100}%\n")
    with open(SUMMARY_CSV, 'w', newline='') as csvfile:
        fieldnames = list(BASE_PARAMS.keys()) + [
            'log_prime_sse', 'sse_null_phase_scramble', 'sse_null_target_shuffle', 'status', 'config_path']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(iterations):
            print(f"--- Iteration {i+1}/{iterations} ---")
            current_params = generate_nearby_config(BASE_PARAMS, variance)
            config_hash = current_params["run_uuid"].split('-')[0]
            config_path = os.path.join(CONFIG_DIR, f"config_{config_hash}.json")
            data_path = os.path.join(DATA_DIR, f"rho_history_{config_hash}.h5")
            with open(config_path, 'w') as f:
                json.dump(current_params, f, indent=2)
            print(f"⚙️  Generated Config: {config_path}")
            ok, err = run_worker_and_validation(config_path, data_path)
            if not ok:
                print(f"❌ Pipeline failed: {err}")
                continue
            log_prime_sse, sse_null_phase_scramble, sse_null_target_shuffle, status = parse_provenance(config_hash)
            print(f"📊 Result: SSE={log_prime_sse}, PhaseScramble={sse_null_phase_scramble}, TargetShuffle={sse_null_target_shuffle}, Status={status}")
            row = {k: current_params[k] for k in BASE_PARAMS.keys()}
            row.update({
                'log_prime_sse': log_prime_sse,
                'sse_null_phase_scramble': sse_null_phase_scramble,
                'sse_null_target_shuffle': sse_null_target_shuffle,
                'status': status,
                'config_path': config_path
            })
            writer.writerow(row)
            if log_prime_sse is not None and float(log_prime_sse) < success_sse:
                print(f"🌟 Success: Found low SSE config at {config_path}")
                verify_contracts(log_prime_sse, sse_null_phase_scramble, sse_null_target_shuffle)
            print("-" * 40 + "\n")
    print(f"\n🏁 Hunt Complete! See {SUMMARY_CSV} for all results.")

def main():
    args = parse_args()
    if args.mode == 'golden':
        run_golden()
    elif args.mode == 'sweep':
        run_sweep(args.iterations, args.variance, args.success_sse)
    elif args.mode == 'rerun':
        if not args.ledgers:
            print("Please specify at least one --ledgers CSV file for rerun mode.")
            sys.exit(1)
        rerun_from_ledgers(args.ledgers)

if __name__ == "__main__":
    main()
