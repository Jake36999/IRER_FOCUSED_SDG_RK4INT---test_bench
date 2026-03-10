#!/usr/bin/env python3
import os
import sys
import json
import argparse
import subprocess
import copy
import logging
import numpy as np
from config_utils import generate_canonical_hash
import aste_hunter

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

"""
Predator Mode: Deterministic Local Basin Exploitation
Refinement sweep for discovered resonance basins using a Truncated Normal kernel.
"""

# Tuning Constants
MICRO_GENS = 10
MICRO_POP_SIZE = 15
DRIFT_PCT = 0.01  # Hard 1% boundary
STALL_TOLERANCE = 1e-5
MAX_STALLS = 3
BRAGG_COLLAPSE_THRESHOLD = 0.5

def generate_micro_mutation(val: float) -> float:
    """Applies a strict Truncated Normal distribution bounded to a 1% drift."""
    if val == 0.0: return 0.0
    max_drift = abs(val) * DRIFT_PCT
    std_dev = max_drift / 3.0
    mutated = np.random.normal(loc=val, scale=std_dev)
    return float(np.clip(mutated, val - max_drift, val + max_drift))

def cleanup_h5_for_hash(config_hash: str, data_dir: str = "simulation_data") -> None:
    if not os.path.exists(data_dir): return
    for name in os.listdir(data_dir):
        if name.endswith('.h5') and config_hash in name:
            try:
                os.remove(os.path.join(data_dir, name))
            except OSError: pass

def run_single_candidate(params: dict, working_dir: str) -> dict:
    config_hash = generate_canonical_hash(params)
    params_path = os.path.join(working_dir, f"micro_{config_hash}.json")
    out_h5 = os.path.join("simulation_data", f"rho_history_{config_hash}.h5")
    
    with open(params_path, 'w') as f:
        json.dump(params, f, indent=2)

    try:
        # [cite_start]Sequential GPU Dispatch [cite: 94]
        subprocess.run([sys.executable, "worker_cupy.py", "--params", params_path, "--output", out_h5], 
                       check=True, stdout=subprocess.DEVNULL)
        subprocess.run([sys.executable, "validation_pipeline.py", "--input", out_h5, "--params", params_path, "--output_dir", "provenance_reports"], 
                       check=True, stdout=subprocess.DEVNULL)
        
        prov_path = os.path.join("provenance_reports", f"provenance_{config_hash}.json")
        if os.path.exists(prov_path):
            with open(prov_path, 'r') as f:
                prov = json.load(f)
            spec = prov.get("spectral_fidelity", {})
            return {
                "hash": config_hash,
                "error": float(spec.get("primary_harmonic_error", 999.0)),
                "bragg": float(spec.get("bragg_lattice_sse", 999.0))
            }
    except Exception as e:
        logging.error(f"Micro-run failure for {config_hash[:8]}: {e}")
    finally:
        cleanup_h5_for_hash(config_hash)
        if os.path.exists(params_path): os.remove(params_path)
    return {"hash": config_hash, "error": 999.0, "bragg": 999.0}

def main():
    parser = argparse.ArgumentParser(description="Predator micro-sweep exploitation")
    parser.add_argument('--target', type=str, required=True, help="Path to JSON payload")
    args = parser.parse_args()

    os.makedirs("input_configs", exist_ok=True)
    os.makedirs("simulation_data", exist_ok=True)

    with open(args.target, 'r') as f:
        payload = json.load(f)

    champ_params = payload["champion_params"]
    best_error = payload["primary_harmonic_error"]
    
    # SEED CONTROL: Lock PRNG for deterministic exploitation
    np.random.seed(payload.get("global_seed", 42))
    
    logging.info(f"🦅 [PREDATOR] Starting Sweep from Error: {best_error:.5f}")
    hunter = aste_hunter.Hunter()
    stall_count = 0

    for gen in range(MICRO_GENS):
        print(f"--- Micro-Gen {gen+1}/{MICRO_GENS} ---")
        best_cand_params = copy.deepcopy(champ_params) # Fail-soft initialization
        candidates = []
        
        # 1. Generate Population using Truncated Normal
        for _ in range(MICRO_POP_SIZE):
            candidate = copy.deepcopy(champ_params)
            # [cite_start]Dimensional Isolation: Mutate ONLY coupling variables [cite: 102, 208]
            if "param_a_coupling" in candidate:
                candidate["param_a_coupling"] = generate_micro_mutation(candidate["param_a_coupling"])
            if "param_splash_coupling" in candidate:
                candidate["param_splash_coupling"] = generate_micro_mutation(candidate["param_splash_coupling"])
            candidates.append(candidate)

        # 2. Evaluate Candidates
        gen_best_res = None
        for cand in candidates:
            res = run_single_candidate(cand, "input_configs")
            
            # --- INTEGRITY GUARD: ABORT ON STRUCTURAL COLLAPSE ---
            if res["bragg"] <= BRAGG_COLLAPSE_THRESHOLD:
                logging.warning(f"🛑 STRUCTURAL COLLAPSE DETECTED (Bragg SSE: {res['bragg']:.3f}). Aborting Sweep.")
                return 0
            # -----------------------------------------------------

            if gen_best_res is None or res["error"] < gen_best_res["error"]:
                gen_best_res = res
                best_cand_params = cand

        # [cite_start]3. Stall Epistemology [cite: 219]
        if gen_best_res and (best_error - gen_best_res["error"]) > STALL_TOLERANCE:
            logging.info(f"✅ Deepened Lock: {best_error:.5f} -> {gen_best_res['error']:.5f}")
            best_error = gen_best_res["error"]
            champ_params = copy.deepcopy(best_cand_params)
            stall_count = 0
        else:
            stall_count += 1
            logging.warning(f"⚠️ Stall detected ({stall_count}/{MAX_STALLS})")
            if stall_count >= MAX_STALLS:
                logging.info("🎯 Local minimum confirmed. Terminating sweep.")
                break

    # Register the final optimized Elite
    final_hash = generate_canonical_hash(champ_params)
    hunter.add_job({**champ_params, "config_hash": final_hash, "origin": "predator_elite"})
    return 0

if __name__ == "__main__":
    sys.exit(main())
