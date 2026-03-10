#!/usr/bin/env python3

"""
adaptive_hunt_orchestrator.py
CLASSIFICATION: Master Driver (ASTE V12.1 - NSGA-II Aligned)
GOAL: Manages the entire end-to-end simulation lifecycle. 
      Now fully synchronized with the Multi-Objective NSGA-II Hunter,
      tracking true composite fitness rather than just scalar SSE.
"""

import os
import json
import subprocess
import sys
import copy
import uuid
import logging
import argparse
import shutil
import sqlite3
import concurrent.futures
import threading
from typing import Dict, Any, List, Optional
try:
    import yaml  # type: ignore
except ImportError:
    pass

from config_utils import generate_canonical_hash
import aste_hunter

# Import the new GIF pipeline manager
sys.path.append(os.path.join(os.path.dirname(__file__), 'visual_plotting'))
try:
    import gif_pipeline_manager  # type: ignore
except ImportError:
    pass

PROVENANCE_DIR = "provenance_reports"

# --- GLOBAL THREAD-SAFE BLEED INFRASTRUCTURE ---
bleed_lock = threading.Lock()
locked_for_bleed = set()
# Limit to 2 workers to prevent HDD mechanical thrashing
bleed_executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)

def bleed_artifact_to_hdd(source_path: str, target_dir: str):
    """Asynchronously moves heavy artifacts to the HDD to protect NVMe."""
    try:
        os.makedirs(target_dir, exist_ok=True)
        target_path = os.path.join(target_dir, os.path.basename(source_path))
        logging.info(f"🚚 [Memory Bleed] HDD Transfer initiated: {os.path.basename(source_path)}")
        shutil.move(source_path, target_path)
        logging.info(f"💾 [Memory Bleed] HDD Transfer complete: {os.path.basename(source_path)}")
    except Exception as e:
        logging.error(f"🛑 [Memory Bleed] HDD I/O Failure! Error: {e}")
    finally:
        # Guaranteed strict thread-safe unlock
        with bleed_lock:
            locked_for_bleed.discard(source_path)
# -----------------------------------------------

# --- Logging Setup ---
def setup_logging():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s')
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    if not logger.handlers:
        logger.addHandler(ch)

def load_configuration(config_path):
    if config_path.endswith('.json'):
        with open(config_path, 'r') as f:
            config = json.load(f)
    elif (config_path.endswith('.yaml') or config_path.endswith('.yml')) and 'yaml' in sys.modules:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        raise ValueError("Unsupported config file format. Use .json or .yaml")
    return config

def setup_directories(config):
    os.makedirs(config.get("config_dir", "input_configs"), exist_ok=True)
    os.makedirs(config.get("data_dir", "simulation_data"), exist_ok=True)
    os.makedirs(config.get("provenance_dir", PROVENANCE_DIR), exist_ok=True)
    os.makedirs(config.get("snapshot_dir", "pareto_snapshots"), exist_ok=True) # <-- NEW
    logging.info(f"I/O directories ensured.")

# --- Multi-Seed Robustness Verification ---
def overwrite_seed_in_config(params_filepath: str, seed: int):
    with open(params_filepath, 'r') as f:
        config = json.load(f)
    config['global_seed'] = seed
    with open(params_filepath, 'w') as f:
        json.dump(config, f, indent=2)

def extract_log_prime_sse(canonical_hash: str, seed: int) -> float:
    import time
    provenance_path = os.path.join(PROVENANCE_DIR, f"provenance_{canonical_hash}.json")
    max_retries = 3
    for attempt in range(max_retries):
        if os.path.exists(provenance_path):
            try:
                with open(provenance_path, 'r') as f:
                    provenance = json.load(f)
                spectral_data = provenance.get('spectral_fidelity', {})
                return spectral_data.get('log_prime_sse', 999.0)
            except json.JSONDecodeError:
                pass 
        time.sleep(1.0)
    logging.error(f"[{canonical_hash}] Provenance I/O timeout. Defaulting to penalty.")
    return 999.0

def save_pareto_snapshot(source_h5: str, dest_npz: str):
    """Extracts ONLY the final 25MB frame before the 6.2GB history file is vaporized."""
    import h5py  # type: ignore
    import numpy as np
    try:
        with h5py.File(source_h5, 'r') as f:
            final_psi = f['final_psi'][:]
            final_rho = f['final_rho'][:]
        np.savez_compressed(dest_npz, final_psi=final_psi, final_rho=final_rho)
    except Exception as e:
        logging.warning(f"Failed to create Pareto snapshot: {e}")

def dispatch_job_batch(gen, param_batch, config, hunter):
    jobs_to_run = []
    seen_hashes = set() 

    for base_params in param_batch:
        # Branch every parameter set into 3 distinct seeds immediately
        for seed in range(3):
            params = copy.deepcopy(base_params)
            params["generation"] = gen
            if "simulation" in config:
                params["simulation"] = config["simulation"]
            
            # Explicitly lock the seed into the parameters BEFORE hashing
            params["global_seed"] = seed

            if "config_hash" in params:
                del params["config_hash"] 

            # This guarantees 24 unique hashes for a population of 8
            true_hash = generate_canonical_hash(params)
            params["config_hash"] = true_hash

            if true_hash in seen_hashes:
                continue
            seen_hashes.add(true_hash)

            hunter.add_job(params)

            config_filename = f"config_{true_hash}.json"
            config_filepath = os.path.join(config.get("config_dir", "input_configs"), config_filename)

            with open(config_filepath, 'w') as f:
                json.dump(params, f, indent=4)

            jobs_to_run.append({
                aste_hunter.HASH_KEY: true_hash,
                "params_filepath": config_filepath
            })

    return jobs_to_run


def run_generation_on_gpu(jobs_to_run, config):
    """PHASE 1: Sequential GPU Batching (1-to-1 Mapping)"""
    logging.info(f"🚀 --- PHASE 1: SEQUENTIAL GPU EXECUTION ({len(jobs_to_run)} Jobs) --- 🚀")
    worker_script = "worker_cupy.py"
    data_dir = config.get("data_dir", "simulation_data")

    for job in jobs_to_run:
        config_hash = job[aste_hunter.HASH_KEY]
        params_filepath = job["params_filepath"]
        output_h5 = os.path.join(data_dir, f"rho_history_{config_hash}.h5")
        
        if os.path.exists(output_h5):
            logging.info(f"[GPU] Skipping {config_hash[:10]} - Artifact exists.")
            continue
            
        logging.info(f"[GPU ENGINE] Simulating {config_hash[:10]}...")
        try:
            subprocess.run([
                sys.executable, worker_script, 
                "--params", params_filepath, 
                "--output", output_h5
            ], check=True)
        except subprocess.CalledProcessError as e:
            logging.error(f"[GPU ENGINE] FATAL ERROR on {config_hash[:10]}: {e}")

def evaluate_robustness(config_hash, params_filepath, num_seeds=3):
    """PHASE 2: CPU Validation Sweep."""
    data_dir = "simulation_data"
    
    def _validate_seed(seed):
        seed_params = params_filepath.replace(".json", f"_seed{seed}.json")
        output_h5 = os.path.join(data_dir, f"rho_history_{config_hash}_seed{seed}.h5")
        
        # 🚨 NEW FAILSAFE: If the twin filter fails or a file vanishes, fail gracefully.
        if not os.path.exists(seed_params):
            logging.warning(f"[{config_hash}] Seed {seed} params missing. Skipping gracefully.")
            return 999.0, None

        with open(seed_params, 'r') as f:
            seed_config = json.load(f)
        canonical_hash = generate_canonical_hash(seed_config)
        
        if not os.path.exists(output_h5):
            logging.warning(f"[{config_hash}] H5 artifact missing for seed {seed}. Max penalty.")
            return 999.0, canonical_hash
            
        logging.info(f"[CPU VALIDATION] Analyzing {config_hash[:10]} | Seed {seed}...")
        try:
            subprocess.run([
                sys.executable, "validation_pipeline.py",
                "--input", output_h5,
                "--params", seed_params,
                "--output_dir", PROVENANCE_DIR
            ], check=True)
        except subprocess.CalledProcessError as e:
            logging.error(f"[CPU VALIDATION] Failed on {config_hash[:10]} Seed {seed}: {e}")
            return 999.0, canonical_hash
            
        return extract_log_prime_sse(canonical_hash, seed), canonical_hash

    sses = []
    canonical_hashes = []
    seed0_canonical_hash = None
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_seeds) as executor:
        futures = {executor.submit(_validate_seed, s): s for s in range(num_seeds)}
        for future in concurrent.futures.as_completed(futures):
            seed = futures[future]
            sse, c_hash = future.result()
            sses.append(sse)
            if c_hash:
                canonical_hashes.append(c_hash)
            if seed == 0:
                seed0_canonical_hash = c_hash
            
    canonical_source_h5 = os.path.join(data_dir, f"rho_history_{config_hash}_seed0.h5")
    canonical_dest_h5 = os.path.join(data_dir, f"rho_history_{config_hash}.h5")
    if os.path.exists(canonical_source_h5):
        shutil.copy(canonical_source_h5, canonical_dest_h5)
        
    if seed0_canonical_hash:
        prov_source = os.path.join(PROVENANCE_DIR, f"provenance_{seed0_canonical_hash}.json")
        prov_dest = os.path.join(PROVENANCE_DIR, f"provenance_{config_hash}.json")
        if os.path.exists(prov_source):
            shutil.copy(prov_source, prov_dest)

    # --- THE JSON MICRO-PURGE ---
    for c_hash in canonical_hashes:
        prov_file = os.path.join(PROVENANCE_DIR, f"provenance_{c_hash}.json")
        csv_file = os.path.join(PROVENANCE_DIR, f"{c_hash}_quantule_events.csv")
        if os.path.exists(prov_file):
            try: os.remove(prov_file)
            except OSError: pass
        if os.path.exists(csv_file):
            try: os.remove(csv_file)
            except OSError: pass
            
    for seed in range(num_seeds):
        seed_params = params_filepath.replace(".json", f"_seed{seed}.json")
        if os.path.exists(seed_params):
            try: os.remove(seed_params)
            except OSError: pass
            
    return sum(sses) / len(sses) if sses else 999.0

def run_simulation_job(config_hash: str, params_filepath: str, suffix: str = "") -> tuple[bool, str]:
    output_h5 = os.path.join("simulation_data", f"rho_history_{config_hash}{suffix}.h5")
    try:
        worker_cmd = [sys.executable, "worker_cupy.py", "--params", params_filepath, "--output", output_h5]
        subprocess.run(worker_cmd, check=True)
        val_cmd = [sys.executable, "validation_pipeline.py", "--input", output_h5, "--params", params_filepath, "--output_dir", PROVENANCE_DIR]
        result = subprocess.run(val_cmd, check=True, capture_output=True, text=True)
        import re
        match = re.search(r"Generated Canonical config_hash:\s*([a-fA-F0-9]+)", (result.stdout or "") + "\n" + (result.stderr or ""))
        return True, match.group(1) if match else config_hash
    except subprocess.CalledProcessError as e:
        logging.error(f"[{config_hash[:10]}] FSS Job failed: {e}")
        return False, config_hash

def fss_grid_invariance_harness(params_filepath, config_hash, sse_threshold=0.2):
    import copy, tempfile
    with open(params_filepath, 'r') as f:
        params = json.load(f)
        
    sse_results = []
    for N in [32, 64]:
        params_mod = copy.deepcopy(params)
        if 'simulation' not in params_mod:
            params_mod['simulation'] = {}
        params_mod['simulation']['N_grid'] = N
        
        with tempfile.NamedTemporaryFile('w', suffix='.json', delete=False) as tmpf:
            json.dump(params_mod, tmpf, indent=2)
            tmpf.flush()
            temp_params_path = tmpf.name
            
        success, canonical_hash = run_simulation_job(config_hash, temp_params_path, suffix=f"_FSS_{N}")
        
        if not success:
            sse_results.append(999.0)
            os.remove(temp_params_path)
            continue
            
        seed = params_mod.get('global_seed', 42)
        sse = extract_log_prime_sse(canonical_hash, seed)
        sse_results.append(sse)
        
        if os.path.exists(temp_params_path): os.remove(temp_params_path)
        fss_h5_path = os.path.join("simulation_data", f"rho_history_{config_hash}_FSS_{N}.h5")
        if os.path.exists(fss_h5_path):
            try: os.remove(fss_h5_path)
            except OSError: pass
                
    return all((sse < sse_threshold and sse < 900.0) for sse in sse_results)

def aggregate_results(gen, jobs_to_run, config, hunter):
    """PHASE 2: Parallel CPU Validation & Sequential FSS"""
    job_hashes_completed = []
    data_dir = config.get("data_dir", "simulation_data")
    prov_dir = config.get("provenance_dir", PROVENANCE_DIR)
    
    # 1. PARALLEL CPU VALIDATION
    def _validate_job(job):
        config_hash = job[aste_hunter.HASH_KEY]
        params_filepath = job["params_filepath"]
        output_h5 = os.path.join(data_dir, f"rho_history_{config_hash}.h5")
        
        if not os.path.exists(output_h5):
            logging.warning(f"[{config_hash[:10]}] H5 artifact missing. Skipping validation.")
            return None
            
        logging.info(f"[CPU VALIDATION] Analyzing {config_hash[:10]}...")
        try:
            subprocess.run([
                sys.executable, "validation_pipeline.py",
                "--input", output_h5,
                "--params", params_filepath,
                "--output_dir", prov_dir
            ], check=True)
            return config_hash, params_filepath
        except subprocess.CalledProcessError as e:
            logging.error(f"[CPU VALIDATION] Failed on {config_hash[:10]}: {e}")
            return None

    successful_jobs = []
    # 3 threads maximize CPU without bottlenecking I/O
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(_validate_job, job) for job in jobs_to_run]
        for future in concurrent.futures.as_completed(futures):
            res = future.result()
            if res:
                successful_jobs.append(res)
                job_hashes_completed.append(res[0])

    # --- THE VERIFICATION HELPER ---
    expected = len(jobs_to_run)
    actual = len(job_hashes_completed)
    if actual < expected:
        logging.warning(f"⚠️ [Verification] Only {actual}/{expected} runs passed validation. {expected - actual} drops detected.")
    else:
        logging.info(f"✅ [Verification] 100% Success! {actual}/{expected} runs validated.")

    # 2. SEQUENTIAL FSS & TOPOLOGY (VRAM Safe!)
    for c_hash, p_path in successful_jobs:
        prov_file = os.path.join(prov_dir, f"provenance_{c_hash}.json")
        if os.path.exists(prov_file):
            try:
                with open(prov_file, 'r') as f:
                    prov = json.load(f)
                
                # Check directly against the strict harmonic error
                err = float(prov.get("spectral_fidelity", {}).get("primary_harmonic_error", 999.0))
                
                # Restore the Grid Invariance FSS Check
                if err < 0.2:
                    logging.info(f"🌟 GOLDEN RUN DETECTED ({c_hash[:10]}). Executing Sequential FSS Check...")
                    if fss_grid_invariance_harness(p_path, c_hash, sse_threshold=0.2):
                        logging.info(f"  -> {c_hash[:10]} PASSED FSS Check.")
                    else:
                        logging.warning(f"  -> {c_hash[:10]} FAILED FSS Check. Nullifying fitness.")
                        prov["spectral_fidelity"]["primary_harmonic_error"] = 999.0
                        prov["spectral_fidelity"]["log_prime_sse"] = 999.0
                        with open(prov_file, 'w') as f:
                            json.dump(prov, f, indent=4)
            except Exception as e:
                logging.error(f"FSS parsing error on {c_hash[:10]}: {e}")

        # Extract Topology (CPU bound, safe to run sequentially here)
        rho_file = os.path.join(data_dir, f"rho_history_{c_hash}.h5")
        topo_file = os.path.join(data_dir, f"topology_metrics_{c_hash}.npz")
        if os.path.exists(rho_file) and os.path.exists("extract_topology_metrics.py"):
            try:
                subprocess.run([sys.executable, "extract_topology_metrics.py", "--input", rho_file, "--output", topo_file], check=True, stdout=subprocess.DEVNULL)
            except Exception: pass

    logging.info(f"GENERATION {gen} PIPELINE COMPLETE. Passing Data to NSGA-II Hunter...")
    hunter.process_generation_results(provenance_dir=prov_dir, job_hashes=job_hashes_completed)

def execute_predator_strikes():
    """
    Zero-VRAM CPU-side traffic controller. Drains the predator queue sequentially.
    Passes environment variables to preserve CuPy/JAX virtual environment pathing.
    """
    queue_file = "predator_queue.json"
    target_file = "current_predator_target.json"
    
    while True:
        if not os.path.exists(queue_file):
            break

        try:
            with open(queue_file, 'r') as f:
                queue = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logging.error(f"[Predator Hook] Queue read failed: {e}")
            break

        if not queue:
            try: os.remove(queue_file)
            except OSError: pass
            break

        # Safely pop the top target
        target = queue.pop(0)
        
        # Atomically rewrite the remaining queue
        try:
            if queue:
                tmp_file = queue_file + ".tmp"
                with open(tmp_file, 'w') as f:
                    json.dump(queue, f, indent=4)
                os.replace(tmp_file, queue_file)
            else:
                os.remove(queue_file)
        except IOError:
            break

        # Write target to a temporary handoff file
        with open(target_file, 'w') as f:
            json.dump(target, f, indent=4)

        # BLOCKING EXECUTION
        logging.info(f"🦅 [PREDATOR MODE] Intercepting generation. GPU Traffic halted. Executing sweep...")
        try:
            # CRITICAL: Preserve env so subprocess targets the correct venv Python & dependencies
            current_env = os.environ.copy()
            subprocess.run([sys.executable, "predator_sweep.py", "--target", target_file], 
                           check=True, env=current_env)
            logging.info("🦅 [PREDATOR MODE] Sweep concluded successfully.")
        except subprocess.CalledProcessError as e:
            logging.error(f"⚠️ [PREDATOR MODE] Sweep aborted or collapsed (Code {e.returncode}).")
        finally:
            if os.path.exists(target_file):
                try: os.remove(target_file)
                except OSError: pass


def main():
    global_best_fitness = -1.0
    global_best_hash = None
    
    setup_logging()
    parser = argparse.ArgumentParser(description="ASTE Adaptive Hunt Orchestrator")
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--start-gen', type=int, default=None, help='Generation to start from')
    parser.add_argument('--dry-run', action='store_true', help='Validate and exit')
    args = parser.parse_args()
    
    config = load_configuration(args.config)
    setup_directories(config)
    hunter = aste_hunter.Hunter(db_file=config["db_path"])
    start_gen = args.start_gen if args.start_gen is not None else hunter.get_current_generation()
    
    if args.dry_run:
        print("[DRY RUN] Config validated. Exiting.")
        return
        
    for gen in range(start_gen, config["generations"]):
        # --- HOTFIX: Correct Soft-Halt Indentation and single loop ---
        if os.path.exists("stop_after_gen.txt"):
            logging.warning("🛑 [Deployment] 'stop_after_gen.txt' detected.")
            logging.info("🛑 Executing clean suspension. The Orchestrator will now exit safely.")
            if 'bleed_executor' in globals():
                bleed_executor.shutdown(wait=True) # Forces completion of HDD transfers
            break
            
        logging.info(f"==== ASTE ORCHESTRATOR: STARTING GENERATION {gen} ====")
        execute_predator_strikes()
        
        # --- NEW: BACKLOG QUEUE INGESTION ---
        backlog_file = "backlog_queue.json"
        parameter_batch = []
        
        if os.path.exists(backlog_file):
            try:
                with open(backlog_file, "r") as f:
                    backlog = json.load(f)
                
                if backlog:
                    # Take a chunk of configs equal to the population size
                    pop_size = config.get("population_size", 8)
                    parameter_batch = backlog[:pop_size]
                    remaining_backlog = backlog[pop_size:]
                    
                    with open(backlog_file, "w") as f:
                        json.dump(remaining_backlog, f, indent=4)
                        
                    logging.info(f"📦 [Backlog] Ingested {len(parameter_batch)} legacy configs. {len(remaining_backlog)} remaining in queue.")
                else:
                    os.remove(backlog_file) # Queue is empty, delete it
            except Exception as e:
                logging.error(f"[Backlog] Error reading queue: {e}")
        
        # If no backlog exists or it just emptied, fallback to standard genetic generation
        if not parameter_batch:
            parameter_batch = hunter.generate_next_generation(
                population_size=config["population_size"],
                bounds=config.get("bounds")
            )

        # --- NEW: CONFIDENCE-GATED SCALING INGESTION ---
        probe_file = "scaling_probe.json"
        if os.path.exists(probe_file):
            try:
                with open(probe_file, "r") as f:
                    probe = json.load(f)
                
                if probe.get("confidence", 0.0) >= 0.65:
                    logging.info(f"🧬 [Orchestrator] Deduction Probe Verified (Confidence: {probe.get('confidence'):.3f}).")
                    if "confidence" in probe: del probe["confidence"]
                    probe["generation"] = gen
                    probe["origin"] = "synthetic_scaling_probe"
                    parameter_batch.append(probe)
                else:
                    logging.warning("⚠️ [Orchestrator] Rejected low-confidence probe.")
                os.remove(probe_file)
            except Exception as e:
                logging.error(f"[Orchestrator] Failed to ingest scaling probe: {e}")
        # ------------------------------------------------

        jobs_to_run = dispatch_job_batch(gen, parameter_batch, config, hunter)

        # --- PHASE 1: ASYNCHRONOUS GPU BATCHING ---
        run_generation_on_gpu(jobs_to_run, config)
        
        # --- PHASE 2: CPU VALIDATION SWEEP ---
        logging.info("🧠 --- PHASE 2: CPU VALIDATION & MICRO-PURGING --- 🧠")
        aggregate_results(gen, jobs_to_run, config, hunter)

        # --- PHASE 3: NSGA-II INTELLIGENCE & TELEMETRY ---
        data_dir = config.get("data_dir", "simulation_data")
        current_gen_elite = None
        
        # 1. Fetch the absolute best run of the CURRENT generation
        try:
            with hunter._get_connection() as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute("SELECT config_hash, fitness, log_prime_sse FROM results WHERE generation=? ORDER BY fitness DESC LIMIT 1", (gen,))
                row = cursor.fetchone()
                if row:
                    current_gen_elite = dict(row)
        except Exception as e:
            logging.error(f"Failed to fetch Gen {gen} Elite: {e}")
        
        if current_gen_elite:
            gen_fitness = current_gen_elite['fitness']
            gen_hash = current_gen_elite['config_hash']
            logging.info(f"🧬 Gen {gen} Elite: {gen_hash[:10]}... (Fitness: {gen_fitness:.4f}, SSE: {current_gen_elite['log_prime_sse']:.6f})")
            
            # 2. Render the Generation Elite to GIF *before* the purge
            gen_h5_path = os.path.join(data_dir, f"rho_history_{gen_hash}.h5")
            try:
                if 'gif_pipeline_manager' in sys.modules and os.path.exists(gen_h5_path):
                    logging.info(f"[Orch] Rendering Generation {gen} Elite to GIF Dashboard...")
                    sys.modules['gif_pipeline_manager'].process_new_high_score(gen_h5_path, gen_fitness)
            except Exception as e:
                logging.error(f"GIF Pipeline failed: {e}")

            # 3. Update the Global Protection Tracker
            if gen_fitness > global_best_fitness and gen_fitness > 0.0:
                logging.info(f"[Orch] 🌟 NEW GLOBAL CHAMPION! Protecting .h5 Artifact.")
                global_best_fitness = gen_fitness
                global_best_hash = gen_hash

        # --- PHASE 4: MASTER LEDGER DUMP ---
        json_backup_path = os.path.join(data_dir, f"master_ledger_gen_{gen}.json")
        try:
            hunter.export_ledger_to_json(json_backup_path)
            logging.info(f"[Data Contract] Ledger backed up to {json_backup_path}")
        except Exception as e:
            logging.error(f"[Data Contract] Failed JSON ledger export: {e}")
            
        # --- PHASE 4.5: PARETO SNAPSHOT EXTRACTION ---
        snapshot_dir = config.get("snapshot_dir", "pareto_snapshots")
        try:
            with hunter._get_connection() as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                # Fetch all hashes that made it to the Pareto Archive this generation
                cursor.execute("SELECT config_hash FROM pareto_archive WHERE generation=?", (gen,))
                pareto_hashes = [row['config_hash'] for row in cursor.fetchall()]
                
                if pareto_hashes:
                    logging.info(f"[Snapshot] Preserving {len(pareto_hashes)} Pareto Front geometries...")
                    for phash in pareto_hashes:
                        h5_path = os.path.join(data_dir, f"rho_history_{phash}.h5")
                        npz_path = os.path.join(snapshot_dir, f"snapshot_{phash}.npz")
                        
                        if os.path.exists(h5_path) and not os.path.exists(npz_path):
                            save_pareto_snapshot(h5_path, npz_path)
                            logging.info(f"  -> Saved 25MB full-res state for Pareto Elite: {phash[:10]}")
        except Exception as e:
            logging.error(f"Pareto snapshot extraction failed: {e}")

        # --- NEW: PHASE 4.8 - ASYNCHRONOUS MEMORY BLEED TRIGGER ---
        logging.info("[Orchestrator] Scanning for High-Value artifacts for HDD Archival...")
        try:
            with hunter._get_connection() as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT m.config_hash 
                    FROM metrics m
                    JOIN runs r ON m.config_hash = r.config_hash
                    WHERE r.generation = ? AND (m.primary_harmonic_error < 0.01 OR m.collapse_event_count > 0)
                """, (gen,))
                
                high_value_runs = cursor.fetchall()
                bleed_dir = r"E:\GPU_BleedOut_store"
                
                for row in high_value_runs:
                    h5_path = os.path.join(data_dir, f"rho_history_{row['config_hash']}.h5")
                    if os.path.exists(h5_path):
                        with bleed_lock:
                            if h5_path not in locked_for_bleed:
                                locked_for_bleed.add(h5_path)
                                bleed_executor.submit(bleed_artifact_to_hdd, h5_path, bleed_dir)
                                logging.info(f"  -> Queued Artifact for Bleed: {row['config_hash'][:10]}")
        except Exception as e:
            logging.error(f"[Memory Bleed] Failed to query or submit high-value runs: {e}")
        # ----------------------------------------------------------

        # --- PHASE 5: SSD PURGE (Protect the Global Elite) ---
        logging.info(f"[Cleanup] Post-generation purge of heavy .h5 artifacts...")
        purged_count = 0
        champion_file = f"rho_history_{global_best_hash}.h5" if global_best_hash else None

        for filename in os.listdir(data_dir):
            if filename.endswith(".h5") and filename != champion_file:
                file_path = os.path.join(data_dir, filename)
                
                # --- NEW: EXPLICIT MUTEX BLEED PROTECTOR ---
                with bleed_lock:
                    is_locked = file_path in locked_for_bleed
                
                if is_locked:
                    logging.info(f"[Cleanup] Skipping {filename} - currently in transit to HDD.")
                    continue
                # -------------------------------------------
                
                try:
                    os.remove(file_path)
                    purged_count += 1
                except Exception as e:
                    logging.warning(f"[Cleanup] Could not delete {file_path}: {e}")

        logging.info(f"[Cleanup] Purge complete: {purged_count} .h5 files destroyed. Saved champion: {champion_file}. Ready for Gen {gen+1}.")

    logging.info("==== ASTE ORCHESTRATOR: ALL GENERATIONS COMPLETE ====")
    # Ensure threads terminate cleanly
    bleed_executor.shutdown(wait=False)

if __name__ == "__main__":
    logging.info("[Orchestrator] Script entry: starting main()...")
    try:
        main()
    except Exception as e:
        logging.critical(f"[Orchestrator] Unhandled exception: {e}", exc_info=True)
        sys.exit(1)