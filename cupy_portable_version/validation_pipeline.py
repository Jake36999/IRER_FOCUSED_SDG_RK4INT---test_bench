#!/usr/bin/env python3

from config_utils import generate_canonical_hash
"""
validation_pipeline.py
ASSET: A6 (Spectral Fidelity & Provenance Module)
VERSION: 2.0 (Phase 3 Scientific Mandate)
CLASSIFICATION: Final Implementation Blueprint / Governance Instrument
GOAL: Serves as the immutable source of truth that cryptographically binds
      experimental intent (parameters) to scientific fact (spectral fidelity)
      and Aletheia cognitive coherence.
"""

import argparse
import json
import os
import sys
import tempfile
import hashlib
from datetime import datetime, timezone
from typing import Any, Dict, Optional, List
import numpy as np
import pandas as pd  # type: ignore[import]
import h5py  # type: ignore
# NOTE: config_hash computation removed here to avoid using undefined params_dict at import time.

# --- Import analysis/validation modules ---
import tda_profiler
import Alethiea.aletheia_diagnostics as aletheia_diagnostics  # type: ignore[import]
import metrics.collapse_dynamics as collapse_metrics  # type: ignore[import]
import metrics.monte_carlo_engine as monte_carlo_engine  # type: ignore[import]
import metrics.spdc_empirical_bridge as spdc_empirical_bridge
import metrics.tensor_validation as tensor_validation
import quantulemapper_real as cep_profiler

# --- Logging setup ---
import logging

def setup_logger(name: str):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('[%(asctime)s] %(levelname)s %(name)s: %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger

logger = setup_logger("validation_pipeline")
# Import Scipy for new Aletheia Metrics
try:
    from scipy.signal import coherence as scipy_coherence
    from scipy.stats import entropy as scipy_entropy
except ImportError:
    print("FATAL: Missing 'scipy'. Please install: pip install scipy", file=sys.stderr)
    sys.exit(1)

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'gravity'))
try:
    import unified_omega
except ImportError:
    pass


# --- MODULE CONSTANTS ---
SCHEMA_VERSION = "SFP-v2.0-ARCS" # Upgraded schema version

# ---
# SECTION 1: PROVENANCE KERNEL (EVIDENTIAL INTEGRITY)
# ---

# ---
# SECTION 2: FIDELITY KERNEL (SCIENTIFIC VALIDATION)
# ---

def run_quantule_profiler(
    rho_history_path: str,
    temp_file_path: Optional[str] = None # Added for explicit temporary file handling
) -> Dict[str, Any]:
    """
    Orchestrates the core scientific analysis by calling the
    Quantule Profiler (CEPP v1.0 / quantulemapper.py).

    This function replaces the v1.0 mock logic. It loads the HDF5 artifact,
    saves it as a temporary .npy file (as required by the profiler's API),
    and runs the full analysis.
    """
    if temp_file_path is None:
        # Create a temporary .npy file for the profiler to consume
        with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as tmp:
            temp_file_path = tmp.name
        _cleanup_temp_file = True
    else:
        _cleanup_temp_file = False

    try:
        # 1. Load ONLY the final state from HDF5 (Bypasses massive RAM overhead)
        with h5py.File(rho_history_path, 'r') as f:
            if 'final_rho' in f:
                rho_final_3d = f['final_rho'][:]
            elif 'rho_history' in f: # Fallback for legacy files
                rho_final_3d = f['rho_history'][-1]
            else:
                raise ValueError("No valid rho data found in HDF5.")

        if rho_final_3d.ndim != 3:
            raise ValueError(f"Input HDF5 'final_rho' is not 3D. Shape: {rho_final_3d.shape}")

        # 2. Convert to .npy (Now writes ~16MB instead of 6GB)
        np.save(temp_file_path, rho_final_3d)
        
        # 3. Run the Quantule Profiler (CEPP v2.0)
        logger.info(f"[FidelityKernel] Calling Quantule Profiler (CEPP v2.0) on {temp_file_path}")

        # --- NEW "FAIL LOUD" PATCH ---
        try:
            # Load the .npy file and run prime_log_sse
            rho_data = np.load(temp_file_path)
            profiler_results = cep_profiler.prime_log_sse(rho_data)

            # --- Bragg Peak Crystallography ---
            # Use the final state for 2D Bragg analysis
            if rho_data.ndim == 4:
                rho_final = rho_data[-1, :, :, :]
            elif rho_data.ndim == 3:
                rho_final = rho_data
            else:
                raise ValueError(f"rho_data has unexpected shape: {rho_data.shape}")

            # Project to 2D for crystallography (max projection)
            rho_final_2d = np.max(rho_final, axis=0)
            n_bragg_peaks = cep_profiler.detect_bragg_peaks(rho_final_2d)
            bragg_prime_sse = cep_profiler.validate_prime_bragg_lattice(rho_final_2d)

            # Extract metrics with safe defaults for legacy payloads.
            log_prime_sse = float(profiler_results.get("log_prime_sse", 999.0))
            validation_status = "PASS" if log_prime_sse < 1.0 else "FAIL: HIGH_SSE"

            # Get Sprint 2 Falsifiability Metrics
            metrics_sse_null_a = float(profiler_results.get("sse_null_phase_scramble", 999.0))
            metrics_sse_null_b = float(profiler_results.get("sse_null_target_shuffle", 999.0))

        except Exception as e:
            logger.critical(f"CRITICAL: CEPP Profiler failed: {e}")
            # Re-raise the exception to fail the validation step.
            # This will stop the orchestrator and show us the error.
            raise

        # 4. Extract key results for the SFP artifact
        spectral_fidelity = {
            "validation_status": validation_status,
            "log_prime_sse": log_prime_sse,
            "scaling_factor_S": profiler_results.get("scaling_factor_S", 0.0),
            "dominant_peak_k": profiler_results.get("dominant_peak_k", 0.0),
            "analysis_protocol": "CEPP v2.0",
            "prime_log_targets": getattr(cep_profiler, "LOG_PRIME_TARGETS", np.array([])).tolist(),
            "sse_null_phase_scramble": metrics_sse_null_a,
            "sse_null_target_shuffle": metrics_sse_null_b,
            "primary_harmonic_error": float(profiler_results.get("primary_harmonic_error", 999.0)),
            "missing_peak_penalty": float(profiler_results.get("missing_peak_penalty", 0.0)),
            "noise_penalty": float(profiler_results.get("noise_penalty", 0.0)),
            "best_single_error": float(profiler_results.get("best_single_error", 999.0)),
            # Bragg Peak Crystallography
            "n_bragg_peaks": n_bragg_peaks,
            "bragg_prime_sse": bragg_prime_sse,
            # New diagnostic fields:
            "n_peaks_found_main": profiler_results.get("n_peaks_found_main", 0),
            "failure_reason_main": profiler_results.get("failure_reason_main", None),
            "n_peaks_found_null_a": profiler_results.get("n_peaks_found_null_a", 0),
            "failure_reason_null_a": profiler_results.get("failure_reason_null_a", None),
            "n_peaks_found_null_b": profiler_results.get("n_peaks_found_null_b", 0),
            "failure_reason_null_b": profiler_results.get("failure_reason_null_b", None),
            
            # --- THE FIX: PASS THE ARRAYS THROUGH TO THE JSON ---
            "measured_peaks": profiler_results.get("measured_peaks", []),
            "scaled_peaks": profiler_results.get("scaled_peaks", [])
        }
        return {
            "spectral_fidelity": spectral_fidelity,
            "classification_results": profiler_results.get("csv_files", {}),
            "raw_rho_final_state": rho_final
        }

    except Exception as e:
        logger.error(f"[FidelityKernel Error] Failed during Quantule Profiler execution or data loading: {e}")
        raise # Re-raise to ensure orchestrator catches the failure
    finally:
        # Clean up the temporary .npy file if it was created by this function
        if _cleanup_temp_file and temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)

def extract_lom_telemetry(h5_path: str, config_hash: str, output_dir: str, L_domain: float, N_grid: int, params_dict: dict) -> int:
    try:
        if not os.path.exists(h5_path): return 0
        with h5py.File(h5_path, 'r') as f:
            if 'final_psi' not in f: return 0
            psi_final_3d = f['final_psi'][:]
            
            # 1. TEMPORAL GRAVITY (Agent 2's logic: Extract the timeline array)
            omega_temporal_mean = 0.0
            bandwidth_dk_val = 0.0
            emergence_t_step_val = 0
            
            if "quantule_events" in f:
                q_grp = f["quantule_events"]
                if "omega_local" in q_grp and len(q_grp["omega_local"]) > 0:
                    omega_temporal_mean = float(np.mean(q_grp["omega_local"][:]))
                    bandwidth_dk_val = float(q_grp["bandwidth"][0])
                    emergence_t_step_val = int(q_grp["t_step"][0])
                    
                    # Output a separate timeline file for analytical review
                    pd.DataFrame({
                        "t_step": q_grp["t_step"][:],
                        "omega_local": q_grp["omega_local"][:],
                        "spectral_bandwidth_dk": q_grp["bandwidth"][:]
                    }).to_csv(os.path.join(output_dir, f"{config_hash}_gravity_timeline.csv"), index=False)
            
        rho = np.abs(psi_final_3d)**2
        theta = np.angle(psi_final_3d)
        
        z_indices, y_indices, x_indices = np.where(rho > 0.8)
        if len(z_indices) == 0: return 0
            
        dx = L_domain / N_grid
        phys_x = (x_indices - N_grid / 2) * dx
        phys_y = (y_indices - N_grid / 2) * dx
        phys_z = (z_indices - N_grid / 2) * dx
            
        normalized_phase = (theta[z_indices, y_indices, x_indices] + np.pi) / (2 * np.pi)
        
        # 2. SPATIAL GRAVITY (My logic: Give the laser etcher the true 3D fluid topology)
        omega_sq_field = unified_omega.derive_stable_conformal_factor(rho, params_dict)
        spatial_gravity_pressure = omega_sq_field[z_indices, y_indices, x_indices]
        
        df = pd.DataFrame({
            'idx_x': x_indices, 'idx_y': y_indices, 'idx_z': z_indices,
            'phys_x': phys_x, 'phys_y': phys_y, 'phys_z': phys_z,
            'rho_intensity': rho[z_indices, y_indices, x_indices],
            'complex_phase_normalized': normalized_phase,
            'temporal_omega_mean': omega_temporal_mean,          # The overall temporal stability
            'spatial_gravity_omega_sq': spatial_gravity_pressure, # The specific voxel pressure for the laser!
            'bandwidth_dk': bandwidth_dk_val,
            'emergence_t_step': emergence_t_step_val
        })
        
        # --- FABRICATION SCALING ---
        scale_nm = (L_domain * 1e9) / N_grid
        df["x_nm"] = df["phys_x"] * scale_nm
        df["y_nm"] = df["phys_y"] * scale_nm
        df["z_nm"] = df["phys_z"] * scale_nm
        
        os.makedirs(output_dir, exist_ok=True)
        df.to_csv(os.path.join(output_dir, f"{config_hash}_etch_ready.csv"), index=False)
        return len(z_indices)
    except Exception as e:
        logger.warning(f"[LOM Telemetry] Failed to extract events: {e}")
        return 0


def perform_phase_ablation(psi_final_3d: np.ndarray) -> np.ndarray:
    """
    [Falsifiability Protocol] True Phase Ablation
    Strips the phase from the complex field to create a 'null' state.
    Equation: psi_null = |psi|
    
    This is used to verify that emergent geometry relies on topological 
    phase dynamics, not just static density clustering (density blobs).
    """
    # Taking the absolute value strips the phase angle, leaving only the magnitude.
    # We cast it back to complex64 if downstream spectral profilers expect complex types.
    psi_null = np.abs(psi_final_3d).astype(np.complex64)
    
    return psi_null

def compute_tau_c(time_series: np.ndarray) -> float:
    """[Epistemic Guardrail] Adaptive Temporal Windowing (Tau_C)."""
    if len(time_series) < 2: return 0.0
    ts_norm = time_series - np.mean(time_series)
    if np.all(ts_norm == 0): return 0.0
    autocorr = np.correlate(ts_norm, ts_norm, mode='full')
    autocorr = autocorr[autocorr.size // 2:]
    autocorr /= (autocorr[0] + 1e-12)
    tau_c = np.where(autocorr < np.exp(-1))[0]
    return float(tau_c[0]) if len(tau_c) > 0 else float(len(autocorr))

"""
# ---
# SECTION 3: ALETHEIA COHERENCE METRICS (PHASE 3)
# ---
"""

def calculate_pli(rho_final_state: np.ndarray) -> float:
    """
    [Phase 3] Calculates the Principled Localization Index (PLI).
    Analogue: Mott Insulator phase.
    Implementation: Inverse Participation Ratio (IPR).

    IPR = sum(psi^4) / (sum(psi^2))^2
    A value of 1.0 is perfectly localized (Mott), 1/N is perfectly delocalized (Superfluid).
    We use the density field `rho` as our `psi^2` equivalent.
    """
    try:
        # Normalize the density field (rho is already > 0)
        sum_rho = np.sum(rho_final_state)
        if sum_rho == 0:
            return 0.0
        rho_norm = rho_final_state / sum_rho

        # Calculate IPR on the normalized density
        # IPR = sum(p_i^2)
        pli_score = np.sum(rho_norm**2)

        # Scale by N to get a value between (0, 1)
        N_cells = rho_final_state.size
        pli_score_normalized = float(pli_score * N_cells)

        if np.isnan(pli_score_normalized):
            return 0.0
        return pli_score_normalized

    except Exception as e:
        logger.warning(f"[AletheiaMetrics] PLI calculation failed: {e}")
        return 0.0

def calculate_ic(rho_final_state: np.ndarray) -> float:
    """
    [Phase 3] Calculates the Informational Compressibility (IC).
    Analogue: Thermodynamic compressibility.
    Implementation: K_I = dS / dE (numerical estimation).
    """
    try:
        # 1. Proxy for System Energy (E):
        # We use the L2 norm of the field (sum of squares) as a simple energy proxy.
        proxy_E = np.sum(rho_final_state**2)

        # 2. Proxy for System Entropy (S):
        # We treat the normalized field as a probability distribution
        # and calculate its Shannon entropy.
        rho_flat = rho_final_state.flatten()
        sum_rho_flat = np.sum(rho_flat)
        if sum_rho_flat == 0:
            return 0.0 # Cannot calculate entropy for zero field
        rho_prob = rho_flat / sum_rho_flat
        # Add epsilon to avoid log(0)
        proxy_S = scipy_entropy(rho_prob + 1e-9)

        # 3. Calculate IC = dS / dE
        # We perturb the system slightly to estimate the derivative

        # Create a tiny perturbation (add 0.1% energy)
        epsilon = 0.001
        rho_perturbed = rho_final_state * (1.0 + epsilon)

        # Calculate new E and S
        proxy_E_p = np.sum(rho_perturbed**2)

        rho_p_flat = rho_perturbed.flatten()
        sum_rho_p_flat = np.sum(rho_p_flat)
        if sum_rho_p_flat == 0:
            return 0.0
        rho_p_prob = rho_p_flat / sum_rho_p_flat
        proxy_S_p = scipy_entropy(rho_p_prob + 1e-9)

        # Numerical derivative
        dE = proxy_E_p - proxy_E
        dS = proxy_S_p - proxy_S

        if dE == 0 or np.isnan(dE) or np.isnan(dS):
            return 0.0 # Incompressible or calculation failed

        ic_score = float(dS / dE)

        if np.isnan(ic_score):
            return 0.0
        return ic_score

    except Exception as e:
        logger.warning(f"[AletheiaMetrics] IC calculation failed: {e}")
        return 0.0

# ---
# SECTION 4: MAIN ORCHESTRATION (DRIVER HOOK)
# ---

def parse_manifest(manifest_path: str) -> List[Dict[str, str]]:
    """
    Parse a JSON or CSV manifest file listing input artifacts and parameter files.
    Returns a list of dicts with keys: 'input', 'params'.
    """
    if manifest_path.endswith('.json'):
        # Text file: open() is correct, do not use indexing
        from typing import TextIO
        with open(manifest_path, 'r') as f:  # type: TextIO
            assert not hasattr(f, '__getitem__'), "TextIOWrapper is not indexable. Use json.load(f) or f.read()."
            manifest = json.load(f)
        if isinstance(manifest, list):
            return manifest
        elif isinstance(manifest, dict) and 'jobs' in manifest:
            return manifest['jobs']
        else:
            raise ValueError("Manifest JSON must be a list or have a 'jobs' key.")
    elif manifest_path.endswith('.csv'):
        import csv
        from typing import TextIO
        jobs = []
        with open(manifest_path, 'r', newline='') as f:  # type: TextIO
            assert not hasattr(f, '__getitem__'), "TextIOWrapper is not indexable. Use csv.DictReader(f)."
            reader = csv.DictReader(f)
            for row in reader:
                jobs.append({'input': row['input'], 'params': row['params']})
        return jobs
    else:
        raise ValueError("Manifest must be .json or .csv")

def run_pipeline_single(input_path: str, params_path: str, output_dir: str) -> bool:
    """
    Run the full validation pipeline for a single input/params pair.
    Returns True if successful, False otherwise.
    """
    logger.info(f"--- SFP Module (Asset A6, v2.0) Initiating Validation ---")
    logger.info(f"  Input Artifact: {input_path}")
    logger.info(f"  Params File:    {params_path}")

    # --- 1. Provenance Kernel (Hashing) ---
    logger.info("[1. Provenance Kernel]")
    try:
        from typing import TextIO
        with open(params_path, 'r') as param_file:  # type: TextIO
            assert not hasattr(param_file, '__getitem__'), "TextIOWrapper is not indexable. Use json.load(f) or f.read()."
            params_dict = json.load(param_file)
    except Exception as e:
        logger.error(f"Could not load params file: {e}")
        return False

    try:
        # THE HASH MUTATION FIX: Trust the Orchestrator's hash if it exists
        if "config_hash" in params_dict:
            config_hash = params_dict["config_hash"]
        else:
            # Fallback for manual or legacy runs
            config_hash = generate_canonical_hash(params_dict)
    except Exception as e:
        logger.error(f"Failed to extract/generate config hash: {e}")
        return False

    logger.info(f"  Canonical config_hash: {config_hash}")
    
    # KEEP THIS: Essential for mapping legacy F: Drive results
    param_hash_legacy = params_dict.get("param_hash_legacy", None)

    # --- 2. Fidelity Kernel (CEPP v2.0) ---
    logger.info("[2. Fidelity Kernel (CEPP v2.0)]")

    profiler_run_results = {
        "spectral_fidelity": {"validation_status": "FAIL: MOCK_INPUT", "log_prime_sse": 999.9},
        "classification_results": {},
        "raw_rho_final_state": np.zeros((16,16,16)) # Dummy shape
    }

    if input_path == "rho_history_mock.h5":
        logger.warning("Using 'rho_history_mock.h5'. This file is empty. Fidelity and Aletheia Metrics will be 0 or FAIL.")
    else:
        if not os.path.exists(input_path):
            logger.error(f"Input file not found: {input_path}")
            return False
        try:
            profiler_run_results = run_quantule_profiler(input_path)
        except Exception as e:
            logger.error(f"Quantule Profiler execution failed: {e}")
            return False

    spectral_fidelity_results = profiler_run_results.get("spectral_fidelity", {})
    classification_data = profiler_run_results.get("classification_results", {})
    rho_final = profiler_run_results.get("raw_rho_final_state", None)

    if not isinstance(spectral_fidelity_results, dict):
        spectral_fidelity_results = {}
    if not isinstance(classification_data, dict):
        classification_data = {}

    logger.info(f"  Validation Status: {spectral_fidelity_results.get('validation_status', 'N/A')}")
    logger.info(f"  Calculated SSE:    {spectral_fidelity_results.get('log_prime_sse', float('nan')):.6f}")

    # =========================================================
    # --- FAST TELEMETRY EXTRACTION (~2ms) ---
    # We extract this even for bad runs so the Hunter DB has accurate mapping data.
    # =========================================================
    psi_final_3d = None
    rho_final_3d = None
    final_psi = None
    gpu_telemetry = {} 

    if input_path != "rho_history_mock.h5" and os.path.exists(input_path) and input_path.lower().endswith('.h5'):
        try:
            import h5py
            with h5py.File(input_path, 'r') as h5f:
                logger.info(f"  Loaded HDF5 Artifact: {input_path}")
                if 'final_psi' in h5f:
                    psi_final_3d = h5f['final_psi'][:]
                    final_psi = psi_final_3d
                if 'final_rho' in h5f:
                    rho_final_3d = h5f['final_rho'][:]
                    
                # Extract True GPU-Calculated Phase Telemetry 
                if 'j_info_l2_history' in h5f:
                    j_info_hist = h5f['j_info_l2_history'][:]
                    gpu_telemetry['j_info_l2_mean'] = float(np.mean(j_info_hist))
                    
                    # --- PHANTOM FILTER (Relative Stability Gating) ---
                    gpu_telemetry['tau_c'] = compute_tau_c(j_info_hist)
                    gpu_telemetry['relative_variance'] = float(np.var(j_info_hist) / (np.mean(j_info_hist)**2 + 1e-12))
                    
                if 'grad_phase_var_history' in h5f:
                    grad_phase_var_hist = h5f['grad_phase_var_history'][:]
                    gpu_telemetry['grad_phase_var_mean'] = float(np.mean(grad_phase_var_hist))
                    gpu_telemetry['grad_phase_var_tau_c'] = compute_tau_c(grad_phase_var_hist)
                # --- NEW: Phase Coherence, Amplitude, Clamp, and Omega telemetry ---
                if 'phase_coherence_history' in h5f:
                    gpu_telemetry['phase_coherence_final'] = float(h5f['phase_coherence_history'][-1])
                if 'max_amp_history' in h5f:
                    gpu_telemetry['max_amp_peak'] = float(np.max(h5f['max_amp_history'][:]))
                if 'clamp_fraction_history' in h5f:
                    gpu_telemetry['clamp_fraction_mean'] = float(np.mean(h5f['clamp_fraction_history'][:]))
                if 'omega_sat_history' in h5f:
                    gpu_telemetry['omega_sat_mean'] = float(np.mean(h5f['omega_sat_history'][:]))
                    
        except Exception as e:
            logger.warning(f"Could not extract advanced states from HDF5: {e}")

    # =========================================================
    # --- METRIC CONTRACT ENFORCEMENT ---
    # =========================================================
    try:
        import yaml
        if os.path.exists("metric_contracts.yaml"):
            with open("metric_contracts.yaml", "r") as yf:
                contracts = yaml.safe_load(yf)
            for k, bounds in contracts.get("spectral_fidelity", {}).items():
                val = spectral_fidelity_results.get(k)
                if val is not None:
                    if not (bounds.get("min", -float('inf')) <= val <= bounds.get("max", float('inf'))):
                        spectral_fidelity_results["validation_status"] = "FAIL: NUMERICAL_INVALID"
                        spectral_fidelity_results["primary_harmonic_error"] = 999.0
                        logger.warning(f"Contract violation for {k}: {val}")
    except Exception as e:
        pass
    # =========================================================
    # --- PHASE 4: EARLY REJECTION SHORT-CIRCUIT ---
    # =========================================================
    target_sse = spectral_fidelity_results.get("log_prime_sse", 999.0)

    # --- DEFAULT NULL VARIABLES (in case of Early Rejection) ---
    c4_contrast = 0.0
    ablated_c4_contrast = 0.0
    symmetry_error = None
    shear_stress = None
    metrics_pcs, metrics_pli, metrics_ic = 0.0, 0.0, 1.0
    nonlinear_balance, correlation_length, fractal_dim = None, None, None
    p_value = 1.0
    mean_random_sse = None

    if target_sse > 15.0:
        logger.warning(f"  [Early Rejection] High SSE ({target_sse:.2f}). Skipping heavy scientific metrics.")
    else:
        # =========================================================
        # 🚀 RUN HEAVY METRICS ONLY FOR STRONG CANDIDATES 🚀
        # =========================================================

        # --- PHASE 2.5: TOPOLOGICAL DATA ANALYSIS (TDA) ---
        logger.info("[TDA Engine] Executing Persistent Homology analysis...")
        try:
            if isinstance(rho_final, np.ndarray):
                quantule_events_csv_content, taxonomy_counts = tda_profiler.extract_and_classify_topology(rho_final)
                logger.info(f"  Topological Taxonomy Detected: {taxonomy_counts}")
                tda_csv_filename = f"{config_hash}_quantule_events.csv"
                tda_csv_path = os.path.join(output_dir, tda_csv_filename)
                with open(tda_csv_path, 'w') as f:
                    f.write(quantule_events_csv_content)
                logger.info(f"  Saved TDA Quantule Events CSV: {tda_csv_path}")
            else:
                logger.warning("rho_final is not a valid ndarray; skipping TDA analysis.")
        except Exception as e:
            logger.warning(f"Could not save or compute TDA CSV: {e}")

        # --- Monte Carlo ---
        if rho_final_3d is not None and target_sse < 1.0:
            try:
                p_value, mean_random_sse = monte_carlo_engine.run_monte_carlo_p_value(target_sse, grid_shape=rho_final_3d.shape, n_iterations=500)
            except Exception as e:
                print(f"WARNING: Monte Carlo Statistical Validation failed: {e}", file=sys.stderr)
        logger.info(f"  -> P(prime-lock by chance): {p_value:.6e}")

        # --- PHASE 6: SPDC EMPIRICAL BRIDGE (Quantum Optics) ---
        logger.info("[CEPP v2.0] Generating Empirical Quantum Optics Bridge...")
        if psi_final_3d is not None:
            try:
                jsa_simulated = spdc_empirical_bridge.calculate_joint_spectral_amplitude(psi_final_3d)
                c4_interference = spdc_empirical_bridge.deconvolve_to_c4_interference(jsa_simulated)
                c4_contrast = float(np.max(c4_interference) - np.mean(c4_interference))
                logger.info(f"  -> C4 Interference Contrast: {c4_contrast:.4f}")
            except Exception as e:
                print(f"WARNING: SPDC Empirical Bridge computation failed: {e}", file=sys.stderr)
                
            # --- PHASE ABLATION NULL TEST (Falsifiability) ---
            try:
                logger.info("[Falsifiability] Executing True Phase Ablation (psi_null = |psi|)...")
                psi_null = np.abs(psi_final_3d).astype(np.complex64)
                jsa_null = spdc_empirical_bridge.calculate_joint_spectral_amplitude(psi_null)
                c4_interference_null = spdc_empirical_bridge.deconvolve_to_c4_interference(jsa_null)
                ablated_c4_contrast = float(np.max(c4_interference_null) - np.mean(c4_interference_null))
                logger.info(f"  -> Ablated C4 Contrast: {ablated_c4_contrast:.4f}")
            except Exception as e:
                logger.warning(f"Phase Ablation Null Test failed: {e}")

        # --- STRESS-ENERGY TENSOR VALIDATION ---
        if rho_final_3d is not None:
            phi_final_3d = final_psi
            if phi_final_3d is not None and phi_final_3d.shape == rho_final_3d.shape:
                if not isinstance(phi_final_3d, np.ndarray):
                    phi_final_3d = np.array(phi_final_3d)
                try:
                    T = tensor_validation.construct_T_info(rho_final_3d, phi_final_3d)
                    symmetry_error = tensor_validation.tensor_symmetry_test(T)
                    shear_stress = tensor_validation.perfect_fluid_reduction_test(T)
                    logger.info(f"  Tensor Symmetry Error: {symmetry_error:.3e}")
                    logger.info(f"  Perfect Fluid Shear: {shear_stress:.3e}")
                except Exception as e:
                    logger.warning(f"Tensor validation failed: {e}")
            else:
                logger.warning("phi_final_3d not available or shape mismatch; skipping tensor validation.")

        # --- Compute Aletheia and Collapse Metrics ---
        _rho_for_metrics = rho_final_3d if rho_final_3d is not None else rho_final
        if _rho_for_metrics is not None and isinstance(_rho_for_metrics, np.ndarray):
            try: metrics_pli = calculate_pli(_rho_for_metrics)
            except: pass
            try: metrics_ic = calculate_ic(_rho_for_metrics)
            except: pass
            try: nonlinear_balance = collapse_metrics.compute_nonlinear_balance(_rho_for_metrics)
            except: pass
            try: correlation_length = collapse_metrics.compute_correlation_length(_rho_for_metrics)
            except: pass
            try: fractal_dim = collapse_metrics.compute_fractal_dimension_boxcount(_rho_for_metrics, threshold=0.1)
            except: pass

    # =========================================================
    # --- PHASE 7: LOM TELEMETRY EXTRACTION (ASTE V3.0) ---
    # =========================================================
    validation_status = spectral_fidelity_results.get("validation_status", "FAIL")
    primary_err = spectral_fidelity_results.get("primary_harmonic_error", 999.0)
    collapse_count = 0
    
    # Safely extract simulation bounds from the loaded params_dict
    L_domain = float(params_dict.get("simulation", {}).get("L_domain", 10.0))
    N_grid = int(params_dict.get("simulation", {}).get("N_grid", 64))

    # Trigger ONLY if the run passed fundamental physics, OR locked onto a precise single harmonic
    if validation_status == "PASS" or primary_err < 1.0:
        logger.info("[LOM Telemetry] High fidelity lock detected. Extracting offline collapse events...")
        collapse_count = extract_lom_telemetry(input_path, config_hash, output_dir, L_domain, N_grid, params_dict)
        logger.info(f"  -> Preserved {collapse_count} physical collapse events for fabrication.")

    # INJECT EVENT COUNT INTO PAYLOAD (Prevents double file I/O down the line)
    spectral_fidelity_results["collapse_event_count"] = collapse_count

    # =========================================================
    # --- 4. Assemble & Save Canonical Artifacts ---
    # =========================================================
    
    # RESTORED STRICT SCHEMA: Matches what aste_hunter.py expects
    provenance_artifact = {
        "metadata": {
            "config_hash": config_hash,
            "legacy_hash_reference": param_hash_legacy,
            "timestamp": datetime.utcnow().isoformat() + "Z",
        },
        "spectral_fidelity": spectral_fidelity_results,
        "aletheia_metrics": {
            # Strictly use the pure GPU computation now
            "pcs": gpu_telemetry.get('phase_coherence_final', 0.0),
            "pli": metrics_pli,
            "ic": metrics_ic,
            # GPU Telemetry mapped for Hunter penalties
            "j_info_l2_mean": gpu_telemetry.get('j_info_l2_mean', 0.0),
            "grad_phase_var_mean": gpu_telemetry.get('grad_phase_var_mean', 0.0),
            "max_amp_peak": gpu_telemetry.get('max_amp_peak', 0.0),
            "clamp_fraction_mean": gpu_telemetry.get('clamp_fraction_mean', 0.0),
            "omega_sat_mean": gpu_telemetry.get('omega_sat_mean', 0.0),
            
            # --- NEW: Phantom Filter Epistemic Guardrails ---
            "tau_c": gpu_telemetry.get('tau_c', 0.0),
            "relative_variance": gpu_telemetry.get('relative_variance', 0.0)
        },
        "empirical_bridge": {
            "c4_interference_contrast": c4_contrast,
            "ablated_c4_contrast": ablated_c4_contrast
        },
        "tensor_validation": {
            "symmetry_error": symmetry_error,
            "shear_stress": shear_stress
        },
        "statistical_validation": {
            "p_value": p_value,
            "mean_random_sse": mean_random_sse
        }
    }

    output_filename = f"provenance_{config_hash}.json"
    output_filepath = os.path.join(output_dir, output_filename)

    try:
        with open(output_filepath, 'w') as f:
            json.dump(provenance_artifact, f, indent=4)
        logger.info(f"  Successfully saved validation artifact: {output_filepath}")
        return True
    except Exception as e:
        logger.error(f"Failed to save validation artifact: {e}")
        return False

def main():

    parser = argparse.ArgumentParser(
        description="Spectral Fidelity & Provenance (SFP) Module: Validates simulation output, generates provenance, and computes metrics. Supports batch and single modes."
    )
    parser.add_argument(
        "--input",
        type=str,
        help="Path to the input rho_history.h5 data artifact (single mode)."
    )
    parser.add_argument(
        "--params",
        type=str,
        help="Path to the parameters.json file for this run (single mode)."
    )
    parser.add_argument(
        "--manifest",
        type=str,
        help="Path to a manifest file (.json or .csv) listing input/params pairs for batch mode."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=".",
        help="Directory to save the provenance.json and atlas CSVs. Default: current directory."
    )
    parser.add_argument(
        "--dry-run",
        action='store_true',
        help="Validate arguments and exit without running validation."
    )
    args = parser.parse_args()

    if args.dry_run:
        print("[DRY RUN] Arguments validated. Exiting.")
        return

    if args.manifest:
        if not os.path.exists(args.manifest):
            parser.error(f"Manifest file not found: {args.manifest}")
        jobs = parse_manifest(args.manifest)
        logger.info(f"Batch mode: {len(jobs)} jobs from manifest {args.manifest}")
        n_success = 0
        for i, job in enumerate(jobs):
            logger.info(f"[Batch {i+1}/{len(jobs)}] Processing input: {job['input']} params: {job['params']}")
            ok = run_pipeline_single(job['input'], job['params'], args.output_dir)
            if ok:
                n_success += 1
        logger.info(f"Batch complete: {n_success}/{len(jobs)} jobs succeeded.")
    elif args.input and args.params:
        if not os.path.exists(args.input):
            parser.error(f"Input file not found: {args.input}")
        if not os.path.exists(args.params):
            parser.error(f"Params file not found: {args.params}")
        ok = run_pipeline_single(args.input, args.params, args.output_dir)
        if not ok:
            logger.error("Pipeline failed for single input.")
    else:
        parser.error("Must specify either --manifest for batch mode or both --input and --params for single mode.")


if __name__ == "__main__":
    main()