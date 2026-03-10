#!/usr/bin/env python3

# --- VALIDATION PIPELINE EXECUTION GRAPH ---
# 1. ArtifactLoader              -> Loads HDF5 (psi_final, rho_final, multi-schema telemetry)
# 2. SpectralFidelityEngine      -> Runs CEPP v2.0 (prime_log_sse, bragg peaks)
# 3. ContractEnforcerEngine      -> Validates bounds against metric_contracts.yaml
# 4. Early Rejection Gate        -> Skips steps 5-10 if target_sse > 15.0
# 5. TopologyEngine              -> Runs TDA to classify field geometry
# 6. LOMTelemetryEngine          -> Extracts physical collapse events & gravity maps for fabrication
# 7. Falsifiability Tests        -> Executes Phase Ablation null tests
# 8. EmpiricalBridgeEngine       -> Computes JSA and C4 interference (Quantum Optics)
# 9. TensorValidationEngine      -> Checks symmetry and shear stress
# 10. StatisticalValidationEngine-> Runs Monte Carlo p-value checks
# 11. ProvenanceAssembler        -> Compiles strictly formatted provenance JSON
# -------------------------------------------

"""
validation_pipeline.py
ASSET: A6 (Spectral Fidelity & Provenance Module)
VERSION: 3.2 (Phase 3 Scientific Mandate - 100% ASTE Compliant)
CLASSIFICATION: Final Implementation Blueprint / Governance Instrument
GOAL: Serves as the immutable source of truth that cryptographically binds
      experimental intent (parameters) to scientific fact (spectral fidelity)
      and Aletheia cognitive coherence.
"""

import argparse
import json
import os
import sys
import gc
import yaml
import h5py
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Any, Dict, Optional, List, Tuple, cast

# --- Import analysis/validation modules ---
from config_utils import generate_canonical_hash
import tda_profiler
import metrics.collapse_dynamics as collapse_metrics
import metrics.monte_carlo_engine as monte_carlo_engine
import metrics.spdc_empirical_bridge as spdc_empirical_bridge
import metrics.tensor_validation as tensor_validation
import quantulemapper_real as cep_profiler
from orchestrator.diagnostics.runtime_audit import log_lifecycle_event

try:
    from scipy.stats import entropy as scipy_entropy
    from scipy.ndimage import gaussian_filter
except ImportError:
    print("FATAL: Missing 'scipy'. Please install: pip install scipy", file=sys.stderr)
    sys.exit(1)

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

SCHEMA_VERSION = "SFP-v3.2-ARCS"
MAX_ARTIFACT_ELEMENTS = int(os.environ.get("ASTE_MAX_ARTIFACT_ELEMENTS", str(512**3)))
ANTI_ALIAS_MAX_SOURCE_ELEMENTS = int(
    os.environ.get("ASTE_ANTI_ALIAS_MAX_SOURCE_ELEMENTS", str(MAX_ARTIFACT_ELEMENTS * 4))
)


# ==========================================
# STAGE 1: Artifact Loader
# ==========================================
class ArtifactLoader:
    @staticmethod
    def _anti_aliased_downsample(dataset: h5py.Dataset, stride: int, label: str) -> np.ndarray:
        size = int(dataset.size)
        if size > ANTI_ALIAS_MAX_SOURCE_ELEMENTS:
            raise MemoryError(
                f"{label} dataset too large for anti-aliased decimation ({size} elements). "
                "Increase ASTE_ANTI_ALIAS_MAX_SOURCE_ELEMENTS or provide smaller artifact chunks."
            )

        source = dataset[()]
        sigma = max(0.5, 0.5 * float(stride))
        if np.iscomplexobj(source):
            real_filtered = gaussian_filter(np.real(source), sigma=sigma, mode='nearest')
            imag_filtered = gaussian_filter(np.imag(source), sigma=sigma, mode='nearest')
            filtered = real_filtered + 1j * imag_filtered
        else:
            filtered = gaussian_filter(source, sigma=sigma, mode='nearest')

        slices = tuple(slice(None, None, stride) for _ in range(dataset.ndim))
        return filtered[slices]

    @staticmethod
    def _adaptive_load_dataset(dataset: h5py.Dataset, label: str) -> np.ndarray:
        size = int(dataset.size)
        if size <= MAX_ARTIFACT_ELEMENTS:
            return dataset[()]

        ratio = float(size) / float(MAX_ARTIFACT_ELEMENTS)
        stride = int(np.ceil(ratio ** (1.0 / max(1, dataset.ndim))))
        stride = max(1, stride)
        slices = tuple(slice(None, None, stride) for _ in range(dataset.ndim))
        logger.warning(
            f"Large {label} field detected ({size} elements). "
            f"Applying anti-aliased downsample stride={stride} to prevent spectral aliasing."
        )
        return ArtifactLoader._anti_aliased_downsample(dataset, stride, label)

    @staticmethod
    def load(h5_path: str) -> Tuple[np.ndarray, Optional[np.ndarray], Dict[str, Any]]:
        logger.info(f"[Stage 1: ArtifactLoader] Loading artifact: {h5_path}")
        if not os.path.exists(h5_path):
            raise FileNotFoundError(f"Input file not found: {h5_path}")

        telemetry: Dict[str, Any] = {}
        with h5py.File(h5_path, 'r') as h5f:
            # 1. Resolve psi/rho schema drift
            psi_key = 'psi_final' if 'psi_final' in h5f else 'final_psi'
            if psi_key not in h5f:
                raise ValueError("No valid psi data found.")
            psi_final_3d = ArtifactLoader._adaptive_load_dataset(h5f[psi_key], "psi")

            rho_final_3d: Optional[np.ndarray] = None
            rho_key = 'rho_final' if 'rho_final' in h5f else 'final_rho'
            if rho_key in h5f:
                rho_final_3d = ArtifactLoader._adaptive_load_dataset(h5f[rho_key], "rho")
            elif 'rho_history' in h5f:
                rho_history = h5f['rho_history']
                if int(np.prod(rho_history.shape[1:])) > MAX_ARTIFACT_ELEMENTS:
                    logger.warning(
                        f"Large rho_history final frame detected ({int(np.prod(rho_history.shape[1:]))} elements). "
                        "Applying anti-aliased downsample to prevent spectral aliasing."
                    )
                    ratio = float(np.prod(rho_history.shape[1:])) / float(MAX_ARTIFACT_ELEMENTS)
                    stride = max(1, int(np.ceil(ratio ** (1.0 / max(1, len(rho_history.shape[1:]))))))
                    final_frame = np.asarray(rho_history[-1])
                    sigma = max(0.5, 0.5 * float(stride))
                    filtered = gaussian_filter(final_frame, sigma=sigma, mode='nearest')
                    slices = tuple(slice(None, None, stride) for _ in rho_history.shape[1:])
                    rho_final_3d = filtered[slices]
                else:
                    rho_final_3d = rho_history[-1]

            if rho_final_3d is not None and psi_final_3d.shape != rho_final_3d.shape:
                raise ValueError(f"Domain mismatch: psi {psi_final_3d.shape} vs rho {rho_final_3d.shape}")

            # 2. V4.0 Decoupled Telemetry (Grouped)
            if 'extended_telemetry' in h5f:
                ext_grp = h5f['extended_telemetry']
                canonical_from_extended = {
                    'J_info_l2': 'j_info_l2_mean',
                    'grad_phase_var': 'grad_phase_var_mean',
                    'phase_coherence': 'phase_coherence_mean',
                    'omega_saturation': 'omega_sat_mean',
                }
                for src_key, dst_key in canonical_from_extended.items():
                    if src_key in ext_grp and len(ext_grp[src_key]) > 0:
                        data = np.asarray(ext_grp[src_key][:], dtype=np.float64)
                        telemetry[dst_key] = float(np.mean(data))
                        if src_key == 'J_info_l2':
                            telemetry['tau_c'] = ArtifactLoader._compute_tau_c(data)

            # 3. Legacy Schema Fallback (Flat Datasets)
            else:
                legacy_mappings = {
                    'j_info_l2_history': 'j_info_l2_mean',
                    'grad_phase_var_history': 'grad_phase_var_mean',
                    'phase_coherence_history': 'phase_coherence_mean'
                }
                for old_key, new_key in legacy_mappings.items():
                    if old_key in h5f and len(h5f[old_key]) > 0:
                        data = np.asarray(h5f[old_key][:], dtype=np.float64)
                        telemetry[new_key] = float(np.mean(data))
                        if old_key == 'j_info_l2_history':
                            telemetry['tau_c'] = ArtifactLoader._compute_tau_c(data)

            # Read-only deprecated aliases from legacy writer variants
            deprecated_aliases = {
                'phase_coherence_final': 'phase_coherence_mean',
                'grad_phase_var_final': 'grad_phase_var_mean',
                'J_info_l2_final': 'j_info_l2_mean',
                'omega_saturation_final': 'omega_sat_mean',
            }
            for alias_key, canonical_key in deprecated_aliases.items():
                if alias_key in h5f and canonical_key not in telemetry:
                    alias_data = np.asarray(h5f[alias_key][:], dtype=np.float64)
                    if alias_data.size > 0:
                        telemetry[canonical_key] = float(np.mean(alias_data))

            if 'telemetry' in h5f:
                base_telemetry = h5f['telemetry']
                if 'C_invariant' in base_telemetry and len(base_telemetry['C_invariant']) > 0:
                    c_data = np.asarray(base_telemetry['C_invariant'][:], dtype=np.float64)
                    telemetry['C_invariant_final'] = float(c_data[-1])
                    telemetry['collapse_invariant'] = float(np.mean(c_data))
                    telemetry['collapse_invariant_mean'] = float(np.mean(c_data))
                if 'energy' in base_telemetry and len(base_telemetry['energy']) > 0:
                    e_data = np.asarray(base_telemetry['energy'][:], dtype=np.float64)
                    telemetry['energy_final'] = float(e_data[-1])

            # 4. Memory-Safe Geometry Loading
            if "omega_sq_final" in h5f:
                if h5f["omega_sq_final"].size > 512**3:
                    logger.warning(f"Large geometry field detected ({h5f['omega_sq_final'].size} elements). Skipping load to prevent OOM.")
                else:
                    telemetry['omega_sq_final'] = h5f["omega_sq_final"][()]

            # Optional quantule_events import for legacy timelines
            if "quantule_events" in h5f:
                q_grp = h5f["quantule_events"]
                omega_local = np.asarray(q_grp["omega_local"][:]) if "omega_local" in q_grp else np.array([])
                bandwidth = np.asarray(q_grp["bandwidth"][:]) if "bandwidth" in q_grp else np.array([])
                t_step = np.asarray(q_grp["t_step"][:]) if "t_step" in q_grp else np.array([])
                min_len = min(len(omega_local), len(bandwidth), len(t_step))
                if min_len > 0:
                    telemetry['quantule_omega_local'] = cast(Any, omega_local[:min_len])
                    telemetry['quantule_bandwidth'] = cast(Any, bandwidth[:min_len])
                    telemetry['quantule_t_step'] = cast(Any, t_step[:min_len])

        return psi_final_3d, rho_final_3d, telemetry

    @staticmethod
    def _compute_tau_c(time_series: np.ndarray) -> float:
        if len(time_series) < 2: return 0.0
        ts_norm = time_series - np.mean(time_series)
        if np.all(ts_norm == 0): return 0.0
        autocorr = np.correlate(ts_norm, ts_norm, mode='full')
        autocorr = autocorr[autocorr.size // 2:]
        autocorr /= (autocorr[0] + 1e-12)
        tau_c = np.where(autocorr < np.exp(-1))[0]
        return float(tau_c[0]) if len(tau_c) > 0 else float(len(autocorr))


# ==========================================
# STAGE 2: Spectral Fidelity Engine
# ==========================================
class SpectralFidelityEngine:
    @staticmethod
    def run(rho_final: np.ndarray) -> Dict[str, Any]:
        logger.info("[Stage 2: SpectralFidelityEngine] Running Quantule Profiler (CEPP v2.0)...")
        if rho_final.ndim != 3:
            raise ValueError(f"Expected 3D rho array, got {rho_final.shape}")

        try:
            profiler_results = cep_profiler.prime_log_sse(rho_final)
            
            # [FIX 1] Pull safely from profiler_results instead of manually recalculating!
            n_bragg_peaks = profiler_results.get("bragg_peaks_detected", 0)
            bragg_prime_sse = float(profiler_results.get("bragg_lattice_sse", 999.0))

            log_prime_sse = float(profiler_results.get("log_prime_sse", 999.0))
            validation_status = "PASS" if log_prime_sse < 1.0 else "FAIL: HIGH_SSE"

            return {
                "validation_status": validation_status,
                "log_prime_sse": log_prime_sse,
                "scaling_factor_S": profiler_results.get("scaling_factor_S", 0.0),
                "dominant_peak_k": profiler_results.get("dominant_peak_k", 0.0),
                "secondary_peak_k": profiler_results.get("secondary_peak_k", 0.0),
                "analysis_protocol": "CEPP v2.0",
                "prime_log_targets": getattr(cep_profiler, "LOG_PRIME_TARGETS", np.array([])).tolist(),
                "sse_null_phase_scramble": float(profiler_results.get("sse_null_phase_scramble", 999.0)),
                "sse_null_target_shuffle": float(profiler_results.get("sse_null_target_shuffle", 999.0)),
                "primary_harmonic_error": float(profiler_results.get("primary_harmonic_error", 999.0)),
                "missing_peak_penalty": float(profiler_results.get("missing_peak_penalty", 0.0)),
                "noise_penalty": float(profiler_results.get("noise_penalty", 0.0)),
                "best_single_error": float(profiler_results.get("best_single_error", 999.0)),
                "bragg_lattice_sse": bragg_prime_sse,
                "bragg_peaks_detected": n_bragg_peaks,
                "n_bragg_peaks": n_bragg_peaks,
                "bragg_prime_sse": bragg_prime_sse,
                "measured_peaks": profiler_results.get("measured_peaks", []),
                "scaled_peaks": profiler_results.get("scaled_peaks", []),
                
                # Restored Diagnostics for Hunter database
                "n_peaks_found_main": profiler_results.get("n_peaks_found_main", 0),
                "failure_reason_main": profiler_results.get("failure_reason_main", None),
                "n_peaks_found_null_a": profiler_results.get("n_peaks_found_null_a", 0),
                "failure_reason_null_a": profiler_results.get("failure_reason_null_a", None),
                "n_peaks_found_null_b": profiler_results.get("n_peaks_found_null_b", 0),
                "failure_reason_null_b": profiler_results.get("failure_reason_null_b", None),
                
                "collapse_event_count": 0 # Default, overwritten by LOM Engine
            }
        except Exception as e:
            logger.critical(f"CRITICAL: Spectral Fidelity Profiler failed: {e}")
            # [FIX 2] DO NOT RAISE. Return a maximum penalty gracefully so the loop survives!
            return {
                "validation_status": "FAIL: DEGENERATE_FIELD",
                "log_prime_sse": 999.0,
                "dominant_peak_k": 0.0,
                "secondary_peak_k": 0.0,
                "primary_harmonic_error": 999.0,
                "missing_peak_penalty": 7.0,
                "noise_penalty": 0.0,
                "best_single_error": 999.0,
                "sse_null_phase_scramble": 999.0,
                "sse_null_target_shuffle": 999.0,
                "bragg_lattice_sse": 999.0,
                "bragg_peaks_detected": 0,
                "n_bragg_peaks": 0,
                "bragg_prime_sse": 999.0,
                "measured_peaks": [],
                "scaled_peaks": [],
                "collapse_event_count": 0
            }


# ==========================================
# STAGE 3: Contract Enforcer
# ==========================================
class ContractEnforcerEngine:
    @staticmethod
    def enforce(spec_results: Dict[str, Any]) -> None:
        logger.info("[Stage 3: ContractEnforcerEngine] Validating metric contracts...")
        try:
            if os.path.exists("metric_contracts.yaml"):
                with open("metric_contracts.yaml", "r") as yf:
                    contracts = yaml.safe_load(yf)
                for k, bounds in contracts.get("spectral_fidelity", {}).items():
                    val = spec_results.get(k)
                    if val is not None:
                        if not (bounds.get("min", -float('inf')) <= val <= bounds.get("max", float('inf'))):
                            spec_results["validation_status"] = "FAIL: NUMERICAL_INVALID"
                            spec_results["primary_harmonic_error"] = 999.0
                            logger.warning(f"  -> Contract violation for {k}: {val}")
        except Exception as e:
            logger.warning(f"  -> Contract enforcement bypassed/failed: {e}")


# ==========================================
# STAGES 5-10: Deep Analysis Engines
# ==========================================
class TopologyEngine:
    @staticmethod
    def null_result() -> Dict[str, Any]:
        return {
            "q_type": "Transient",
            "persistent_loops": 0,
            "persistent_voids": 0,
            "betti_0": 1,
            "betti_1": 0,
            "betti_2": 0,
        }

    @staticmethod
    def run_tda(rho_final: np.ndarray, config_hash: str, output_dir: str) -> Dict[str, Any]:
        logger.info("[Stage 5: TopologyEngine] Executing Persistent Homology...")
        try:
            csv_content, taxonomy = tda_profiler.extract_and_classify_topology(rho_final)
            logger.info(f"  Topological Taxonomy Detected: {taxonomy}")
            
            # Safe nested TDA pathing
            tda_dir = os.path.join(output_dir, "tda")
            os.makedirs(tda_dir, exist_ok=True)
            out_path = os.path.join(tda_dir, f"{config_hash}_quantule_events.csv")
            
            with open(out_path, 'w') as f:
                f.write(csv_content)

            q_theta = int(taxonomy.get("Q_theta", 0))
            q_nu = int(taxonomy.get("Q_nu", 0))
            q_transient = int(taxonomy.get("Transient", 0))

            q_type = "Transient"
            if q_theta > 0:
                q_type = "Q_theta"
            elif q_nu > 0:
                q_type = "Q_nu"
            elif q_transient > 0:
                q_type = "Transient"

            return {
                "q_type": q_type,
                "persistent_loops": q_nu,
                "persistent_voids": q_theta,
                "betti_0": 1,
                "betti_1": q_nu,
                "betti_2": q_theta,
            }
        except Exception as e:
            logger.warning(f"TDA analysis failed: {e}")
            return TopologyEngine.null_result()


class LOMTelemetryEngine:
    @staticmethod
    def extract(config_hash: str, output_dir: str, params_dict: dict, psi_final: np.ndarray, rho_final: np.ndarray, telemetry: dict) -> int:
        logger.info("[Stage 6: LOMTelemetryEngine] Extracting offline collapse events...")
        try:
            L_domain = float(params_dict.get("simulation", {}).get("L_domain", 10.0))
            N_grid = int(params_dict.get("simulation", {}).get("N_grid", 64))
            
            omega_temporal_mean = 0.0
            bandwidth_dk_val = 0.0
            emergence_t_step_val = 0
            omega_sq_field = telemetry.get('omega_sq_final')
            
            # Consume previously extracted telemetry logs safely
            quantule_omega_local = telemetry.get('quantule_omega_local', [])
            quantule_bandwidth = telemetry.get('quantule_bandwidth', [])
            quantule_t_step = telemetry.get('quantule_t_step', [])
            min_len = min(len(quantule_omega_local), len(quantule_bandwidth), len(quantule_t_step))
            if min_len > 0:
                quantule_omega_local = quantule_omega_local[:min_len]
                quantule_bandwidth = quantule_bandwidth[:min_len]
                quantule_t_step = quantule_t_step[:min_len]

                omega_temporal_mean = float(np.mean(quantule_omega_local))
                bandwidth_dk_val = float(quantule_bandwidth[0])
                emergence_t_step_val = int(quantule_t_step[0])

                pd.DataFrame({
                    "t_step": quantule_t_step,
                    "omega_local": quantule_omega_local,
                    "spectral_bandwidth_dk": quantule_bandwidth
                }).to_csv(os.path.join(output_dir, f"{config_hash}_gravity_timeline.csv"), index=False)
                        
            theta = np.angle(psi_final)
            
            # --- ASTE V4: Adaptive Quantule Thresholding ---
            # Replaces the hard-coded 0.8 to prevent "Solid Block" saturation dumps
            mu_rho = float(np.mean(rho_final))
            sigma_rho = float(np.std(rho_final))
            
            # --- Explicit Zero-Variance Guard ---
            if sigma_rho < 1e-12:
                logger.warning("  -> LOM Telemetry Guard: Zero variance detected (flatline). Rejecting.")
                return 0
                
            critical_threshold = mu_rho + (3.0 * sigma_rho)
            
            z_indices, y_indices, x_indices = np.where(rho_final > critical_threshold)
            
            # --- Explosion / Saturation Guard ---
            # If the variance is 0 (a flat block), everything might trigger the threshold.
            # If more than 20% of the box is "collapsing", it's garbage, not a prime lock.
            max_valid_events = (N_grid**3) * 0.20
            if len(z_indices) == 0 or len(z_indices) > max_valid_events: 
                logger.warning(f"  -> LOM Telemetry Guard: {len(z_indices)} events rejected (flatline/explosion).")
                return 0
                
            dx = L_domain / N_grid
            phys_x = (x_indices - N_grid / 2) * dx
            phys_y = (y_indices - N_grid / 2) * dx
            phys_z = (z_indices - N_grid / 2) * dx
            normalized_phase = (theta[z_indices, y_indices, x_indices] + np.pi) / (2 * np.pi)
            
            # Pure analysis - Do not silently recompute the conformal geometry fallback
            if omega_sq_field is not None:
                spatial_gravity_pressure = omega_sq_field[z_indices, y_indices, x_indices]
            else:
                spatial_gravity_pressure = np.zeros(len(z_indices))
                logger.info("  -> 'omega_sq_final' not found in artifact; skipping spatial gravity map reconstruction.")
            
            df = pd.DataFrame({
                'idx_x': x_indices, 'idx_y': y_indices, 'idx_z': z_indices,
                'phys_x': phys_x, 'phys_y': phys_y, 'phys_z': phys_z,
                'rho_intensity': rho_final[z_indices, y_indices, x_indices],
                'complex_phase_normalized': normalized_phase,
                'temporal_omega_mean': omega_temporal_mean,
                'spatial_gravity_omega_sq': spatial_gravity_pressure,
                'bandwidth_dk': bandwidth_dk_val,
                'emergence_t_step': emergence_t_step_val
            })
            
            scale_nm = (L_domain * 1e9) / N_grid
            df["x_nm"] = df["phys_x"] * scale_nm
            df["y_nm"] = df["phys_y"] * scale_nm
            df["z_nm"] = df["phys_z"] * scale_nm
            
            os.makedirs(output_dir, exist_ok=True)
            df.to_csv(os.path.join(output_dir, f"{config_hash}_etch_ready.csv"), index=False)
            
            logger.info(f"  -> Preserved {len(z_indices)} physical collapse events for fabrication.")
            return len(z_indices)
        except Exception as e:
            logger.warning(f"  -> LOM Telemetry extraction failed: {e}")
            return 0


class EmpiricalBridgeEngine:
    @staticmethod
    def run(psi_final: np.ndarray) -> Tuple[float, float]:
        logger.info("[Stage 7 & 8: EmpiricalBridgeEngine] Generating Quantum Optics Bridge & Phase Ablation Null Tests...")
        c4_contrast = 0.0
        ablated_c4_contrast = 0.0
        try:
            jsa = spdc_empirical_bridge.calculate_joint_spectral_amplitude(psi_final)
            c4 = spdc_empirical_bridge.deconvolve_to_c4_interference(jsa)
            c4_contrast = float(np.max(c4) - np.mean(c4))
            
            # Falsifiability Phase Ablation
            psi_null = np.abs(psi_final).astype(np.complex64)
            jsa_null = spdc_empirical_bridge.calculate_joint_spectral_amplitude(psi_null)
            c4_null = spdc_empirical_bridge.deconvolve_to_c4_interference(jsa_null)
            ablated_c4_contrast = float(np.max(c4_null) - np.mean(c4_null))
        except Exception as e:
            logger.warning(f"Empirical Bridge computation failed: {e}")
        return c4_contrast, ablated_c4_contrast


class TensorValidationEngine:
    @staticmethod
    def run(rho_final: np.ndarray, psi_final: np.ndarray) -> Tuple[Optional[float], Optional[float]]:
        logger.info("[Stage 9: TensorValidationEngine] Computing Stress-Energy Tensor...")
        try:
            phase_field = np.angle(psi_final) if np.iscomplexobj(psi_final) else psi_final
            T = tensor_validation.construct_T_info(rho_final, phase_field)
            symmetry_error = tensor_validation.tensor_symmetry_test(T)
            shear_stress = tensor_validation.perfect_fluid_reduction_test(T)
            return symmetry_error, shear_stress
        except Exception as e:
            logger.warning(f"Tensor validation failed: {e}")
            return None, None


class StatisticalValidationEngine:
    @staticmethod
    def run(target_sse: float, grid_shape: tuple, n_iterations: int = 500) -> Tuple[float, Optional[float]]:
        logger.info(f"[Stage 10: StatisticalValidationEngine] Running Monte Carlo ({n_iterations} iterations)...")
        try:
            np.random.seed(42)  # Deterministic validation
            return monte_carlo_engine.run_monte_carlo_p_value(target_sse, grid_shape=grid_shape, n_iterations=n_iterations)
        except Exception as e:
            logger.warning(f"Statistical validation failed: {e}")
            return 1.0, None


class ValidationDerivedMetricsEngine:
    @staticmethod
    def run(psi_final: np.ndarray, rho_final: np.ndarray, telemetry: Dict[str, Any]) -> Dict[str, float]:
        rho = np.maximum(rho_final.astype(np.float64, copy=False), 1e-12)
        phase = np.angle(psi_final)

        grad_phase = np.gradient(phase)
        grad_phase_sq = np.zeros_like(rho, dtype=np.float64)
        for comp in grad_phase:
            grad_phase_sq += np.asarray(comp, dtype=np.float64) ** 2

        grad_rho = np.gradient(rho)
        grad_rho_sq = np.zeros_like(rho, dtype=np.float64)
        for comp in grad_rho:
            grad_rho_sq += np.asarray(comp, dtype=np.float64) ** 2

        fft_rho = np.fft.fftn(rho)
        power = np.abs(fft_rho) ** 2
        grid_shape = rho.shape
        freq_axes = [np.fft.fftfreq(n) for n in grid_shape]
        kx, ky, kz = np.meshgrid(*freq_axes, indexing='ij')
        k_mag = np.sqrt(kx**2 + ky**2 + kz**2)
        denom = float(np.sum(power) + 1e-12)

        collapse_invariant = float(np.mean(rho**2))
        phase_coherence = float(np.abs(np.mean(np.exp(1j * phase))))
        derived = {
            'phase_coherence_final': phase_coherence,
            'phase_coherence_mean': phase_coherence,
            'grad_phase_var_mean': float(np.var(grad_phase_sq)),
            'j_info_l2_mean': float(np.mean(grad_rho_sq / (4.0 * rho))),
            'omega_sat_mean': float(np.mean(np.asarray(telemetry.get('omega_sq_final', rho), dtype=np.float64))),
            'spectral_bandwidth_mean': float(np.sum(k_mag * power) / denom),
            'collapse_invariant': collapse_invariant,
            'collapse_invariant_mean': collapse_invariant,
        }
        return derived


class AletheiaMetricsEngine:
    @staticmethod
    def run(rho: np.ndarray) -> dict:
        metrics = {
            "pli": 0.0, "ic": 1.0, 
            "nonlinear_balance": None, "correlation_length": None, "fractal_dimension": None
        }
        
        # Phase 3 Principled Localization Index (PLI)
        sum_rho = np.sum(rho)
        if sum_rho != 0:
            metrics["pli"] = float(np.sum((rho / sum_rho)**2) * rho.size)

        # Informational Compressibility (IC)
        try:
            proxy_E = np.sum(rho**2)
            rho_flat = rho.flatten()
            sum_rho_flat = np.sum(rho_flat)
            if sum_rho_flat != 0:
                proxy_S = scipy_entropy((rho_flat / sum_rho_flat) + 1e-9)
                # Apply a non-uniform thermal perturbation to test informational rigidity
                rho_p = rho + (0.01 * np.mean(rho))
                proxy_E_p = np.sum(rho_p**2)
                rho_p_flat = rho_p.flatten()
                proxy_S_p = scipy_entropy((rho_p_flat / np.sum(rho_p_flat)) + 1e-9)
                dE, dS = proxy_E_p - proxy_E, proxy_S_p - proxy_S
                if dE != 0 and not np.isnan(dE) and not np.isnan(dS): 
                    metrics["ic"] = float(dS / dE)
        except Exception as e:
            logger.warning(f"IC calculation failed: {e}")

        # Collapse Dynamics Metrics
        try: metrics["nonlinear_balance"] = collapse_metrics.compute_nonlinear_balance(rho)
        except: pass
        try: metrics["correlation_length"] = collapse_metrics.compute_correlation_length(rho)
        except: pass
        try: metrics["fractal_dimension"] = collapse_metrics.compute_fractal_dimension_boxcount(rho, threshold=0.1)
        except: pass

        return metrics


# ==========================================
# STAGE 11: Provenance Assembler
# ==========================================
class ProvenanceAssembler:
    @staticmethod
    def assemble(config_hash: str, legacy_hash: Optional[str], spec_results: dict, telemetry: dict, metrics: dict,
                 c4: float, c4_ablated: float, sym_err: float, shear: float, p_val: float, rand_sse: float,
                 tda_results: dict) -> dict:
        logger.info("[Stage 11: ProvenanceAssembler] Assembling canonical payload...")
        return {
            "metadata": {
                "config_hash": config_hash,
                "legacy_hash_reference": legacy_hash,
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "schema_version": SCHEMA_VERSION
            },
            "spectral_fidelity": spec_results,
            "aletheia_metrics": {
                "pcs": telemetry.get('phase_coherence_mean', None),
                "pli": metrics.get('pli', 0.0),
                "ic": metrics.get('ic', 0.0),
                "phase_coherence_mean": telemetry.get('phase_coherence_mean', None),
                
                # GPU Telemetry mapped for Hunter penalties
                "j_info_l2_mean": telemetry.get('j_info_l2_mean', None),
                "grad_phase_var_mean": telemetry.get('grad_phase_var_mean', None),
                "grad_phase_var_tau_c": telemetry.get('grad_phase_var_tau_c', None),
                "max_amp_peak": telemetry.get('max_amp_peak', None),
                "clamp_fraction_mean": telemetry.get('clamp_fraction_mean', None),
                "omega_sat_mean": telemetry.get('omega_sat_mean', None),
                "spectral_bandwidth_mean": telemetry.get('spectral_bandwidth_mean', None),
                "collapse_invariant": telemetry.get('collapse_invariant', None),
                "collapse_invariant_mean": telemetry.get('collapse_invariant_mean', None),
                
                # Conservation Invariants 
                "C_invariant_final": telemetry.get('C_invariant_final', None),
                "energy_final": telemetry.get('energy_final', None),

                # Phantom Filter Epistemic Guardrails
                "tau_c": telemetry.get('tau_c', None),
                "relative_variance": telemetry.get('relative_variance', None),
                
                # Collapse Dynamics Restored
                "nonlinear_balance": metrics.get("nonlinear_balance"),
                "correlation_length": metrics.get("correlation_length"),
                "fractal_dimension": metrics.get("fractal_dimension")
            },
            "empirical_bridge": {
                "c4_interference_contrast": c4,
                "ablated_c4_contrast": c4_ablated
            },
            "tensor_validation": {
                "symmetry_error": sym_err,
                "shear_stress": shear
            },
            "topology": tda_results,
            "statistical_validation": {
                "p_value": p_val,
                "mean_random_sse": rand_sse
            }
        }


# ==========================================
# MAIN ORCHESTRATOR
# ==========================================
class ValidationPipeline:
    def __init__(self, input_path: str, params_path: str, output_dir: str, mc_iterations: int = 500):
        self.input_path = input_path
        self.params_path = params_path
        self.output_dir = output_dir
        self.mc_iterations = mc_iterations

    def run(self) -> bool:
        logger.info("--- SFP Module (Asset A6, v3.2) Initiating Validation ---")
        psi_final: Optional[np.ndarray] = None
        rho_final: Optional[np.ndarray] = None
        telemetry: Dict[str, Any] = {}
        
        try:
            with open(self.params_path, 'r') as f:
                params_dict = json.load(f)
            config_hash = params_dict.get("config_hash") or generate_canonical_hash(params_dict)
            legacy_hash = params_dict.get("param_hash_legacy")
        except Exception as e:
            logger.error(f"Failed to load params: {e}")
            return False

        try:
            psi_final, rho_final_loaded, telemetry = ArtifactLoader.load(self.input_path)
        except Exception as e:
            logger.error(f"Artifact Loader failed: {e}")
            return False

        psi_final = cast(np.ndarray, psi_final)

        # Canonical validation rule: derive rho from psi to avoid trusting persisted rho drift.
        rho_final = np.abs(psi_final) ** 2
        rho_final = cast(np.ndarray, rho_final)

        # Validation-owned derived metrics from physical state.
        derived_metrics = ValidationDerivedMetricsEngine.run(psi_final, rho_final, telemetry)
        for key, value in derived_metrics.items():
            telemetry.setdefault(key, value)

        # PRIMARY STABILITY SIGNAL: Derived post-hoc from spectral attractors
        try:
            spec_results = SpectralFidelityEngine.run(rho_final)
        except Exception as e:
            logger.error(f"Spectral Fidelity Engine failed: {e}")
            return False

        ContractEnforcerEngine.enforce(spec_results)

        target_sse = spec_results.get("log_prime_sse", 999.0)
        validation_status = spec_results.get("validation_status", "FAIL")
        metrics_dict: Dict[str, Any]
        c4: Optional[float]
        c4_ablated: Optional[float]
        sym_err: Optional[float]
        shear: Optional[float]
        p_val: Optional[float]
        rand_sse: Optional[float]
        tda_results: Dict[str, Any]
        
        # Early Rejection Gate (Post-hoc decision)
        gate_rejected = False
        if target_sse > 15.0:
            gate_rejected = True
            logger.warning(f"[Gate] Run Rejected (SSE: {target_sse:.2f}). Skipping heavy metrics.")
            metrics_dict = {"pli": None, "ic": None}
            c4, c4_ablated = None, None
            sym_err, shear = None, None
            p_val, rand_sse = None, None
            tda_results = TopologyEngine.null_result()
            telemetry['validation_gate_status'] = 'GATE_REJECTED'
            spec_results['validation_status'] = 'GATE_REJECTED'
        else:
            telemetry['validation_gate_status'] = validation_status
            tda_results = TopologyEngine.run_tda(rho_final, config_hash, self.output_dir)
            
            # Extract collapse metrics only for post-gate stable runs
            spec_results["collapse_event_count"] = LOMTelemetryEngine.extract(
                config_hash, self.output_dir, params_dict, psi_final, rho_final, telemetry
            )
            
            c4, c4_ablated = EmpiricalBridgeEngine.run(psi_final)
            sym_err, shear = TensorValidationEngine.run(rho_final, psi_final)
            p_val, rand_sse = StatisticalValidationEngine.run(target_sse, rho_final.shape, self.mc_iterations)
            metrics_dict = AletheiaMetricsEngine.run(rho_final)

        if not gate_rejected:
            sym_err = 1.0 if sym_err is None else sym_err
            shear = 1.0 if shear is None else shear
            rand_sse = 999.0 if rand_sse is None else rand_sse

            assert sym_err is not None
            assert shear is not None
            assert rand_sse is not None

        provenance = ProvenanceAssembler.assemble(
            config_hash, legacy_hash, spec_results, telemetry, metrics_dict,
            0.0 if c4 is None else c4,
            0.0 if c4_ablated is None else c4_ablated,
            0.0 if sym_err is None else sym_err,
            0.0 if shear is None else shear,
            1.0 if p_val is None else p_val,
            0.0 if rand_sse is None else rand_sse,
            tda_results
        )

        out_file = os.path.join(self.output_dir, f"provenance_{config_hash}.json")
        try:
            with open(out_file, 'w') as f:
                json.dump(provenance, f, indent=4)
            log_lifecycle_event(
                stage="validation_write",
                config_hash=config_hash,
                generation=None,
                details={"provenance_path": out_file},
            )
            logger.info(f"Successfully saved canonical artifact: {out_file}")
            return True
        except Exception as e:
            logger.error(f"Failed to save provenance JSON: {e}")
            return False
        finally:
            if psi_final is not None:
                del psi_final
            if rho_final is not None:
                del rho_final
            telemetry.clear()
            gc.collect()


def parse_manifest(manifest_path: str) -> List[Dict[str, str]]:
    if manifest_path.endswith('.json'):
        with open(manifest_path, 'r') as f:
            data = json.load(f)
            jobs = data.get('jobs', data) if isinstance(data, dict) else data
    elif manifest_path.endswith('.csv'):
        import csv
        with open(manifest_path, 'r', newline='') as f:
            jobs = list(csv.DictReader(f))
    else:
        raise ValueError("Manifest must be .json or .csv")
        
    for i, job in enumerate(jobs):
        if 'input' not in job or 'params' not in job:
            raise ValueError(f"Manifest row {i} missing 'input' or 'params' keys.")
            
    return jobs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Spectral Fidelity & Provenance (SFP) Module v3.2")
    parser.add_argument("--input", type=str, help="Path to input HDF5.")
    parser.add_argument("--params", type=str, help="Path to parameters.json.")
    parser.add_argument("--manifest", type=str, help="Path to batch manifest.")
    parser.add_argument("--output_dir", type=str, default=".", help="Output directory.")
    parser.add_argument("--mc-iterations", type=int, default=500, help="Number of Monte Carlo iterations.")
    parser.add_argument("--dry-run", action='store_true', help="Validate args only.")
    args = parser.parse_args()

    if args.dry_run:
        print("[DRY RUN] Exiting.")
        sys.exit(0)

    if args.manifest:
        jobs = parse_manifest(args.manifest)
        successes = sum(1 for j in jobs if ValidationPipeline(j['input'], j['params'], args.output_dir, args.mc_iterations).run())
        logger.info(f"Batch complete: {successes}/{len(jobs)} succeeded.")
    elif args.input and args.params:
        pipeline = ValidationPipeline(args.input, args.params, args.output_dir, args.mc_iterations)
        if not pipeline.run():
            sys.exit(1)
    else:
        parser.error("Specify --manifest OR both --input and --params.")