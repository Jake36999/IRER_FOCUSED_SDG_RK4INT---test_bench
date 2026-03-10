#!/usr/bin/env python3

"""
worker_cupy.py
CLASSIFICATION: GPU-Accelerated Simulation Worker (ASTE V3.1)
GOAL: Offloads the 3D complex Psi field to Nvidia VRAM using CuPy.
      Optimized for FP32 (Single Precision) to maximize GTX 1080 TeraFLOPS.
      Eliminates PCIe Host-Device sync bottlenecks.
      Features In-Place RK4, Dynamic Omega, and mathematically safe J_info extraction.
"""

import numpy as np
import cupy as cp
import h5py
import json
import os
import sys
import argparse
import time
import logging
from datetime import datetime
from typing import NamedTuple, Tuple, Dict, Any

# --- Emergent Gravity Bridge ---
sys.path.append(os.path.join(os.path.dirname(__file__), 'gravity'))
try:
    from unified_omega import derive_stable_conformal_factor
except ImportError:
    derive_stable_conformal_factor = None
assert derive_stable_conformal_factor is not None, "CRITICAL ERROR: Failed to load unified_omega. Halting to prevent flat-space physics."

class PsiSimStateGPU(NamedTuple):
    psi: cp.ndarray
    k_vectors: Tuple[cp.ndarray, cp.ndarray, cp.ndarray]
    k_squared: cp.ndarray

def psi_full_rhs(psi, t, k_squared, params):
    # --- FP32 HEADROOM PROTECTION ---
    MAX_AMPLITUDE = 20.0
    amp = cp.abs(psi)
    scale_factor = cp.minimum(1.0, MAX_AMPLITUDE / (amp + 1e-12))
    psi_safe = psi * scale_factor

    # 1. DYNAMIC OMEGA (Fixes RK4 Lag)
    rho = cp.abs(psi_safe)**2
    omega_sq = derive_stable_conformal_factor(rho, params)

    # 2. Spectral Laplacian
    psi_k = cp.fft.fftn(psi_safe)
    flat_laplacian_real = cp.fft.ifftn(-k_squared * psi_k)
    # ZERO-OVERHEAD ARMOR: Added 1e-12 to prevent stiff division
    covariant_laplacian = flat_laplacian_real / (omega_sq + 1e-12) 
    
    # 3. Apply physics parameters
    D = float(params.get('param_D', 1.0))
    eta = float(params.get('param_eta', 1.0))
    a_coupling = float(params.get('param_a_coupling', 1.0))
    splash_coupling = float(params.get('param_splash_coupling', 0.0))
    splash_fraction = float(params.get('param_splash_fraction', -0.5))
    rho_vac = float(params.get('param_rho_vac', 1.0))

    laplacian_term = D * covariant_laplacian
    dissipation_term = -eta * psi_safe
    cubic_term = a_coupling * psi_safe * rho
    quartic_term = splash_coupling * psi_safe * (rho**2)
    splash_frac_term = splash_fraction * psi_safe * (rho**3)
    vacuum_term = 1j * rho_vac * psi_safe

    return laplacian_term + dissipation_term + cubic_term + quartic_term + splash_frac_term + vacuum_term

def rk4_step_psi_full(psi, t, dt, k_squared, params, buffers):
    # Unpack pre-allocated memory buffers to prevent VRAM fragmentation
    psi_temp, k1, k2, k3, k4 = buffers
    
    # k1
    k1[:] = psi_full_rhs(psi, t, k_squared, params)
    
    # k2
    cp.multiply(k1, 0.5 * dt, out=psi_temp)
    cp.add(psi, psi_temp, out=psi_temp)
    k2[:] = psi_full_rhs(psi_temp, t + 0.5 * dt, k_squared, params)
    
    # k3
    cp.multiply(k2, 0.5 * dt, out=psi_temp)
    cp.add(psi, psi_temp, out=psi_temp)
    k3[:] = psi_full_rhs(psi_temp, t + 0.5 * dt, k_squared, params)
    
    # k4
    cp.multiply(k3, dt, out=psi_temp)
    cp.add(psi, psi_temp, out=psi_temp)
    k4[:] = psi_full_rhs(psi_temp, t + dt, k_squared, params)
    
    # Combine (In-place update of psi_temp to hold the final delta)
    psi_temp[:] = k1
    cp.add(psi_temp, 2.0 * k2, out=psi_temp)
    cp.add(psi_temp, 2.0 * k3, out=psi_temp)
    cp.add(psi_temp, k4, out=psi_temp)
    cp.multiply(psi_temp, dt / 6.0, out=psi_temp)
                
    # --- Fully In-Place execution to prevent final array allocation ---
    cp.add(psi, psi_temp, out=psi_temp)
    return psi_temp

def compute_local_bandwidth(psi_gpu, k_mag_grid):
    """Calculates physical spectral sharpness (dk) entirely in VRAM."""
    spectrum = cp.fft.fftn(psi_gpu)
    mag = cp.abs(spectrum)
    norm = mag / (cp.sum(mag) + 1e-12)
    mean_k = cp.sum(k_mag_grid * norm)
    variance = cp.sum((k_mag_grid - mean_k)**2 * norm)
    return cp.sqrt(variance)

def psi_simulation(N_grid: int, L_domain: float, T_steps: int, DT: float, psi_params: Dict[str, Any], global_seed: int):
    # 1. INITIALIZE ON CPU (For deterministic seeding)
    key = np.random.default_rng(global_seed)
    dx = L_domain / N_grid
    k_1D = 2 * np.pi * np.fft.fftfreq(N_grid, d=dx)
    kx, ky, kz = np.meshgrid(k_1D, k_1D, k_1D, indexing='ij')
    k_sq_cpu = (kx**2 + ky**2 + kz**2).astype(np.float32) 
    
    psi0_cpu = ((key.normal(size=(N_grid, N_grid, N_grid)) + 
                 1j * key.normal(size=(N_grid, N_grid, N_grid))) / np.sqrt(2)).astype(np.complex64)

    # 2. TRANSFER TO GPU VRAM
    logging.info("[Worker] Transferring arrays to Nvidia VRAM...")
    with cp.cuda.Device(0): 
        psi_gpu = cp.asarray(psi0_cpu)
        k_squared_gpu = cp.asarray(k_sq_cpu)
        
        kx_gpu = cp.asarray(kx, dtype=cp.float32)
        ky_gpu = cp.asarray(ky, dtype=cp.float32)
        kz_gpu = cp.asarray(kz, dtype=cp.float32)
        
        # Zero-Memory K_MAG derivation
        K_MAG = cp.sqrt(k_squared_gpu)

        stride = max(1, T_steps // 250)
        t = 0.0
        
        # --- ENHANCED EPISTEMIC TELEMETRY ---
        j_info_l2_history_cpu = []
        grad_phase_var_history_cpu = []
        phase_coherence_history_cpu = []
        max_amp_history_cpu = []
        clamp_fraction_history_cpu = []
        omega_sat_history_cpu = []
        h_norm_history_cpu = []
        quantule_log = []
        
        MAX_AMPLITUDE = 20.0
        kill_switch_triggered = False
        bssn_breakdown_logged = False
        event_counter = 0
        prev_phase = None

        start_time = time.time()
        logging.info("[Worker] Commencing GPU Evolution Loop (Zero-Sync Mode)...")

        buffers = (cp.empty_like(psi_gpu), cp.empty_like(psi_gpu), cp.empty_like(psi_gpu), cp.empty_like(psi_gpu), cp.empty_like(psi_gpu))

        for step in range(T_steps):
            psi_next = rk4_step_psi_full(psi_gpu, t, DT, k_squared_gpu, psi_params, buffers)
            
            amp_next = cp.abs(psi_next)
            scale_factor = cp.minimum(1.0, MAX_AMPLITUDE / (amp_next + 1e-12))
            cp.multiply(psi_next, scale_factor, out=psi_gpu)
            
            # HPC FIX: Staggered scalar extraction (Zero PCIe Sync during hot loop)
            if step % 8 == 0:
                max_amp = float(cp.max(amp_next).item())
                clamp_frac = float(cp.mean(scale_factor < 1.0).item())

                # --- V3.1 CONTINUOUS GRAVITY TRACKING ---
                current_phase = cp.angle(psi_gpu)
                if max_amp**2 > 0.8:
                    rho_gpu = amp_next**2
                    collapse_mask = rho_gpu > 0.8
                    
                    # Heavy 3D FFT: Only compute on the *first* step of a new collapse
                    if event_counter == 0:
                        bandwidth = float(compute_local_bandwidth(psi_gpu, K_MAG).get())
                    else:
                        bandwidth = 0.0
                        
                    if prev_phase is not None:
                        phase_diff = current_phase - prev_phase
                        phase_diff = cp.where(phase_diff > cp.pi, phase_diff - 2*cp.pi, phase_diff)
                        phase_diff = cp.where(phase_diff < -cp.pi, phase_diff + 2*cp.pi, phase_diff)
                        # Divide by (8 * DT) because prev_phase is from 8 steps ago
                        omega_local = float(cp.median(cp.abs(phase_diff[collapse_mask])) / (8 * DT))
                    else:
                        omega_local = 0.0
                        
                    quantule_log.append({
                        "t_step": int(step),
                        "omega_local": omega_local,
                        "spectral_bandwidth_dk": bandwidth
                    })
                    event_counter += 1
                else:
                    # If the wave dissipates, reset the counter.
                    event_counter = 0
                
                # Always update the cache so we have a valid delta for the next 8th step
                prev_phase = current_phase
                # ---------------------------------------------------------
                
                # 🚨 HAMILTONIAN L2 NORM GATE (Ontological Kill-Switch)
                rhs_val = psi_full_rhs(psi_gpu, t, k_squared_gpu, psi_params)
                h_norm = float(cp.sqrt(cp.mean(cp.abs(rhs_val)**2)).item())
                
                # 🚨 EPISTEMIC SENSOR (SDG Conformal Gravity Paradigm)
                if not np.isfinite(max_amp) or h_norm > 1e15: 
                    logging.error(f"[KillSwitch] FATAL: Numerical Infinity or VRAM limit reached at step {step}. Halting.")
                    kill_switch_triggered = True
                    break
                elif h_norm > 1e7 and not bssn_breakdown_logged: 
                    logging.warning(f"[RESONANCE SENSOR] Step {step}: H-Norm explosion ({h_norm:.2e}). Muting further warnings.")
                    bssn_breakdown_logged = True 

            t += DT

            # --- PERIODIC HOST SYNC & SAVE ---
            if step % stride == 0 or step == T_steps - 1:
                psi_k = cp.fft.fftn(psi_gpu)
                grad_x_psi = cp.fft.ifftn(1j * kx_gpu * psi_k)
                grad_y_psi = cp.fft.ifftn(1j * ky_gpu * psi_k)
                grad_z_psi = cp.fft.ifftn(1j * kz_gpu * psi_k)
                
                psi_conj = cp.conj(psi_gpu)
                J_x = cp.imag(psi_conj * grad_x_psi)
                J_y = cp.imag(psi_conj * grad_y_psi)
                J_z = cp.imag(psi_conj * grad_z_psi)
                J_info_l2 = cp.sqrt(cp.sum(J_x**2 + J_y**2 + J_z**2))
                
                rho_safe = cp.maximum(cp.abs(psi_gpu)**2, 1e-12) 
                grad_x_phase = J_x / rho_safe
                grad_y_phase = J_y / rho_safe
                grad_z_phase = J_z / rho_safe
                grad_phase_var = cp.var(grad_x_phase) + cp.var(grad_y_phase) + cp.var(grad_z_phase)
                
                phase_coherence = cp.abs(cp.mean(cp.exp(1j * cp.angle(psi_gpu))))
                
                # Omega Saturation Tracking
                omega_sq = derive_stable_conformal_factor(rho_safe, psi_params)
                omega_sat = float(cp.mean((omega_sq <= 1e-6) | (omega_sq >= 1e6)).item())
                
                j_info_l2_history_cpu.append(float(J_info_l2.get()))
                grad_phase_var_history_cpu.append(float(grad_phase_var.get()))
                phase_coherence_history_cpu.append(float(phase_coherence.get()))
                max_amp_history_cpu.append(max_amp)
                clamp_fraction_history_cpu.append(clamp_frac)
                omega_sat_history_cpu.append(omega_sat)
                h_norm_history_cpu.append(h_norm)

        cp.cuda.Stream.null.synchronize() 
        end_time = time.time()
        total_time = end_time - start_time
        avg_step_time = total_time / (step+1)

        if kill_switch_triggered:
            pad_steps = (T_steps // stride + 1) - len(j_info_l2_history_cpu)
            for _ in range(pad_steps):
                j_info_l2_history_cpu.append(0.0)
                grad_phase_var_history_cpu.append(0.0)
                phase_coherence_history_cpu.append(0.0)
                max_amp_history_cpu.append(0.0)
                clamp_fraction_history_cpu.append(1.0) 
                omega_sat_history_cpu.append(1.0)      
                h_norm_history_cpu.append(999.0)

        history = (
            np.array(j_info_l2_history_cpu, dtype=np.float32),
            np.array(grad_phase_var_history_cpu, dtype=np.float32),
            np.array(phase_coherence_history_cpu, dtype=np.float32),
            np.array(max_amp_history_cpu, dtype=np.float32),
            np.array(clamp_fraction_history_cpu, dtype=np.float32),
            np.array(omega_sat_history_cpu, dtype=np.float32),
            np.array(h_norm_history_cpu, dtype=np.float32)
        )
        
        final_state = PsiSimStateGPU(psi=psi_gpu.get(), k_vectors=(kx, ky, kz), k_squared=k_sq_cpu)
        
        return final_state, history, avg_step_time, total_time, kill_switch_triggered, quantule_log

def save_simulation_artifact(output_path, final_state, history, avg_step, total_time, params, sim_params, global_seed, quantule_log):
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # UNPACK ALL 7 TELEMETRY ARRAYS
    j_info_hist, grad_phase_var_hist, phase_coh_hist, max_amp_hist, clamp_hist, omega_sat_hist, h_norm_hist = history
    
    with h5py.File(output_path, 'w') as f:
        # Save Telemetry Datasets
        f.create_dataset('j_info_l2_history', data=j_info_hist)
        f.create_dataset('grad_phase_var_history', data=grad_phase_var_hist)
        f.create_dataset('phase_coherence_history', data=phase_coh_hist)
        f.create_dataset('max_amp_history', data=max_amp_hist)
        f.create_dataset('clamp_fraction_history', data=clamp_hist)
        f.create_dataset('omega_sat_history', data=omega_sat_hist)
        f.create_dataset('h_norm_history', data=h_norm_hist)

        # Final state arrays ONLY (Shrinks file from ~6.2GB to ~32MB)
        f.create_dataset('final_psi', data=final_state.psi, compression="gzip")
        f.create_dataset('final_rho', data=np.abs(final_state.psi)**2, compression="gzip")
        
        # Array-Safe LOM Database Save
        if "quantule_events" in f:
            del f["quantule_events"]
        if quantule_log:
            grp = f.create_group("quantule_events")
            grp.create_dataset("t_step", data=[e["t_step"] for e in quantule_log])
            grp.create_dataset("omega_local", data=[e["omega_local"] for e in quantule_log])
            grp.create_dataset("bandwidth", data=[e["spectral_bandwidth_dk"] for e in quantule_log])
        
        f.attrs['manifest'] = json.dumps({"global_seed": global_seed, "fmia_params": params, "sim_params": sim_params})
        f.attrs['avg_step_time_ms'] = avg_step * 1000
        f.attrs['total_run_time_s'] = total_time

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')

    with open(args.params, 'r') as f:
        params = json.load(f)

    N_GRID = params.get('simulation', {}).get('N_grid', 64)
    L_DOMAIN = params.get('simulation', {}).get('L_domain', 10.0)
    T_STEPS = params.get('simulation', {}).get('T_steps', 750)
    DT = params.get('simulation', {}).get('dt', 0.001)
    GLOBAL_SEED = params.get('global_seed', 42)

    psi_params_clean = {}
    def extract_params(source_dict):
        for k, v in source_dict.items():
            if isinstance(v, dict): extract_params(v)
            elif k.startswith('param_'):
                try: psi_params_clean[k] = float(v)
                except: pass
            elif k == 'global_seed':
                try: psi_params_clean[k] = int(v)
                except: pass

    extract_params(params)
    
    final_state, history, avg_step, total_time, kill_switch, quantule_log = psi_simulation(
        N_grid=N_GRID, L_domain=L_DOMAIN, T_steps=T_STEPS, DT=DT,
        psi_params=psi_params_clean, global_seed=GLOBAL_SEED
    )

    save_simulation_artifact(args.output, final_state, history, avg_step, total_time, psi_params_clean, params.get('simulation', {}), GLOBAL_SEED, quantule_log)
    logging.info(f"[Worker] GPU Execution complete in {total_time:.2f} seconds.")