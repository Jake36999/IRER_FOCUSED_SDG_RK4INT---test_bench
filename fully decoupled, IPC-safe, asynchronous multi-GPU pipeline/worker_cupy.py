#!/usr/bin/env python3

"""
worker_cupy.py
CLASSIFICATION: ASTE-Compliant ETDRK4 GPU Solver
GOAL: High-performance pseudo-spectral integrator for the Sourced Non-Local Complex Ginzburg-Landau equation.
      Implements rigorous mathematical compliance with IRER algebraic geometry rules.
      Optimized for FP32 (Single Precision) to maximize GPU TeraFLOPS.
"""

import numpy as np
import cupy as cp
import cupyx.scipy.fft as cufft
from cupyx.scipy.fftpack import get_fft_plan
import h5py
import json
import os
import sys
import argparse
import time
import logging

# --- Emergent Gravity Bridge ---
sys.path.append(os.path.join(os.path.dirname(__file__), 'gravity'))
from orchestrator.diagnostics.runtime_audit import log_lifecycle_event
try:
    from unified_omega import (
        derive_stable_conformal_factor,
        derive_stable_conformal_factor_with_gradient,
    )
except ImportError:
    # Mandatory Fix 1: Hard failure to prevent flat-space physics fallback and preserve PDE class
    raise RuntimeError("unified_omega module required")

# =============================================================================
# GPU Fused Kernels (Rule 7: Minimizing Memory Passes)
# =============================================================================

@cp.fuse()
def calculate_cov_laplacian_fused(psi, dx, dy, dz, lap_flat, omega, omega_sq, d_omega_d_rho, D_spatial):
    # Dynamically resolve gradient of density from complex fields
    psi_conj = cp.conj(psi)
    gx = 2.0 * cp.real(psi_conj * dx)
    gy = 2.0 * cp.real(psi_conj * dy)
    gz = 2.0 * cp.real(psi_conj * dz)
    
    # Resolve gradient of conformal factor
    g_om_x = d_omega_d_rho * gx
    g_om_y = d_omega_d_rho * gy
    g_om_z = d_omega_d_rho * gz
    
    grad_omega_dot_grad_psi = g_om_x*dx + g_om_y*dy + g_om_z*dz
    cov_term = (D_spatial - 2.0) * grad_omega_dot_grad_psi / omega
    return (lap_flat + cov_term) / omega_sq

@cp.fuse()
def calculate_nonlinear_rhs(psi, rho, lap_cov, lap_flat, D_diff, a, s, f):
    nonlin = a * psi * rho + s * psi * (rho**2) + f * psi * (rho**3)
    # Exact PDE split: geometry perturbation isolates the difference between flat and covariant diffusion
    return D_diff * (lap_cov - lap_flat) + nonlin

@cp.fuse()
def compute_kt_stage_base(E2, psi_k, Q, N_k):
    return E2 * psi_k + Q * N_k

@cp.fuse()
def compute_kt_stage_c(E2, a_k, Q, N_b, N_a):
    return E2 * a_k + Q * (2.0 * N_b - N_a)

@cp.fuse()
def combine_kt_etdrk4(psi_k, N_n, N_a, N_b, N_c, E, f1, f2, f3):
    return E * psi_k + f1 * N_n + 2.0 * f2 * (N_a + N_b) + f3 * N_c

# =============================================================================
# ETDRK4 Solver Architecture (Rules 3, 4, 8, 9, 13)
# =============================================================================

class ETDRK4Solver:
    def __init__(self, N_grid, L_domain, dt, params):
        self.N = N_grid
        self.L = L_domain
        self.dt = dt
        self.D_spatial = 3.0  # 3D Space Geometry Correction
        self.params = params  # Retain params to pass to gravity module

        # Extract PDE Physics parameters
        self.D_diff = params.get('param_D', 1.0)
        self.eta = params.get('param_eta', 0.1)
        self.rho_vac = params.get('param_rho_vac', 0.0)
        self.a = params.get('param_a', 0.0)
        self.s = params.get('param_s', 0.0)
        self.f = params.get('param_f', 0.0)

        # ---------------------------------------------------------
        # Precompute Spectral Space & Operators
        # ---------------------------------------------------------
        k = cp.fft.fftfreq(N_grid, d=L_domain/N_grid).astype(cp.float32) * (2 * np.float32(np.pi))
        self.kx, self.ky, self.kz = cp.meshgrid(k, k, k, indexing='ij')
        self.k_sq = self.kx**2 + self.ky**2 + self.kz**2

        # Pre-calculated derivative factors for in-place batch generation
        self.ikx = 1j * self.kx
        self.iky = 1j * self.ky
        self.ikz = 1j * self.kz
        self.minus_k_sq = -self.k_sq

        # Linear Operator: Incorporates the flat Laplacian for exponential stiff integration
        self.L_k = -self.D_diff * self.k_sq - self.eta + 1j * self.rho_vac
        
        # True Kassam-Trefethen ETDRK4 Coefficient integration per-mode
        M = 32
        r = 1.0
        theta = 2.0 * np.pi * np.arange(1, M+1) / M
        z = cp.asarray(r * np.exp(1j * theta)).astype(cp.complex64)
        z = z[:, None, None, None]  # Broadcast to 3D grid
        
        w = self.L_k * dt
        w_exp = w[None, ...] + z
        
        # Integrating factors & phi-functions using contour integration
        self.Q = dt * cp.mean((cp.exp(w_exp / 2.0) - 1.0) / w_exp, axis=0)
        
        f1_integrand = (-4.0 - w_exp + cp.exp(w_exp) * (4.0 - 3.0*w_exp + w_exp**2)) / (w_exp**3)
        f2_integrand = (2.0 + w_exp + cp.exp(w_exp) * (w_exp - 2.0)) / (w_exp**3)
        f3_integrand = (-4.0 - 3.0*w_exp - w_exp**2 + cp.exp(w_exp) * (4.0 - w_exp)) / (w_exp**3)
        
        self.f1 = dt * cp.mean(f1_integrand, axis=0)
        self.f2 = dt * cp.mean(f2_integrand, axis=0)
        self.f3 = dt * cp.mean(f3_integrand, axis=0)
        
        self.E = cp.exp(w)
        self.E2 = cp.exp(w / 2.0)

        # Aggressive 1/2 Dealiasing Mask for Rational Geometry Coupling (Ω^-2)
        self.dealias_mask = (cp.sqrt(self.k_sq) <= (0.5 * cp.max(cp.sqrt(self.k_sq)))).astype(cp.float32)

        # ---------------------------------------------------------
        # Buffer Pre-allocation (Batched Transforms)
        # ---------------------------------------------------------
        shape = (N_grid, N_grid, N_grid)
        self.N_real_buf = cp.empty(shape, dtype=cp.complex64)
        
        # We batch Psi and its 4 derivatives into one contiguous memory block
        self.batch_k = cp.empty((5, N_grid, N_grid, N_grid), dtype=cp.complex64)
        self.batch_real = cp.empty((5, N_grid, N_grid, N_grid), dtype=cp.complex64)
        
        # Geometry Pre-allocations to prevent in-stage re-allocations
        self.rho = cp.empty(shape, dtype=cp.float32)
        self.omega = cp.empty(shape, dtype=cp.float32)
        self.omega_sq = cp.empty(shape, dtype=cp.float32)

        # Reusable FFT Plans
        self.single_plan = get_fft_plan(self.N_real_buf, axes=(0,1,2))
        self.batch_plan = get_fft_plan(self.batch_k, axes=(1,2,3))

    def fft_single(self, x):
        return cufft.fftn(x, axes=(0,1,2), plan=self.single_plan)

    def ifft_single(self, x_k):
        return cufft.ifftn(x_k, axes=(0,1,2), plan=self.single_plan)

    def ifft_batch(self, stack_k):
        return cufft.ifftn(stack_k, axes=(1,2,3), plan=self.batch_plan)

    def N_op(self, psi_k):
        """
        Calculates N(psi) directly from the spectral state.
        Uses batched cuFFT transforms and single fused kernels to decimate overhead.
        """
        # 1. Build derivative spectra in-place
        cp.copyto(self.batch_k[0], psi_k)
        cp.multiply(self.ikx, psi_k, out=self.batch_k[1])
        cp.multiply(self.iky, psi_k, out=self.batch_k[2])
        cp.multiply(self.ikz, psi_k, out=self.batch_k[3])
        cp.multiply(self.minus_k_sq, psi_k, out=self.batch_k[4])

        # 2. Batched Transform (1 invocation produces all 5 real-space arrays)
        self.batch_real[:] = self.ifft_batch(self.batch_k)
        
        psi = self.batch_real[0]
        grad_x = self.batch_real[1]
        grad_y = self.batch_real[2]
        grad_z = self.batch_real[3]
        lap_flat = self.batch_real[4]

        # 3. Algebraic Geometry Sub-Pipeline with Preallocated Buffers
        cp.multiply(psi.real, psi.real, out=self.rho)
        self.rho += psi.imag**2
        cp.maximum(self.rho, 1e-12, out=self.rho)
        
        omega_sq_tmp, d_omega_d_rho_tmp = derive_stable_conformal_factor_with_gradient(
            self.rho, self.params
        )
        cp.copyto(self.omega_sq, omega_sq_tmp)
        
        # Clamping
        cp.clip(self.omega_sq, 1e-9, 1e6, out=self.omega_sq)
        cp.sqrt(self.omega_sq, out=self.omega)

        # Derivative stability via clipping mask preventing high geometry curvature spikes
        clip_mask = (self.omega_sq > 1e-9) & (self.omega_sq < 1e6)
        d_omega_d_rho = d_omega_d_rho_tmp * clip_mask

        # 4. Covariant Laplacian (Fully Fused)
        lap_cov = calculate_cov_laplacian_fused(
            psi, grad_x, grad_y, grad_z, lap_flat, 
            self.omega, self.omega_sq, d_omega_d_rho, self.D_spatial
        )

        # 5. Synthesize Non-linear Field Operator
        # Force cast to complex64 to prevent silent upcasting by Python float64 scalars
        self.N_real_buf[:] = calculate_nonlinear_rhs(
            psi, self.rho, lap_cov, lap_flat,
            self.D_diff, self.a, self.s, self.f
        ).astype(cp.complex64, copy=False)

        # 6. Single Spectral Transform & Dealiasing
        N_k = self.fft_single(self.N_real_buf)
        N_k *= self.dealias_mask
        
        return N_k

    def step(self, psi_k):
        """Kassam-Trefethen ETDRK4 Integrator operating natively in Spectral Space"""
        
        # --- Stage A (n) ---
        N_n = self.N_op(psi_k)
        
        # --- Stage B (a) ---
        a_k = compute_kt_stage_base(self.E2, psi_k, self.Q, N_n)
        N_a = self.N_op(a_k)
        
        # --- Stage C (b) ---
        b_k = compute_kt_stage_base(self.E2, psi_k, self.Q, N_a)
        N_b = self.N_op(b_k)
        
        # --- Stage D (c) ---
        c_k = compute_kt_stage_c(self.E2, a_k, self.Q, N_b, N_a)
        N_c = self.N_op(c_k)

        # --- Recombination ---
        psi_next_k = combine_kt_etdrk4(psi_k, N_n, N_a, N_b, N_c, self.E, self.f1, self.f2, self.f3)
        psi_next_k *= self.dealias_mask
        
        return psi_next_k

# =============================================================================
# Worker Orchestration & Telemetry 
# =============================================================================

def initialize_psi(N, L, seed):
    """Initializes standard Gaussian packet bound to domain."""
    cp.random.seed(seed)
    x = cp.linspace(-L/2, L/2, N, endpoint=False, dtype=cp.float32)
    X, Y, Z = cp.meshgrid(x, x, x, indexing='ij')
    R2 = X**2 + Y**2 + Z**2
    psi = cp.exp(-R2 / 2.0).astype(cp.complex64)
    noise = (cp.random.randn(*psi.shape) + 1j * cp.random.randn(*psi.shape)).astype(cp.complex64)
    return psi + 0.01 * noise

def run_simulation(
    N_grid,
    L_domain,
    T_steps,
    dt,
    seed,
    psi_params,
    output_path,
    config_hash=None,
    generation=None,
):
    solver = ETDRK4Solver(N_grid, L_domain, dt, psi_params)
    psi = initialize_psi(N_grid, L_domain, seed)
    
    # State is permanently held in Spectral Space
    psi_k = solver.fft_single(psi) * solver.dealias_mask
    
    collapse_threshold = psi_params.get('collapse_threshold', 1e10)
    
    # Pre-allocate metrics
    dV = (L_domain / N_grid)**3
    history = []
    extended_telemetry = []
    
    start_time = time.time()
    
    for step in range(T_steps):
        # 1. Core Evolution (in Spectral Space)
        psi_k = solver.step(psi_k)
        
        # 2. Spectral Phase Centering (Zero FFT cost)
        if step % 50 == 0:
            mean_psi = psi_k[0, 0, 0] / (N_grid**3)
            mean_phase = cp.angle(mean_psi)
            psi_k *= cp.exp(-1j * mean_phase)
            
        # 3. Lightweight Telemetry (1 extra IFFT every 10 steps)
        if step % 10 == 0 or step == T_steps - 1:
            psi_real = solver.ifft_single(psi_k)
            rho = cp.maximum(cp.abs(psi_real)**2, 1e-12)
            C_inv = float(cp.sum(rho**2)) * dV
            energy = float(cp.sum(rho)) * dV
            
            if cp.isnan(rho).any() or cp.isinf(rho).any():
                logging.error(f"Worker Terminated: NaN detected at step {step}.")
                break
                
            if C_inv > collapse_threshold:
                logging.error(f"Worker Terminated: Explosive collapse detected at step {step} (C_inv: {C_inv:.2e}).")
                break
                
            history.append({
                'step': step,
                'C_invariant': C_inv,
                'energy': energy
            })

        # Legacy Extended Telemetry (True Spectral Derivatives)
        if (step > 0 and step % 100 == 0) or step == T_steps - 1:
            psi_real = solver.ifft_single(psi_k)
            rho = cp.maximum(cp.abs(psi_real)**2, 1e-12)
            phase = cp.angle(psi_real)
            
            phase_k = solver.fft_single(phase)
            dp_x = cp.real(solver.ifft_single(solver.ikx * phase_k))
            dp_y = cp.real(solver.ifft_single(solver.iky * phase_k))
            dp_z = cp.real(solver.ifft_single(solver.ikz * phase_k))
            
            rho_k = solver.fft_single(rho)
            dr_x = cp.real(solver.ifft_single(solver.ikx * rho_k))
            dr_y = cp.real(solver.ifft_single(solver.iky * rho_k))
            dr_z = cp.real(solver.ifft_single(solver.ikz * rho_k))
            
            phase_coherence = float(cp.abs(cp.mean(cp.exp(1j * phase))))
            grad_phase_var = float(cp.var(dp_x**2 + dp_y**2 + dp_z**2))
            J_info_l2 = float(cp.sum((dr_x**2 + dr_y**2 + dr_z**2) / (4.0 * rho))) * dV
            
            omega_sq_tel = derive_stable_conformal_factor(rho, psi_params)
            omega_saturation = float(cp.max(omega_sq_tel))
            
            extended_telemetry.append({
                'step': step,
                'J_info_l2': J_info_l2,
                'phase_coherence': phase_coherence,
                'grad_phase_var': grad_phase_var,
                'omega_saturation': omega_saturation
            })

    total_time = time.time() - start_time
    logging.info(f"Evolution complete in {total_time:.2f}s")
    
    # 4. Final Disk I/O 
    psi_final = solver.ifft_single(psi_k)
    rho_final = cp.abs(psi_final)**2
    
    # ASTE D.5: Explicitly derive final geometry for validation ingestion
    omega_sq_final = derive_stable_conformal_factor(rho_final, psi_params)
    
    with h5py.File(output_path, 'w') as f:
        f.create_dataset('psi_final', data=cp.asnumpy(psi_final))
        f.create_dataset('rho_final', data=cp.asnumpy(rho_final))
        f.create_dataset('omega_sq_final', data=cp.asnumpy(omega_sq_final))
        # Ensure rho_history is exported as 4D (Time, X, Y, Z) for downstream TDA profilers
        f.create_dataset('rho_history', data=np.expand_dims(cp.asnumpy(rho_final), axis=0))
        
        hist_grp = f.create_group('telemetry')
        hist_grp.create_dataset('step', data=np.array([t['step'] for t in history]))
        hist_grp.create_dataset('C_invariant', data=np.array([t['C_invariant'] for t in history]))
        hist_grp.create_dataset('energy', data=np.array([t['energy'] for t in history]))
        
        if extended_telemetry:
            ext_grp = f.create_group('extended_telemetry')
            ext_grp.create_dataset('step', data=np.array([t['step'] for t in extended_telemetry]))
            ext_grp.create_dataset('J_info_l2', data=np.array([t['J_info_l2'] for t in extended_telemetry]))
            ext_grp.create_dataset('phase_coherence', data=np.array([t['phase_coherence'] for t in extended_telemetry]))
            ext_grp.create_dataset('grad_phase_var', data=np.array([t['grad_phase_var'] for t in extended_telemetry]))
            ext_grp.create_dataset('omega_saturation', data=np.array([t['omega_saturation'] for t in extended_telemetry]))
        f.flush()

    # ASTE D.6: Force NFS metadata sync and explicitly release VRAM for daemon safety
    if hasattr(os, 'sync'):
        os.sync()
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
    log_lifecycle_event(
        stage="h5_write",
        config_hash=config_hash,
        generation=generation if isinstance(generation, int) else None,
        details={"artifact_url": output_path},
    )

# =============================================================================
# Interface Entry Point
# =============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="ASTE ETDRK4 GPU Solver")
    parser.add_argument("--params", required=False, help="Path to raw parameters JSON")
    parser.add_argument("--manifest", required=False, help="Path to distributed job manifest JSON")
    parser.add_argument("--output", required=False, help="Path to output HDF5 artifact")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')

    if not args.params and not args.manifest:
        parser.error("Must provide either --params or --manifest")

    params = {}
    output_path = args.output
    manifest_config_hash = None
    manifest_generation = None

    # 1. Distributed Manifest Parsing
    if args.manifest:
        with open(args.manifest, 'r') as f:
            manifest_data = json.load(f)
        params = manifest_data.get('params', {})
        manifest_config_hash = manifest_data.get("config_hash")
        manifest_generation = manifest_data.get("generation")
        if not output_path:
            output_path = manifest_data.get('output_path', 'default_output.h5')
            
    # 2. Legacy Raw Params Parsing
    elif args.params:
        with open(args.params, 'r') as f:
            params = json.load(f)
        if not output_path:
            parser.error("Must provide --output when using --params")

    # Extract simulation geometry & time parameters from manifest/config payload.
    sim_params = params.get('simulation', {})
    N_GRID = int(sim_params.get('N_grid', 64))
    L_DOMAIN = float(sim_params.get('L_domain', 10.0))
    T_STEPS = int(sim_params.get('T_steps', 250))
    DT = float(sim_params.get('dt', 0.001))
    GLOBAL_SEED = int(params.get('global_seed', 42))

    # Flatten nested physics parameters into a flat dictionary
    psi_params_clean = {}
    def extract_params(source_dict):
        for k, v in source_dict.items():
            if isinstance(v, dict): 
                extract_params(v)
            elif k.startswith('param_') or k == 'collapse_threshold':
                try: psi_params_clean[k] = float(v)
                except: pass

    extract_params(params)
    
    logging.info(f"Initializing ETDRK4 Worker -> Grid: {N_GRID}^3, Steps: {T_STEPS}")
    
    # Launch Mathematical Core
    run_simulation(
        N_GRID,
        L_DOMAIN,
        T_STEPS,
        DT,
        GLOBAL_SEED,
        psi_params_clean,
        output_path,
        config_hash=manifest_config_hash,
        generation=manifest_generation,
    )
