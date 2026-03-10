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

        # Mild derivative-stage high-k damping to suppress Gibbs ringing in covariant gradients
        k_mag = cp.sqrt(self.k_sq)
        k_max = cp.max(k_mag) + cp.float32(1e-12)
        spectral_filter_alpha = cp.float32(params.get('param_spectral_filter_alpha', 0.05))
        self.derivative_filter = cp.exp(-spectral_filter_alpha * (k_mag / k_max) ** 4).astype(cp.float32)
        self.ikx_filtered = (self.ikx * self.derivative_filter).astype(cp.complex64)
        self.iky_filtered = (self.iky * self.derivative_filter).astype(cp.complex64)
        self.ikz_filtered = (self.ikz * self.derivative_filter).astype(cp.complex64)
        self.minus_k_sq_filtered = (self.minus_k_sq * self.derivative_filter).astype(cp.float32)

        # Linear Operator: Incorporates the flat Laplacian for exponential stiff integration
        self.L_k = -self.D_diff * self.k_sq - self.eta + 1j * self.rho_vac
        
        # True Kassam-Trefethen ETDRK4 Coefficient integration per-mode (VRAM-safe)
        M = 32
        theta = cp.exp(1j * cp.pi * (cp.arange(1, M + 1, dtype=cp.float32) - 0.5) / M).astype(cp.complex64)
        r = cp.float32(1.0)

        w = (self.L_k * dt).astype(cp.complex64, copy=False)

        # Preallocate accumulators to avoid 4D broadcast allocation (M x N x N x N)
        Q_acc = cp.zeros_like(w, dtype=cp.complex64)
        f1_acc = cp.zeros_like(w, dtype=cp.complex64)
        f2_acc = cp.zeros_like(w, dtype=cp.complex64)
        f3_acc = cp.zeros_like(w, dtype=cp.complex64)

        for i in range(M):
            z = r * theta[i]
            w_exp = w + z
            exp_w = cp.exp(w_exp)

            Q_acc += (cp.exp(w_exp / 2.0) - 1.0) / w_exp
            f1_acc += (-4.0 - w_exp + exp_w * (4.0 - 3.0 * w_exp + w_exp**2)) / (w_exp**3)
            f2_acc += (2.0 + w_exp + exp_w * (w_exp - 2.0)) / (w_exp**3)
            f3_acc += (-4.0 - 3.0 * w_exp - w_exp**2 + exp_w * (4.0 - w_exp)) / (w_exp**3)

        self.Q = dt * cp.real(Q_acc / M)
        self.f1 = dt * cp.real(f1_acc / M)
        self.f2 = dt * cp.real(f2_acc / M)
        self.f3 = dt * cp.real(f3_acc / M)
        
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
        cp.multiply(self.ikx_filtered, psi_k, out=self.batch_k[1])
        cp.multiply(self.iky_filtered, psi_k, out=self.batch_k[2])
        cp.multiply(self.ikz_filtered, psi_k, out=self.batch_k[3])
        cp.multiply(self.minus_k_sq_filtered, psi_k, out=self.batch_k[4])

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
    
    # Pre-allocate telemetry
    dV = (L_domain / N_grid)**3
    history = []
    extended_telemetry = []
    final_step = -1
    
    start_time = time.time()
    
    for step in range(T_steps):
        final_step = step
        # 1. Core Evolution (in Spectral Space)
        psi_k = solver.step(psi_k)
        
        # 2. Spectral Phase Centering (Zero FFT cost)
        if step % 50 == 0:
            mean_psi = psi_k[0, 0, 0] / (N_grid**3)
            mean_phase = cp.angle(mean_psi)
            psi_k *= cp.exp(-1j * mean_phase)
            
        # 3. Lightweight Telemetry & Pure Math Termination Guards
        if step % 10 == 0 or step == T_steps - 1:
            psi_real = solver.ifft_single(psi_k)

            # Terminate purely on mathematical overflow (NaN/Inf)
            if not cp.isfinite(psi_real).all():
                logging.error(f"Worker Terminated: Mathematical overflow (NaN/Inf) detected at step {step}.")
                break

            # Terminate on absolute amplitude threshold (Hardware/FP32 limit)
            max_amp = float(cp.max(cp.abs(psi_real)))
            if max_amp > collapse_threshold:
                logging.error(f"Worker Terminated: Explosive amplitude ({max_amp:.2e}) exceeded threshold at step {step}.")
                break

            rho = cp.maximum(cp.abs(psi_real)**2, 1e-12)
            energy = float(cp.sum(rho, dtype=cp.float64)) * dV
            c_invariant = float(cp.sum(rho * rho, dtype=cp.float64)) * dV
            history.append({
                'step': step,
                'energy': energy,
                'C_invariant': c_invariant,
            })

        # Operational Extended Telemetry (Strictly physical parameters)
        if (step > 0 and step % 100 == 0) or step == T_steps - 1:
            extended_telemetry.append({
                'step': step,
                'sim_time': float(step * dt),
                'dt': dt,
                'grid_shape': N_grid,
                'params_hash': config_hash
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
        f.create_dataset('omega_sq_final', data=cp.asnumpy(omega_sq_final))
        
        hist_grp = f.create_group('telemetry')
        hist_grp.create_dataset('step', data=np.array([t['step'] for t in history]))
        hist_grp.create_dataset('energy', data=np.array([t['energy'] for t in history]))
        hist_grp.create_dataset('C_invariant', data=np.array([t['C_invariant'] for t in history]))

        ext_grp = f.create_group('extended_telemetry')
        if extended_telemetry:
            ext_grp.create_dataset('step_count', data=np.array([t['step'] for t in extended_telemetry], dtype=np.int64))
            ext_grp.create_dataset('sim_time', data=np.array([t['sim_time'] for t in extended_telemetry], dtype=np.float64))
            ext_grp.create_dataset('dt', data=np.array([t['dt'] for t in extended_telemetry], dtype=np.float64))
            ext_grp.create_dataset('grid_shape', data=np.array([t['grid_shape'] for t in extended_telemetry], dtype=np.int32))
            ext_grp.create_dataset('params_hash', data=np.array([str(t['params_hash'] or "") for t in extended_telemetry], dtype=h5py.string_dtype(encoding='utf-8')))
        else:
            ext_grp.create_dataset('step_count', data=np.array([final_step], dtype=np.int64))
            ext_grp.create_dataset('sim_time', data=np.array([total_time], dtype=np.float64))
            ext_grp.create_dataset('dt', data=np.array([dt], dtype=np.float64))
            ext_grp.create_dataset('grid_shape', data=np.array([N_grid], dtype=np.int32))
            ext_grp.create_dataset('params_hash', data=np.array([config_hash or ""], dtype=h5py.string_dtype(encoding='utf-8')))
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
