"""
worker/ir_physics/models.py
Immutable State Definitions for the JAX-GFL Physics Kernel.
Architecture: Periodic Torus Topology / RK4 Integration.
"""
from .backend import lazy_load_backend
backend = lazy_load_backend()
xp = backend["xp"]
from typing import NamedTuple

class SimState(NamedTuple):
    """
    The Single Source of Truth for the Simulation at Time T.
    Passed through jax.lax.scan as the 'carry'.
    """
    time_idx: int
    field: object          # Complex Field A (The Matter) | Shape: (N, N, N)
    
    # --- Geometric Feedback Loop Tensors ---
    omega: object          # Conformal Factor (The Geometry)
    grad_omega: object     # Geometric Gradients (The Feedback)
    
    # --- Metrics ---
    h_norm: object         # Hamiltonian Constraint Residual (Scalar Diagnostic)
    
    # --- Governance ---
    config_hash: int            # Provenance tracking (SHA1 proxy)

class SimParams(NamedTuple):
    """
    Static Parameters.
    Closed over by the JIT compiler.
    """
    dt: float                   # Time step size
    epsilon: float              # Linear growth coefficient
    
    # --- GFL Coupling Parameters ---
    alpha: float                # Geometric Coupling Strength (Exponent factor)
    rho_vac: float              # Vacuum Density Baseline
    
    # --- S-NCGL Parameters ---
    c1: float                   # Complex Diffusion coefficient
    c3: float                   # Nonlinear Phase Shift coefficient
    splash_fraction: float      # Strength of non-local interaction
    sigma_k: float              # Width of splash kernel
    dx: float                   # Grid spacing (usually 1.0)