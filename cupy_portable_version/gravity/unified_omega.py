"""
gravity/unified_omega.py (Agnostic GPU/CPU Variant)
Single source of truth for the IRER Unified Gravity derivation.
Now dynamically supports both NumPy (CPU) and CuPy (GPU).
"""

import numpy as np
try:
    import cupy as cp  # type: ignore
except ImportError:
    cp = None
from typing import Dict

def derive_stable_conformal_factor(rho, fmia_params: dict, epsilon: float = 1e-10):
    """
    [THEORETICAL BRIDGE] Derives the stable spatial conformal factor Omega^2.
    Dynamically detects if `rho` is on the GPU or CPU and uses the correct math library.
    """
    # Dynamically select cupy or numpy based on the input array
    xp = cp.get_array_module(rho) if cp else np
    
    rho_vac = float(fmia_params.get('param_rho_vac', 1.0))
    a_coupling = float(fmia_params.get('param_a_coupling', 1.0))
    
    # --- TDA TOPOLOGICAL CAP ---
    mu = xp.mean(rho)
    sigma = xp.std(rho)
    rho_capped = xp.clip(rho, epsilon, mu + 3.0 * sigma)
    
    # Calculate true S-NCGL Omega^2 = (rho_vac / rho_capped)^a
    omega_squared = (rho_vac / rho_capped)**a_coupling
    
    # Clamp to prevent divide-by-zero in the Laplacian
    return xp.clip(omega_squared, 1e-6, 1e6)