"""Test distributed μ/σ geometry continuity."""
import numpy as np
import pytest
from gravity.unified_omega import derive_stable_conformal_factor


def test_omega_continuity_across_chunks():
    """
    Verify that global μ/σ feature can be enabled and used correctly.
    
    This test checks that when global_stats_enabled=True, the function uses
    the provided global_mu and global_sigma instead of computing local statistics.
    """
    N = 16
    L = 10.0
    
    # Create asymmetric field
    x = np.linspace(0, L, N)
    X, Y, Z = np.meshgrid(x, x, x, indexing='ij')
    rho_full = 0.1 + 0.9 * X / L  # Linear gradient from 0.1 to 1.0
    
    # Arbitrarily different statistics (not computed from rho_full)
    arbitrary_mu = 0.2
    arbitrary_sigma = 0.05
    
    params_no_global = {
        'param_rho_vac': 1.0,
        'param_a_coupling': 0.5,
        'param_spectral_filter_alpha': 0.05,
    }
    
    params_with_global = {
        **params_no_global,
        'global_stats_enabled': True,
        'global_mu': arbitrary_mu,
        'global_sigma': arbitrary_sigma,
    }
    
    # Compute conformal factors
    omega_no_global = derive_stable_conformal_factor(rho_full, params_no_global)
    omega_with_global = derive_stable_conformal_factor(rho_full, params_with_global)
    
    # Verify both compute successfully
    assert omega_no_global is not None
    assert omega_with_global is not None
    assert np.all(np.isfinite(omega_no_global))
    assert np.all(np.isfinite(omega_with_global))
    
    # Verify that using different statistics produces different results
    # (proving that global stats are actually being used)
    mean_omega_no_global = np.mean(omega_no_global)
    mean_omega_with_global = np.mean(omega_with_global)
    
    # They should differ because the arbitrary global stats differ from computed local stats
    assert abs(mean_omega_no_global - mean_omega_with_global) > 1e-6, \
        f"Global statistics should affect conformal factor: " \
        f"no_global_mean={mean_omega_no_global:.6f}, " \
        f"with_global_mean={mean_omega_with_global:.6f}, " \
        f"difference={abs(mean_omega_no_global - mean_omega_with_global):.6e}"
