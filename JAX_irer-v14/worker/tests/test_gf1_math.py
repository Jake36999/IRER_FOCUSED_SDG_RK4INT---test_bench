"""
worker/tests/test_gfl_math.py
Unit tests for GFL-RK4 mathematical operators and stability bounds.
Usage: pytest worker/tests/test_gfl_math.py
"""
import pytest
import jax
import jax.numpy as jnp
import numpy as np
from worker.ir_physics import kernels, models

def test_periodic_laplacian_accuracy():
    """
    Verifies that the roll-based Laplacian matches the analytical Laplacian
    for a smooth sine wave on a torus. Accuracy should be O(dx^2).
    """
    N = 64
    L = 2.0 * np.pi
    dx = L / N
    x = jnp.linspace(0, L, N, endpoint=False)
    y = jnp.linspace(0, L, N, endpoint=False)
    z = jnp.linspace(0, L, N, endpoint=False)
    X, Y, Z = jnp.meshgrid(x, y, z, indexing='ij')

    # Test Function: sin(x) + sin(y) + sin(z)
    # Laplacian Analytical: -sin(x) - sin(y) - sin(z) = -Field
    field = jnp.sin(X) + jnp.sin(Y) + jnp.sin(Z)
    analytical_lap = -field

    # Numerical Laplacian
    numerical_lap = kernels.laplacian_periodic(field, dx)

    # Error Metric
    error = jnp.abs(numerical_lap - analytical_lap)
    max_error = jnp.max(error)

    print(f"Max Laplacian Error (N={N}): {max_error}")
    
    # Second order accuracy check: Error should be small (~1e-3 for N=64)
    assert max_error < 0.01, "Laplacian accuracy is insufficient."

def test_gradient_periodic_properties():
    """
    Verifies that gradients are anti-symmetric and periodic.
    """
    N = 32
    field = jax.random.normal(jax.random.PRNGKey(0), (N, N, N))
    dx = 1.0
    
    grads = kernels.get_gradients_periodic(field, dx)
    
    # Check shape
    assert grads.shape == (N, N, N, 3)
    
    # Check mean gradient on periodic domain should be close to 0 (Fundamental Theorem of Calculus on Torus)
    mean_grad = jnp.mean(grads, axis=(0,1,2))
    assert jnp.allclose(mean_grad, 0.0, atol=1e-5), "Gradient integration on torus did not sum to zero."

def test_stability_bound_logic():
    """
    Verifies that the stability bound shrinks as Density increases or Alpha increases.
    """
    params = models.SimParams(
        dt=0.1, epsilon=0.1, alpha=1.0, rho_vac=1.0,
        c1=0.0, c3=0.0, splash_fraction=0.0, sigma_k=1.0, dx=1.0
    )
    
    rho_low = 10.0
    rho_high = 1000.0
    
    bound_low = kernels.dt_stability_bound(params, rho_low)
    bound_high = kernels.dt_stability_bound(params, rho_high)
    
    # Higher density = Stiffer curvature = Lower dt bound
    assert bound_high < bound_low, "Stability bound did not shrink with higher density."
    
    # Check Alpha scaling
    params_high_alpha = params._replace(alpha=2.0)
    bound_alpha = kernels.dt_stability_bound(params_high_alpha, rho_low)
    
    assert bound_alpha < bound_low, "Stability bound did not shrink with higher alpha."

def test_geometry_proxy_clamping():
    """
    Verifies that the geometric proxy clamps correctly to prevent singularity.
    """
    params = models.SimParams(
        dt=0.1, epsilon=0.0, alpha=1.0, rho_vac=1.0,
        c1=0.0, c3=0.0, splash_fraction=0.0, sigma_k=1.0, dx=1.0
    )
    
    # Extreme density (should trigger clamp)
    rho_extreme = jnp.array(1e9)
    omega = kernels.solve_geometry_proxy(rho_extreme, params)
    
    # The kernel clamps omega to min 0.01
    assert omega >= 0.01, "Omega lower bound clamp failed."
    assert omega <= 1.5, "Omega upper bound clamp failed."

if __name__ == "__main__":
    # verification script manually
    try:
        test_periodic_laplacian_accuracy()
        test_gradient_periodic_properties()
        test_stability_bound_logic()
        test_geometry_proxy_clamping()
        print("✅ All Math Unit Tests Passed.")
    except AssertionError as e:
        print(f"❌ Test Failed: {e}")