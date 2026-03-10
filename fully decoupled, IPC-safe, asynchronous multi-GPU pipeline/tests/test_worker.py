import pytest
import jax
jax.config.update("jax_debug_nans", True)
import jax.numpy as jnp
from Alethiea.gravity import unified_omega

# Mock input for JAX hot plane
@pytest.mark.parametrize("input_array", [
    jnp.ones((10, 10)),
    jnp.zeros((5, 5)),
    jnp.linspace(0, 1, 25).reshape(5, 5)
])
def test_metric_aware_laplacian_stability(input_array):
    # This should not raise or produce NaN/Inf
    result = unified_omega.metric_aware_laplacian(input_array)
    assert jnp.all(jnp.isfinite(result)), "Non-finite values in Laplacian output"
