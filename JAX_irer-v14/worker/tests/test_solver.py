"""
worker/test_solver.py
Test Harness for the JAX-SDG Physics Core.
Verifies shapes, stability checks, and JIT compilation.
"""
import time
import pytest
import jax
import jax.numpy as jnp
from ir_physics import models, solver

# Configuration for Test
GRID_SIZE = 16  # Small grid for fast testing
TOTAL_STEPS = 50

@pytest.fixture
def dummy_params():
    return models.SimParams(
        dt=0.01,
        epsilon=0.1,
        alpha=0.1,
        sigma_k=1.0,
        c1=0.1,
        c3=1.0,
        splash_fraction=0.2,
        dx=1.0
    )

@pytest.fixture
def initial_state():
    key = jax.random.PRNGKey(0)
    # Create random complex field
    field = jax.random.normal(key, shape=(GRID_SIZE, GRID_SIZE, GRID_SIZE), dtype=jnp.complex64)
    # Omega starts at 1.0 (Flat space)
    omega = jnp.ones((GRID_SIZE, GRID_SIZE, GRID_SIZE), dtype=jnp.float32)
    
    return models.SimState(
        time_idx=0,
        field=field,
        omega=omega,
        h_norm=jnp.array(0.0), # Scalar array
        config_hash=0
    )

def test_solver_compilation(dummy_params, initial_state):
    """Verifies that the JAX solver compiles and runs without shape errors."""
    print("\n[TEST] Compiling JAX Kernel...")
    start = time.time()
    
    # Run Solver
    final_state, history = solver.run_simulation_scan(
        initial_state,
        dummy_params,
        total_steps=TOTAL_STEPS,
        grid_size=GRID_SIZE
    )
    
    # Force execution
    _ = final_state.h_norm.block_until_ready()
    duration = time.time() - start
    print(f"[TEST] Compilation + Execution took {duration:.4f}s")
    
    # Assertions
    assert final_state.time_idx == TOTAL_STEPS
    assert final_state.field.shape == (GRID_SIZE, GRID_SIZE, GRID_SIZE)
    assert history.shape == (TOTAL_STEPS,) # History of H-Norms

def test_stability_metric(dummy_params, initial_state):
    """Verifies that H-Norm is actually being calculated (non-zero)."""
    final_state, history = solver.run_simulation_scan(
        initial_state,
        dummy_params,
        total_steps=10,
        grid_size=GRID_SIZE
    )
    
    # H-Norm should evolve from 0.0 as gravity kicks in
    max_h = jnp.max(history)
    print(f"\n[TEST] Max H-Norm over 10 steps: {max_h}")
    assert max_h > 0.0, "H-Norm remained zero! Physics is not evolving."

if __name__ == "__main__":
    # Allow running directly with `python test_solver.py`
    import sys
    sys.exit(pytest.main(["-v", __file__]))