import os
import pytest
from ir_physics import kernels, solver, models

def test_backend_switching_numpy(monkeypatch):
    monkeypatch.setenv('IRER_BACKEND', 'numpy')
    backend = kernels.lazy_load_backend()
    xp = backend['xp']
    arr = xp.array([1, 2, 3])
    assert hasattr(arr, 'shape')
    assert backend['backend'] == 'numpy'

def test_backend_switching_jax(monkeypatch):
    monkeypatch.setenv('IRER_BACKEND', 'jax')
    backend = kernels.lazy_load_backend()
    xp = backend['xp']
    arr = xp.array([1, 2, 3])
    assert hasattr(arr, 'shape')
    assert backend['backend'] == 'jax'

def test_solver_runs_both_backends(monkeypatch):
    for backend_name in ['numpy', 'jax']:
        monkeypatch.setenv('IRER_BACKEND', backend_name)
        backend = kernels.lazy_load_backend()
        xp = backend['xp']
        params = models.SimParams(
            dt=0.01, epsilon=0.1, alpha=0.1, sigma_k=1.0, c1=0.1, c3=1.0, splash_fraction=0.2, dx=1.0
        )
        field = xp.ones((8, 8, 8))
        omega = xp.ones((8, 8, 8))
        state = models.SimState(
            time_idx=0, field=field, omega=omega, grad_omega=omega, h_norm=xp.array(0.0), config_hash=0
        )
        # Just check that the solver can be called without error
        try:
            solver.run_simulation_scan(state, params, total_steps=2, grid_size=8)
        except Exception as e:
            pytest.fail(f"Solver failed for backend {backend_name}: {e}")
