"""
validation/map_stability_envelope.py
Sweeps parameter space and maps the nonlinear stability envelope for the GFL engine.
"""
import jax
import jax.numpy as jnp
import numpy as np
from worker.ir_physics import models, solver

# Parameter sweep ranges (edit as needed)
alpha_range = np.linspace(0.1, 2.0, 8)
epsilon_range = np.linspace(0.0, 0.5, 3)
f_s_range = np.linspace(0.0, 0.5, 3)
dx_range = np.array([1.0, 0.5])  # Example: two resolutions

def run_stability_sweep():
    results = []
    for dx in dx_range:
        for alpha in alpha_range:
            for epsilon in epsilon_range:
                for f_s in f_s_range:
                    # Set up parameters
                    params = models.SimParams(
                        dt=0.05 * dx**2,  # CFL-like scaling
                        epsilon=float(epsilon),
                        alpha=float(alpha),
                        rho_vac=1.0,
                        c1=0.0,
                        c3=1.0,
                        splash_fraction=float(f_s),
                        sigma_k=1.0,
                        dx=float(dx)
                    )
                    N = int(32 / dx)
                    grid_size = N
                    steps = 100  # Short run for envelope mapping
                    # Initial state: small random field
                    key = jax.random.PRNGKey(0)
                    field = 0.01 * jax.random.normal(key, (N, N, N)) + 0j
                    state = models.SimState(
                        time_idx=0,
                        field=field,
                        omega=jnp.ones_like(field),
                        grad_omega=jnp.zeros((*field.shape, 3)),
                        h_norm=jnp.array(0.0),
                        config_hash=0
                    )
                    # Run simulation with instability detection
                    final_state, h_norm_hist, sdg_hist, unstable, fail_idx = solver.run_simulation_scan(
                        state, params, steps, grid_size,
                        use_adaptive_dt=True,
                        sdg_check_interval=10,
                        rho_max_threshold=1e4,
                        h_norm_threshold=1e2,
                        sdg_mismatch_threshold=10.0
                    )
                    label = 'stable'
                    if unstable:
                        label = f'unstable@{int(fail_idx)}'
                    results.append({
                        'alpha': float(alpha),
                        'epsilon': float(epsilon),
                        'f_s': float(f_s),
                        'dx': float(dx),
                        'label': label
                    })
                    print(f"alpha={alpha:.2f}, eps={epsilon:.2f}, f_s={f_s:.2f}, dx={dx:.2f} => {label}")
    # Optionally, save or visualize results
    return results

if __name__ == "__main__":
    run_stability_sweep()
