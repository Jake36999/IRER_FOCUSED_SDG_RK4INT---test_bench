"""
worker/ir_physics/solver.py
The Scanning Engine.
"""

from .backend import lazy_load_backend
backend = lazy_load_backend()
xp = backend["xp"]
import jax
from functools import partial
from .models import SimState, SimParams
from .kernels import step_physics, get_green_function, solve_geometry_sdg_spectral

@partial(jax.jit, static_argnames=['total_steps', 'grid_size', 'use_adaptive_dt', 'fully_coupled_geometry', 'sdg_check_interval'])
def run_simulation_scan(
    initial_state: SimState, 
    params: SimParams, 
    total_steps: int,
    grid_size: int,
    use_adaptive_dt: bool = False,
    fully_coupled_geometry: bool = False,
    sdg_check_interval: int = 50,
    rho_max_threshold: float = 1e6,
    h_norm_threshold: float = 1e3,
    sdg_mismatch_threshold: float = 1e2
):
    """
    Executes the full simulation trajectory.
    Args:
        use_adaptive_dt: if True, use curvature-adaptive dt
        fully_coupled_geometry: if True, recompute geometry at each RK4 substep
        sdg_check_interval: interval for SDG spectral comparison (steps)
    """
    splash_green_fn = get_green_function(
        (grid_size, grid_size, grid_size), 
        params.dx, 
        params.sigma_k
    )

    def scan_body(carry_tuple, idx):
        carry, instability_flag, first_instability_idx = carry_tuple
        new_state = step_physics(
            carry, params, splash_green_fn,
            use_adaptive_dt=use_adaptive_dt,
            fully_coupled_geometry=fully_coupled_geometry
        )
        # SDG comparison every sdg_check_interval steps
        do_sdg = (idx % sdg_check_interval) == 0
        rho = xp.abs(new_state.field)**2
        rho_max = xp.max(rho)
        omega_proxy = new_state.omega
        omega_sdg = solve_geometry_sdg_spectral(rho, params.alpha, params.dx)
        sdg_mismatch = xp.sqrt(xp.mean((omega_proxy - omega_sdg)**2))
        # Only record mismatch at interval, else -1
        sdg_mismatch_out = jax.lax.cond(do_sdg, lambda _: sdg_mismatch, lambda _: -1.0, operand=None)
        # Instability checks
        h_norm_val = new_state.h_norm
        # If any instability, set flag and record first index
        new_instability = (
            (rho_max > rho_max_threshold) |
            (xp.abs(h_norm_val) > h_norm_threshold) |
            (do_sdg & (xp.abs(sdg_mismatch) > sdg_mismatch_threshold))
        )
        instability_flag = instability_flag | new_instability
        first_instability_idx = jax.lax.cond(
            (first_instability_idx < 0) & new_instability,
            lambda _: idx,
            lambda _: first_instability_idx,
            operand=None
        )
        # If instability, freeze state (no further evolution)
        new_state = jax.lax.cond(
            instability_flag,
            lambda _: carry,
            lambda _: new_state,
            operand=None
        )
        return (new_state, instability_flag, first_instability_idx), (h_norm_val, sdg_mismatch_out)

    # 3. Execute Scan
    (final_state, instability_flag, first_instability_idx), (h_norm_history, sdg_mismatch_history) = jax.lax.scan(
        scan_body,
        (initial_state, False, -1),
        jnp.arange(total_steps),
    )
    return final_state, h_norm_history, sdg_mismatch_history, instability_flag, first_instability_idx