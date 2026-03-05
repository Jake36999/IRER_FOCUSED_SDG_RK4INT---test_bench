"""
worker/ir_physics/stability_envelope.py
Tools for mapping the nonlinear stability manifold of the GFL engine.
"""
from .backend import lazy_load_backend
backend = lazy_load_backend()
xp = backend["xp"]
from .kernels import dt_stability_bound, alpha_critical

def compute_stability_envelope(params, rho_range, alpha_range):
    """
    Maps (alpha, rho_max) -> dt_bound
    """
    dt_map = xp.zeros((len(alpha_range), len(rho_range)))
    for i, alpha in enumerate(alpha_range):
        p = params._replace(alpha=float(alpha))
        for j, rho in enumerate(rho_range):
            dt_map = dt_map.at[i, j].set(dt_stability_bound(p, rho))
    return dt_map

def compute_alpha_critical_curve(params, rho_range):
    curve = []
    for rho in rho_range:
        curve.append(alpha_critical(params, rho))
    return xp.array(curve)