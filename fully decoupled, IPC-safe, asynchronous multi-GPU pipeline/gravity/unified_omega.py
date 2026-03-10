"""
gravity/unified_omega.py

Single source of truth for the IRER conformal geometry mapping.

This module maps informational density ρ = |ψ|² to the spatial
conformal metric Ω² used by the covariant Laplacian.

The mapping is:

    Ω² = (ρ_vac / ρ_capped)^a

Stability protections ensure safe operation in FP32 spectral solvers.
"""

import numpy as np
from typing import Any, Dict, cast

try:
    import cupy as cp
except ImportError:
    cp = None


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _resolve_mu_sigma(rho_safe, xp, params: Dict[str, float]):
    distributed_active = _as_bool(params.get("distributed_enabled", False)) or _as_bool(
        params.get("domain_decomposed", False)
    )
    use_global_stats = _as_bool(params.get("global_stats_enabled", False)) or _as_bool(
        params.get("use_global_stats", False)
    )

    if not use_global_stats:
        if distributed_active:
            raise RuntimeError(
                "Distributed/decomposed execution requires global mu/sigma stats. "
                "Enable global_stats_enabled and provide synchronized global_mu/global_sigma."
            )
        return xp.mean(rho_safe), xp.std(rho_safe)

    if "global_mu" in params and "global_sigma" in params:
        global_mu = float(params["global_mu"])
        global_sigma = max(float(params["global_sigma"]), 1e-16)
        return xp.asarray(global_mu, dtype=rho_safe.dtype), xp.asarray(global_sigma, dtype=rho_safe.dtype)

    if distributed_active:
        raise RuntimeError(
            "Global stats feature is enabled for distributed mode, but global_mu/global_sigma were not provided. "
            "Inject all-reduced global statistics from the distributed launcher."
        )

    return xp.mean(rho_safe), xp.std(rho_safe)


def derive_stable_conformal_factor(
    rho,
    params: Dict[str, float],
    epsilon: float = 1e-12,
    debug: bool = False
) -> Any:
    """
    Compute conformal metric factor Ω² from density field ρ.

    Stability protections:

    • Density floor: prevents division by zero
    • Topological cap: limits extreme density spikes
    • Conformal horizon: ensures FP32 spectral stability

    Guarantees:

        1e-9 ≤ Ω² ≤ 1e6
    """

    # ASTE HARDENING: Clamp density to prevent FP32 underflow spikes
    if cp is not None and isinstance(rho, cp.ndarray):
        rho = cp.maximum(rho, 1e-20)
    else:
        rho = np.maximum(rho, 1e-20)

    # Device detection
    xp = cp.get_array_module(rho) if (cp is not None) else np

    # Parameter extraction
    rho_vac = float(params.get("param_rho_vac", 1.0))
    a = float(params.get("param_a_coupling", 1.0))

    # Density floor
    rho_safe = xp.maximum(rho, epsilon)

    # Topological cap (μ + 3σ)
    mu, sigma = _resolve_mu_sigma(rho_safe, xp, params)

    rho_cap = mu + 3.0 * sigma
    rho_capped = xp.minimum(rho_safe, rho_cap)

    # Conformal law
    omega_sq = (rho_vac / rho_capped) ** a

    # Conformal horizon
    omega_sq = xp.clip(omega_sq, 1e-9, 1e6)

    if debug:
        if not xp.isfinite(omega_sq).all():
            raise RuntimeError("Non-finite Ω² detected")

    return cast(Any, omega_sq)


def derive_stable_conformal_factor_with_gradient(
    rho,
    params: Dict[str, float],
    epsilon: float = 1e-12,
    debug: bool = False
) -> tuple[Any, Any]:
    """
    Compute conformal metric factor Ω² and its gradient ∂Ω²/∂ρ from density field ρ.

    Returns:
        omega_sq: The conformal factor Ω²
        d_omega_sq_d_rho: Gradient ∂Ω²/∂ρ
    """

    # ASTE HARDENING: Clamp density to prevent FP32 underflow spikes
    if cp is not None and isinstance(rho, cp.ndarray):
        rho = cp.maximum(rho, 1e-20)
    else:
        rho = np.maximum(rho, 1e-20)

    # Device detection
    xp = cp.get_array_module(rho) if (cp is not None) else np

    # Parameter extraction
    rho_vac = float(params.get("param_rho_vac", 1.0))
    a = float(params.get("param_a_coupling", 1.0))

    # Density floor
    rho_safe = xp.maximum(rho, epsilon)

    # Topological cap (μ + 3σ)
    mu, sigma = _resolve_mu_sigma(rho_safe, xp, params)

    rho_cap = mu + 3.0 * sigma
    rho_capped = xp.minimum(rho_safe, rho_cap)

    # Conformal law
    omega_sq = (rho_vac / rho_capped) ** a

    # Conformal horizon
    omega_sq = xp.clip(omega_sq, 1e-9, 1e6)

    # Compute gradient ∂Ω²/∂ρ
    # d/dρ [(ρ_vac / ρ_capped)^a] = a * (ρ_vac / ρ_capped)^(a-1) * (-ρ_vac / ρ_capped^2) * dρ_capped/dρ
    # For ρ_capped = min(ρ_safe, rho_cap), the derivative is 1 where ρ_safe < rho_cap, 0 otherwise
    mask = rho_safe < rho_cap
    d_rho_capped_d_rho = xp.where(mask, 1.0, 0.0)

    ratio = rho_vac / rho_capped
    d_omega_sq_d_rho = a * (ratio ** (a - 1)) * (-ratio / rho_capped) * d_rho_capped_d_rho

    # Clip gradient to prevent numerical issues
    d_omega_sq_d_rho = xp.clip(d_omega_sq_d_rho, -1e6, 1e6)

    if debug:
        if not xp.isfinite(omega_sq).all() or not xp.isfinite(d_omega_sq_d_rho).all():
            raise RuntimeError("Non-finite Ω² or gradient detected")

    return cast(Any, omega_sq), cast(Any, d_omega_sq_d_rho)