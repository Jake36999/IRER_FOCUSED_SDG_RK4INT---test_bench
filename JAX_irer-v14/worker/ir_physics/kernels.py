from .backend import lazy_load_backend
backend = lazy_load_backend()
xp = backend["xp"]
fft = backend["fft"]

def validate_and_clamp_params(params: SimParams):
    """
    Clamp and validate SimParams to safe physical ranges.
    Returns a new SimParams instance.
    """
    dt = xp.clip(params.dt, 1e-8, 1.0)
    epsilon = xp.clip(params.epsilon, -10.0, 10.0)
    alpha = xp.clip(params.alpha, 0.0, 10.0)
    rho_vac = xp.clip(params.rho_vac, 1e-8, 1e8)
    c1 = xp.clip(params.c1, -10.0, 10.0)
    c3 = xp.clip(params.c3, -10.0, 10.0)
    splash_fraction = xp.clip(params.splash_fraction, 0.0, 10.0)
    sigma_k = xp.clip(params.sigma_k, 1e-8, 1e8)
    dx = xp.clip(params.dx, 1e-8, 1e8)
    return SimParams(
        dt=dt,
        epsilon=epsilon,
        alpha=alpha,
        rho_vac=rho_vac,
        c1=c1,
        c3=c3,
        splash_fraction=splash_fraction,
        sigma_k=sigma_k,
        dx=dx
    )
"""
GFL-RK4 Physics Core
Extended with:
- Explicit spectral radius dt bound
- Splash contribution
- Curvature-adaptive dt
- α_critical estimator
- Optional SDG spectral comparison
"""


from .models import SimState, SimParams

# ============================================================
# PERIODIC DIFFERENTIAL OPERATORS (Toroidal Topology)
# ============================================================

def get_gradients_periodic(f, dx):
    gx = (xp.roll(f, -1, 0) - xp.roll(f, 1, 0)) / (2 * dx)
    gy = (xp.roll(f, -1, 1) - xp.roll(f, 1, 1)) / (2 * dx)
    gz = (xp.roll(f, -1, 2) - xp.roll(f, 1, 2)) / (2 * dx)
    return xp.stack([gx, gy, gz], axis=-1)

def laplacian_periodic(f, dx):
    lx = (xp.roll(f, -1, 0) - 2*f + xp.roll(f, 1, 0)) / dx**2
    ly = (xp.roll(f, -1, 1) - 2*f + xp.roll(f, 1, 1)) / dx**2
    lz = (xp.roll(f, -1, 2) - 2*f + xp.roll(f, 1, 2)) / dx**2
    return lx + ly + lz

# ============================================================
# GEOMETRY
# ============================================================

def solve_geometry_proxy(rho, params):
    rho_safe = rho + 1e-9
    ratio = xp.clip(params.rho_vac / rho_safe, 1e-4, 1e4)
    omega = ratio ** (params.alpha / 2.0)
    return xp.clip(omega, 0.01, 1.5)

# ============================================================
# STABILITY ANALYSIS TOOLS
# ============================================================

def spectral_radius_estimate(params: SimParams, rho_max: float):
    omega_min = xp.clip(
        (params.rho_vac / (rho_max + 1e-9)) ** (params.alpha / 2.0),
        0.01, 1.5
    )
    curvature_amp = 1.0 / (omega_min**2 + 1e-9)
    k_max = xp.pi / params.dx
    lap_radius = 6.0 * k_max**2
    splash_radius = params.splash_fraction
    total_radius = curvature_amp * lap_radius + splash_radius + xp.abs(params.epsilon)
    return total_radius

def dt_stability_bound(params: SimParams, rho_max: float):
    radius = spectral_radius_estimate(params, rho_max)
    return 2.8 / (radius + 1e-9)

def adaptive_dt(params: SimParams, rho):
    rho_max = xp.max(xp.abs(rho))
    return dt_stability_bound(params, rho_max)

def alpha_critical(params: SimParams, rho_max: float):
    k_max = xp.pi / params.dx
    lap_radius = 6.0 * k_max**2
    ratio = rho_max / params.rho_vac
    target = 2.8 / (params.dt * lap_radius)
    alpha = xp.log(target) / xp.log(ratio + 1e-9)
    return xp.maximum(alpha, 0.0)

# ============================================================
# COVARIANT OPERATOR
# ============================================================

def covariant_laplacian(field, omega, grad_omega, dx):
    D = 3.0
    lap = laplacian_periodic(field, dx)
    grad_A = get_gradients_periodic(field, dx)
    dot_prod = xp.sum(grad_omega * grad_A, axis=-1)
    coupling = (D - 2.0) * (dot_prod / (omega + 1e-9))
    return (lap + coupling) / (omega**2 + 1e-9)

# ============================================================
# TIME EVOLUTION
# ============================================================

def compute_time_derivative(A, omega, grad_omega, params, splash_green_fn):
    rho = xp.abs(A)**2
    diff_term = covariant_laplacian(A, omega, grad_omega, params.dx)
    linear_op = (1 + 1j * params.c1) * diff_term
    nonlinear = (1 + 1j * params.c3) * rho * A
    A_k = fft.fftn(A)
    splash = fft.ifftn(A_k * splash_green_fn)
    return (params.epsilon * A) + linear_op - nonlinear + (params.splash_fraction * splash)

def calculate_h_norm(omega, rho, params):
    lap_omega = laplacian_periodic(omega, params.dx)
    residual = lap_omega + (params.alpha * rho * omega)
    return xp.sqrt(xp.mean(xp.abs(residual)**2))

# ============================================================
# RK4 STEP
# ============================================================

@jax.jit
def step_physics(state: SimState, params: SimParams, splash_green_fn, use_adaptive_dt=False, fully_coupled_geometry=False):
    """
    RK4 step with optional adaptive dt and geometry evolution mode.
    Args:
        state: SimState
        params: SimParams
        splash_green_fn: Green's function for splash term
        use_adaptive_dt: if True, use curvature-adaptive dt
        fully_coupled_geometry: if True, recompute geometry at each RK4 substep
    """
    A0 = state.field
    rho0 = jnp.abs(A0)**2
    dt = jax.lax.cond(
        use_adaptive_dt,
        lambda _: adaptive_dt(params, rho0),
        lambda _: params.dt,
        operand=None
    )
    omega0 = solve_geometry_proxy(rho0, params)
    grad_omega0 = get_gradients_periodic(omega0, params.dx)

    if not fully_coupled_geometry:
        # Semi-coupled: geometry frozen during RK4
        k1 = compute_time_derivative(A0, omega0, grad_omega0, params, splash_green_fn)
        k2 = compute_time_derivative(A0 + 0.5 * dt * k1, omega0, grad_omega0, params, splash_green_fn)
        k3 = compute_time_derivative(A0 + 0.5 * dt * k2, omega0, grad_omega0, params, splash_green_fn)
        k4 = compute_time_derivative(A0 + dt * k3, omega0, grad_omega0, params, splash_green_fn)
    else:
        # Fully coupled: geometry/gradients recomputed at each substep
        # k1
        k1 = compute_time_derivative(A0, omega0, grad_omega0, params, splash_green_fn)
        # k2
        A1 = A0 + 0.5 * dt * k1
        rho1 = jnp.abs(A1)**2
        omega1 = solve_geometry_proxy(rho1, params)
        grad_omega1 = get_gradients_periodic(omega1, params.dx)
        k2 = compute_time_derivative(A1, omega1, grad_omega1, params, splash_green_fn)
        # k3
        A2 = A0 + 0.5 * dt * k2
        rho2 = jnp.abs(A2)**2
        omega2 = solve_geometry_proxy(rho2, params)
        grad_omega2 = get_gradients_periodic(omega2, params.dx)
        k3 = compute_time_derivative(A2, omega2, grad_omega2, params, splash_green_fn)
        # k4
        A3 = A0 + dt * k3
        rho3 = jnp.abs(A3)**2
        omega3 = solve_geometry_proxy(rho3, params)
        grad_omega3 = get_gradients_periodic(omega3, params.dx)
        k4 = compute_time_derivative(A3, omega3, grad_omega3, params, splash_green_fn)

    A_new = A0 + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    rho_new = jnp.abs(A_new)**2
    omega_new = solve_geometry_proxy(rho_new, params)
    grad_omega_new = get_gradients_periodic(omega_new, params.dx)
    h_norm_val = calculate_h_norm(omega_new, rho_new, params)

    # NaN/Inf detection (JAX-safe)
    def has_nan_or_inf(x):
        return jnp.logical_or(jnp.any(jnp.isnan(x)), jnp.any(jnp.isinf(x)))

    any_bad = (
        has_nan_or_inf(A_new) |
        has_nan_or_inf(omega_new) |
        has_nan_or_inf(grad_omega_new) |
        has_nan_or_inf(h_norm_val)
    )

    # If any NaN/Inf, zero out field and flag h_norm as -1 (or any convention)
    A_new = jax.lax.cond(any_bad, lambda _: jnp.zeros_like(A_new), lambda _: A_new, operand=None)
    omega_new = jax.lax.cond(any_bad, lambda _: jnp.ones_like(omega_new), lambda _: omega_new, operand=None)
    grad_omega_new = jax.lax.cond(any_bad, lambda _: jnp.zeros_like(grad_omega_new), lambda _: grad_omega_new, operand=None)
    h_norm_val = jax.lax.cond(any_bad, lambda _: -1.0, lambda _: h_norm_val, operand=None)

    return SimState(
        time_idx=state.time_idx + 1,
        field=A_new,
        omega=omega_new,
        grad_omega=grad_omega_new,
        h_norm=h_norm_val,
        config_hash=state.config_hash
    )

# ============================================================
# SDG SPECTRAL COMPARISON MODE
# ============================================================

def solve_geometry_sdg_spectral(rho, alpha, dx):
    N = rho.shape[0]
    kx = jnp.fft.fftfreq(N, dx) * 2 * jnp.pi
    ky = jnp.fft.fftfreq(N, dx) * 2 * jnp.pi
    kz = jnp.fft.fftfreq(N, dx) * 2 * jnp.pi
    KX, KY, KZ = jnp.meshgrid(kx, ky, kz, indexing='ij')
    k2 = KX**2 + KY**2 + KZ**2
    rho_k = jnp.fft.fftn(rho)
    omega_k = jnp.where(
        k2 > 0,
        alpha * rho_k / k2,
        1.0 + 0j
    )
    omega_sdg = jnp.real(jnp.fft.ifftn(omega_k))
    return omega_sdg

def sdg_spectral_compare(A):
    A_k = jnp.fft.fftn(A)
    energy = jnp.abs(A_k)**2
    return jnp.mean(energy, axis=(1,2))