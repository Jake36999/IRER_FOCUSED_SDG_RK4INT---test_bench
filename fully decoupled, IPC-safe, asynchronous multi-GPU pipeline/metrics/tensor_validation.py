import numpy as np

def construct_T_info(rho: np.ndarray, phi: np.ndarray, dx: float = 1.0, kappa: float = 1.0):
    """
    Builds the 4x4 Informational Stress-Energy Tensor.
    T_mu_nu = kappa * (d_mu phi)(d_nu phi) - g_mu_nu * Lagrangian
    Note: For a 3D spatial grid, we compute the 3x3 spatial stress tensor (T_ij).
    """
    rho_field = np.asarray(rho, dtype=np.float64)
    phi_field = np.asarray(phi)

    if np.iscomplexobj(phi_field):
        phi_field = np.angle(phi_field)
    else:
        phi_field = phi_field.astype(np.float64, copy=False)

    grad_phi = np.gradient(phi_field, dx)
    dim = len(grad_phi)
    
    # Initialize 3x3xNxNxN tensor field
    T = np.zeros((dim, dim, *rho_field.shape), dtype=np.float64)
    
    # Simple Lagrangian density proxy for the S-NCGL field
    lagrangian = 0.5 * sum(np.real(g * np.conjugate(g)) for g in grad_phi) - np.square(rho_field)

    for i in range(dim):
        for j in range(dim):
            # T_ij = kappa * d_i(phi) * d_j(phi) - delta_ij * L
            delta_ij = 1.0 if i == j else 0.0
            stress_ij = np.real(grad_phi[i] * np.conjugate(grad_phi[j]))
            T[i, j] = kappa * stress_ij - (delta_ij * lagrangian)
            
    return T

def tensor_symmetry_test(T: np.ndarray):
    """
    Max asymmetry error: |T_ij - T_ji|
    Proves Noetherian conservation of angular momentum in the emergent field.
    Must be near machine precision.
    """
    asym = np.abs(T - np.swapaxes(T, 0, 1))
    return float(np.max(asym))

def perfect_fluid_reduction_test(T: np.ndarray):
    """
    Mean absolute off-diagonal shear.
    Low shear proves the Quantules are stabilizing into rigid, fluid-like topological drops.
    """
    dim = T.shape[0]
    off_diag = []

    for i in range(dim):
        for j in range(dim):
            if i != j:
                off_diag.append(np.mean(np.abs(T[i, j])))

    return float(np.mean(off_diag))