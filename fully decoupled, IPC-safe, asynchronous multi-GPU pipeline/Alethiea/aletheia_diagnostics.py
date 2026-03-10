import numpy as np

def extract_phase_and_density(psi: np.ndarray):
    """Extracts the real-valued Resonance Density and Phase Angle from the complex field."""
    rho = np.abs(psi)**2
    phi = np.angle(psi)
    return rho, phi

def compute_informational_current(rho: np.ndarray, phi: np.ndarray, dx: float = 1.0, kappa: float = 1.0):
    """
    MANDATE: J_info = kappa * rho * nabla(phi)
    Calculates the 3D vector field representing the flow of informational potential.
    """
    # Calculate the spatial gradient of the phase
    grad_phi = np.gradient(phi, dx)
    
    # Calculate J for each dimension (x, y, z)
    J_x = kappa * rho * grad_phi[0]
    J_y = kappa * rho * grad_phi[1]
    J_z = kappa * rho * grad_phi[2]
    
    # Combine into a single vector field array of shape (3, N, N, N)
    J_info = np.stack([J_x, J_y, J_z], axis=0)
    
    # Calculate the total global current magnitude
    total_J_norm = float(np.sum(np.linalg.norm(J_info, axis=0)))
    return J_info, total_J_norm

def compute_field_entropy(rho: np.ndarray):
    """
    MANDATE: Shannon Entropy of the Density Field
    Measures the purity/fragmentation of the informational state.
    """
    # Normalize rho into a probability distribution
    rho_sum = np.sum(rho)
    if rho_sum == 0:
        return 0.0
    
    p = rho / rho_sum
    p_flat = p.flatten()
    
    # Filter out absolute zeros to avoid log(0)
    p_safe = p_flat[p_flat > 1e-12]
    
    shannon_entropy = -np.sum(p_safe * np.log(p_safe))
    return float(shannon_entropy)

def compute_stress_energy_trace(rho: np.ndarray, phi: np.ndarray, dx: float = 1.0):
    """
    Calculates a simplified trace of the Informational Stress-Energy Tensor.
    T_mu_nu depends on nabla(rho) and nabla(phi).
    """
    grad_phi = np.gradient(phi, dx)
    kinetic_term = sum(g**2 for g in grad_phi)
    
    # Trace estimation (Energy Density scalar)
    energy_density = rho * kinetic_term + (rho**2)
    total_energy = float(np.sum(energy_density))
    
    return total_energy