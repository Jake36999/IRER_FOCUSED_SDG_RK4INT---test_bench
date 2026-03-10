import numpy as np

def compute_nonlinear_balance(rho: np.ndarray, lambda_param: float = 1.0, mu_param: float = 1.0):
    """
    Calculates R = |lambda * rho^2| / |mu * rho^3|
    Proves whether the system is in the soliton-forming regime.
    """
    rho_safe = np.maximum(rho, 1e-12)
    numerator = np.abs(lambda_param * (rho_safe**2))
    denominator = np.abs(mu_param * (rho_safe**3))
    
    # Return the mean balance ratio across the active field
    R = numerator / denominator
    return float(np.mean(R[rho > np.mean(rho)]))

def compute_correlation_length(rho: np.ndarray):
    """
    Computes the spatial correlation length (xi) using a 3D FFT autocorrelation.
    G(r) = <rho(x)rho(x+r)>
    """
    # Mean-center the density
    delta_rho = rho - np.mean(rho)
    
    # FFT-based Autocorrelation
    rho_k = np.fft.fftn(delta_rho)
    power_spectrum = np.abs(rho_k)**2
    autocorr = np.fft.ifftn(power_spectrum).real
    autocorr = np.fft.fftshift(autocorr)
    
    # Normalize
    if autocorr.max() > 0:
        autocorr /= autocorr.max()
        
    # Find the distance r where autocorrelation drops to 1/e (~0.367)
    # A simple radial average approximation:
    center = np.array(rho.shape) // 2
    y, x, z = np.indices(rho.shape)
    r = np.sqrt((x - center[1])**2 + (y - center[0])**2 + (z - center[2])**2)
    
    mask = (autocorr < np.exp(-1)) & (r > 0)
    if np.any(mask):
        xi = np.min(r[mask])
    else:
        xi = float('inf')
        
    return float(xi)

def compute_fractal_dimension_boxcount(rho: np.ndarray, threshold: float):
    """
    Measures the structural complexity at the boundary of stable zones
    using a 3D box-counting algorithm.
    """
    Z = (rho > threshold)
    if not np.any(Z):
        return 0.0
        
    def get_counts(Z, step):
        # Count 3D boxes that contain part of the boundary
        boxes = Z[::step, ::step, ::step]
        return np.sum(boxes)

    steps = [1, 2, 4, 8]
    counts = [get_counts(Z, s) for s in steps]
    
    # Fit line to log(counts) vs log(1/steps)
    coeffs = np.polyfit(np.log(1.0 / np.array(steps)), np.log(counts), 1)
    return float(coeffs[0])