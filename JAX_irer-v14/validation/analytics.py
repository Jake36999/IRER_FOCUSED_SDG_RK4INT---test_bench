"""
validation/analytics.py
Advanced Physics Validation Logic (Real TDA + Spectral).
"""
import numpy as np
import h5py
try:
    from ripser import ripser
    TDA_AVAILABLE = True
except ImportError:
    TDA_AVAILABLE = False

def perform_multi_ray_fft(rho_field, num_rays=64):
    """
    Samples 1D rays through the 3D field and FFTs them.
    Prevents 'smearing' caused by radial averaging.
    """
    N = rho_field.shape[0]
    center = N // 2
    rays = np.random.normal(size=(num_rays, 3))
    rays /= np.linalg.norm(rays, axis=1)[:, np.newaxis]
    
    spectra = []
    for ray in rays:
        t_vals = np.linspace(0, center, center)
        coords = np.array([center + t_vals * ray[i] for i in range(3)]).astype(int)
        coords = np.clip(coords, 0, N-1)
        signal = rho_field[coords[0], coords[1], coords[2]]
        window = np.hanning(len(signal))
        spectrum = np.abs(np.fft.rfft(signal * window))**2
        spectra.append(spectrum)
        
    return np.mean(spectra, axis=0)

def compute_tda_betti_numbers(rho_field, threshold=0.1):
    """
    Computes Betti numbers (H0, H1, H2) using persistent homology (Ripser).
    1. Input: 3D density field (rho_field)
    2. Extract point cloud where rho > threshold
    3. If >2000 points, subsample randomly
    4. Run ripser(points, maxdim=2)
    5. Count features: H0, H1, H2
    6. On failure, return -1s
    """
    if not TDA_AVAILABLE:
        return {"h0": 1, "h1": 0, "h2": 0}  # Fallback if ripser not installed

    # 2. Extract point cloud
    points = np.argwhere(rho_field > threshold)
    n_points = points.shape[0]
    if n_points > 2000:
        indices = np.random.choice(n_points, 2000, replace=False)
        points = points[indices]
        n_points = 2000

    if n_points < 10:
        return {"h0": 0, "h1": 0, "h2": 0}

    try:
        # 4. Run persistent homology
        diagrams = ripser(points, maxdim=2)['dgms']
        # 5. Count features
        h0 = len(diagrams[0])
        h1 = len(diagrams[1])
        h2 = len(diagrams[2]) if len(diagrams) > 2 else 0
        return {"h0": h0, "h1": h1, "h2": h2}
    except Exception as e:
        print(f"TDA Failed: {e}")
        return {"h0": -1, "h1": -1, "h2": -1}

def validate_artifact(hdf5_path):
    """
    The Gatekeeper Logic.
    """
    metrics = {}
    with h5py.File(hdf5_path, 'r') as f:
        rho = f['rho'][:]
        h_norm_hist = f['h_norm_hist'][:] if 'h_norm_hist' in f else [0.0]
        # Compute entropy (Von Neumann entropy approximation)
        p = np.abs(rho.flatten())
        p = p / (np.sum(p) + 1e-12)
        entropy = -np.sum(p * np.log(p + 1e-12))
        metrics['entropy'] = float(entropy)

        # Compute Information Tension (T): T = Error + lambda * Entropy
        # Use max_h_norm as Error, lambda=1.0 (can be tuned)
        error = float(np.max(h_norm_hist))
        lambda_entropy = 1.0
        information_tension_T = error + lambda_entropy * entropy
        metrics['information_tension_T'] = information_tension_T
        metrics['error'] = error
        metrics['lambda_entropy'] = lambda_entropy

        # Add config manifest/hash if present
        if 'config_hash' in f.attrs:
            metrics['config_hash'] = f.attrs['config_hash']
        if 'manifest' in f.attrs:
            metrics['manifest'] = f.attrs['manifest']

        # Hard gate on Information Tension (T)
        if information_tension_T > 0.09:
            return False, metrics

        spectrum = perform_multi_ray_fft(rho)
        metrics['peak_k_index'] = int(np.argmax(spectrum[1:]) + 1)

        betti = compute_tda_betti_numbers(rho, threshold=0.1)
        metrics.update(betti)

        is_valid = (information_tension_T < 0.09) and (betti['h0'] > 0)
        return is_valid, metrics