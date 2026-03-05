"""
quantulemapper_real.py
CLASSIFICATION: Quantule Profiler (CEPP v2.0 - Sprint 2)
GOAL: Real Isotropic Radial Spectral Analysis, Prime-Log SSE, and Falsifiability.
"""

import numpy as np
import scipy.ndimage
import scipy.signal
from typing import Tuple, List, Dict, Any

# --- IMMUTABLE SCIENTIFIC CONSTANTS ---
TARGET_PRIMES = [2, 3, 5, 7, 11, 13, 17]
TARGET_LN_PRIMES = np.log(TARGET_PRIMES)

def detect_bragg_peaks(rho_field):
    """Uses 2D FFT to detect translational symmetry (square/hexagonal lattices)."""
    if rho_field.ndim == 4:
        rho_field = rho_field[-1]
    if rho_field.ndim == 3:
        mid_z = rho_field.shape[2] // 2
        slice_2d = rho_field[:, :, mid_z]
    else:
        slice_2d = rho_field
        
    fft_2d = np.abs(np.fft.fftshift(np.fft.fft2(slice_2d)))
    center_x, center_y = np.array(fft_2d.shape) // 2
    fft_2d[center_x-2:center_x+3, center_y-2:center_y+3] = 0
    peak_threshold = np.max(fft_2d) * 0.5
    binary_peaks = fft_2d > peak_threshold
    labeled_peaks, num_peaks = scipy.ndimage.label(binary_peaks)
    return num_peaks

def validate_prime_bragg_lattice(rho_field, prime_targets=None):
    if prime_targets is None:
        prime_targets = np.log([2, 3, 5, 7, 11])
    fft_2d = np.abs(np.fft.fftshift(np.fft.fft2(rho_field)))
    center = np.array(fft_2d.shape) // 2
    fft_2d[center[0]-2:center[0]+3, center[1]-2:center[1]+3] = 0
    peak_threshold = np.max(fft_2d) * 0.8
    peak_coords = np.argwhere(fft_2d > peak_threshold)
    if len(peak_coords) == 0:
        return 999.0
    peak_radii = [np.linalg.norm(coord - center) for coord in peak_coords]
    bragg_prime_sse = 0
    for r in peak_radii:
        closest_prime = min(prime_targets, key=lambda p: abs(p - r))
        bragg_prime_sse += (r - closest_prime)**2
    return bragg_prime_sse / len(peak_radii)

def extract_isotropic_peaks(rho_field: np.ndarray) -> List[float]:
    """
    Collapses the 3D Power Spectrum into a 1D Isotropic Radial Profile.
    This captures the true harmonics of the structure across all angles,
    averaging out noise and preventing statistical degeneracy.
    """
    rho_k = np.fft.fftn(rho_field - np.mean(rho_field))
    power_spectrum = np.abs(np.fft.fftshift(rho_k))**2
    
    center = np.array(power_spectrum.shape) // 2
    z, y, x = np.indices(power_spectrum.shape)
    r = np.sqrt((x - center[2])**2 + (y - center[1])**2 + (z - center[0])**2)
    r = np.round(r).astype(int)
    
    tbin = np.bincount(r.ravel(), power_spectrum.ravel())
    nr = np.bincount(r.ravel())
    radialprofile = tbin / np.maximum(nr, 1)
    
    # Ignore DC component and extreme low frequencies
    radialprofile[:2] = 0
    
    # Find distinct harmonic peaks
    height_thresh = np.max(radialprofile) * 0.05
    peaks, properties = scipy.signal.find_peaks(radialprofile, height=height_thresh, distance=2)
    
    max_radius = min(center) - 1
    freq_vals = peaks / max_radius
    
    return freq_vals.tolist()

def calculate_bipartite_sse(measured_peaks: List[float], targets: np.ndarray) -> Dict[str, float]:
    if not measured_peaks or len(measured_peaks) == 0: 
        # Extreme penalty state, default fallback
        return {
            "primary_harmonic_error": 999.0, 
            "missing_peak_penalty": len(targets) * 1.0, 
            "noise_penalty": 0.0, 
            "best_single_error": 999.0,
            "total_sse": 999.0
        }
    
    measured = np.array(measured_peaks)
    matched_errors = []
    missing_penalty = 0.0
    noise_penalty = 0.0
    
    for i, t in enumerate(targets):
        if len(measured) == 0:
            missing_penalty += 1.0 * (len(targets) - i)
            break
        diffs = np.abs(measured - t)
        best_idx = np.argmin(diffs)
        
        # Track individual errors instead of blindly summing them
        matched_errors.append(float(diffs[best_idx]**2))
        measured = np.delete(measured, best_idx)
        
    # --- PENALIZE LEFTOVER PEAKS ---
    if len(measured) > 0:
        noise_penalty += len(measured) * 0.2
        
    # Isolate the absolute best single-prime lock (The "Sniper" metric)
    best_single_error = min(matched_errors) if matched_errors else 999.0
    
    # Calculate the legacy aggregate for total_sse
    total_harmonic_error = sum(matched_errors) if matched_errors else 999.0
    total_sse = total_harmonic_error + missing_penalty + noise_penalty
    
    return {
        "primary_harmonic_error": float(best_single_error),
        "missing_peak_penalty": float(missing_penalty),
        "noise_penalty": float(noise_penalty),
        "best_single_error": float(best_single_error),
        "total_sse": float(total_sse)
    }


def prime_log_sse(rho_field: np.ndarray) -> Dict[str, Any]:
    if rho_field.ndim == 4:
        rho_field = rho_field[-1]
        
    measured_peaks = extract_isotropic_peaks(rho_field)
    
    # --- ROBUST LEAST SQUARES SCALING ---
    def fit_scale_factor(measured, targets):
        if len(measured) == 0: return 1.0
        min_len = min(len(measured), len(targets))
        m_slice = np.sort(measured)[:min_len] # Critical sorting step to prevent alignment drift
        t_slice = targets[:min_len]
        dot_m_m = np.dot(m_slice, m_slice)
        if dot_m_m == 0: return 1.0
        return np.dot(m_slice, t_slice) / dot_m_m

    # 1. Main Signal
    main_scale = fit_scale_factor(measured_peaks, TARGET_LN_PRIMES)
    scaled_peaks = [p * main_scale for p in measured_peaks]
    main_sse_metrics = calculate_bipartite_sse(scaled_peaks, TARGET_LN_PRIMES)
    
    # --- NULL TESTS ---
    # 2. Phase Scramble (Null A)
    shuffled_targets = np.random.permutation(TARGET_LN_PRIMES)
    null_a_scale = fit_scale_factor(measured_peaks, shuffled_targets)
    scaled_null_a = [p * null_a_scale for p in measured_peaks]
    null_a_sse_metrics = calculate_bipartite_sse(scaled_null_a, shuffled_targets)
    
    # 3. Target Shuffle (Null B)
    noise_field = np.random.normal(np.mean(rho_field), np.std(rho_field), size=rho_field.shape)
    noise_peaks = extract_isotropic_peaks(noise_field)
    null_b_scale = fit_scale_factor(noise_peaks, TARGET_LN_PRIMES)
    scaled_noise = [p * null_b_scale for p in noise_peaks]
    null_b_sse_metrics = calculate_bipartite_sse(scaled_noise, TARGET_LN_PRIMES)
    
    num_bragg_peaks = detect_bragg_peaks(rho_field)
    bragg_sse = validate_prime_bragg_lattice(rho_field, TARGET_LN_PRIMES)
    
    main_sse_data = main_sse_metrics
    main_sse = main_sse_data["total_sse"]
    null_a_sse = null_a_sse_metrics["total_sse"]
    null_b_sse = null_b_sse_metrics["total_sse"]

# Ensure all decoupled metrics from calculate_bipartite_sse are explicitly named
    return {
        "log_prime_sse": main_sse_metrics["total_sse"],
        "primary_harmonic_error": main_sse_metrics["primary_harmonic_error"],
        "missing_peak_penalty": main_sse_metrics["missing_peak_penalty"],
        "noise_penalty": main_sse_metrics["noise_penalty"],
        "best_single_error": main_sse_metrics.get("best_single_error", 999.0),

        # Falsifiability Nulls
        "sse_null_target_shuffle": null_a_sse_metrics["total_sse"],
        "sse_null_phase_scramble": null_b_sse_metrics["total_sse"],

        # Legacy & Crystallography
        "bragg_lattice_sse": float(bragg_sse),
        "bragg_peaks_detected": int(num_bragg_peaks),
        "n_peaks_found_main": len(scaled_peaks),
        "n_peaks_found_null_a": len(scaled_null_a),
        "n_peaks_found_null_b": len(scaled_noise),
        "measured_peaks": measured_peaks,
        "scaled_peaks": scaled_peaks,
        "failure_reason_main": "None" if main_sse_metrics["total_sse"] < 1.0 else "High SSE",
        "failure_reason_null_a": "None",
        "failure_reason_null_b": "None"
    }