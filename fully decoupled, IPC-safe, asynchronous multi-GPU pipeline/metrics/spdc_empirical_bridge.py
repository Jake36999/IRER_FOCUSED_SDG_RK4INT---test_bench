import numpy as np
from typing import Optional

def calculate_joint_spectral_amplitude(psi_field: np.ndarray):
    """
    Transforms the abstract complex wave psi into a simulated Joint Spectral Amplitude (JSA).
    """
    # Perform a 3D FFT on the complex field to move to the momentum basis
    jsa_simulated = np.fft.fftn(psi_field)
    
    # Shift zero-frequency component to center
    jsa_simulated = np.fft.fftshift(jsa_simulated)
    return jsa_simulated

def deconvolve_to_c4_interference(jsa_simulated: np.ndarray, pump_function: Optional[np.ndarray] = None):
    """
    Predicts 4-photon interference patterns (C_4) from the JSA.
    If experimental pump_function is not provided, assumes Gaussian.
    """
    # 1. Simulate the Gaussian Pump Intensity
    if pump_function is None:
        shape = jsa_simulated.shape
        center = np.array(shape) // 2
        y, x, z = np.indices(shape)
        # Simple isotropic Gaussian pump
        r2 = (x - center[1])**2 + (y - center[0])**2 + (z - center[2])**2
        pump_function = np.exp(-r2 / (2.0 * (shape[0]/4)**2))
        
    # 2. Deconvolution (Simulated JSI = |JSA|^2)
    jsi_simulated = np.abs(jsa_simulated)**2
    
    # 3. Regularized Division (JSI / (Pump + K)) to ensure invariant signal recovery
    K_reg = 1e-6
    recovered_signal = jsi_simulated / (pump_function + K_reg)
    
    return recovered_signal