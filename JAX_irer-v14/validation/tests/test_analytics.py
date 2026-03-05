"""
validation/tests/test_analytics.py
Pytest suite for analytics.py validation logic.
"""
import numpy as np
import pytest
from unittest import mock
import sys
sys.modules['ripser'] = mock.Mock()  # For environments without ripser
from validation import analytics

def test_perform_multi_ray_fft_shape():
    arr = np.random.rand(16, 16, 16)
    spectrum = analytics.perform_multi_ray_fft(arr, num_rays=8)
    assert spectrum.shape[0] == 9  # rfft of 16 samples

def test_compute_tda_betti_numbers_empty():
    arr = np.zeros((8, 8, 8))
    result = analytics.compute_tda_betti_numbers(arr, threshold=0.5)
    assert result == {'h0': 0, 'h1': 0, 'h2': 0}

def test_compute_tda_betti_numbers_two_blobs():
    arr = np.zeros((10, 10, 10))
    arr[2,2,2] = 1.0
    arr[7,7,7] = 1.0
    # Patch ripser to return two H0 features
    with mock.patch('validation.analytics.ripser', return_value={'dgms': [np.array([[0,1],[0,1]]), [], []]}):
        analytics.TDA_AVAILABLE = True
        result = analytics.compute_tda_betti_numbers(arr, threshold=0.5)
        assert result['h0'] == 2

def test_compute_tda_betti_numbers_ripser_error():
    arr = np.ones((10, 10, 10))
    with mock.patch('validation.analytics.ripser', side_effect=Exception("fail")):
        analytics.TDA_AVAILABLE = True
        result = analytics.compute_tda_betti_numbers(arr, threshold=0.5)
        assert result == {'h0': -1, 'h1': -1, 'h2': -1}

def test_validate_artifact_max_h_norm(monkeypatch):
    # Mock h5py.File to simulate high h_norm
    class DummyFile:
        def __enter__(self):
            class F:
                def __getitem__(self, key):
                    if key == 'rho':
                        return np.ones((8,8,8))
                    if key == 'h_norm_hist':
                        return np.array([0.1, 0.2])
                def __contains__(self, key):
                    return key in ['rho', 'h_norm_hist']
            return F()
        def __exit__(self, exc_type, exc_val, exc_tb):
            pass
    monkeypatch.setattr(analytics.h5py, 'File', lambda *a, **kw: DummyFile())
    is_valid, metrics = analytics.validate_artifact('dummy.h5')
    assert is_valid is False
    assert metrics['max_h_norm'] > 0.09
import numpy as np
import pytest
import types
import h5py
import tempfile
from unittest import mock

import validation.analytics as analytics

def test_compute_tda_betti_numbers_mocked():
    # Mock ripser if not available
    with mock.patch.object(analytics, 'TDA_AVAILABLE', False):
        arr = np.random.rand(8, 8, 8)
        betti = analytics.compute_tda_betti_numbers(arr)
        assert isinstance(betti, dict)
        assert set(betti.keys()) == {'h0', 'h1', 'h2'}

    # If ripser is available, test with a small array
    if analytics.TDA_AVAILABLE:
        arr = np.zeros((8, 8, 8))
        arr[2:6, 2:6, 2:6] = 1.0  # Structured cube
        betti = analytics.compute_tda_betti_numbers(arr, threshold=0.5)
        assert isinstance(betti, dict)
        assert set(betti.keys()) == {'h0', 'h1', 'h2'}


def test_perform_multi_ray_fft_shape():
    arr = np.random.rand(16, 16, 16)
    spectrum = analytics.perform_multi_ray_fft(arr, num_rays=10)
    # Should be 1D, length = rfft of center (8)
    assert spectrum.ndim == 1
    assert spectrum.shape[0] == 5  # rfft of 8 samples: N//2+1


def test_validate_artifact_instability():
    # Create a dummy HDF5 file with high H-Norm
    with tempfile.NamedTemporaryFile(suffix='.h5') as tmp:
        with h5py.File(tmp.name, 'w') as f:
            f.create_dataset('rho', data=np.ones((8, 8, 8)))
            f.create_dataset('h_norm_hist', data=[0.1, 0.2, 0.3])  # All > 0.09
        is_valid, metrics = analytics.validate_artifact(tmp.name)
        assert not is_valid
        assert metrics['max_h_norm'] > 0.09


def test_validate_artifact_valid():
    # Create a dummy HDF5 file with low H-Norm and structure
    with tempfile.NamedTemporaryFile(suffix='.h5') as tmp:
        with h5py.File(tmp.name, 'w') as f:
            arr = np.zeros((8, 8, 8))
            arr[2:6, 2:6, 2:6] = 1.0
            f.create_dataset('rho', data=arr)
            f.create_dataset('h_norm_hist', data=[0.01, 0.02, 0.03])
        is_valid, metrics = analytics.validate_artifact(tmp.name)
        assert is_valid
        assert metrics['max_h_norm'] < 0.09
