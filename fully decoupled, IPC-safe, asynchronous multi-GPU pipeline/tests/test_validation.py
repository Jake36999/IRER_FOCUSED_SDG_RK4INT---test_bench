import pytest
from metrics import tensor_validation, tda_profiler
import numpy as np

def test_tensor_symmetry_error():
    tensor = np.eye(3)
    error = tensor_validation.tensor_symmetry_error(tensor)
    assert error >= 0 and error < 1e-6

def test_betti_number_bounds():
    data = np.random.rand(100, 2)
    betti = tda_profiler.compute_betti_numbers(data)
    assert all(b >= 0 for b in betti)
