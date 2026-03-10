import sqlite3
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np
import pandas as pd

import predator_sweep
import fss_scaling_analyzer
import metrics.tensor_validation as tensor_validation
import validation_pipeline as vp


def test_tensor_construct_handles_complex_phi_with_real_output():
    rho = np.ones((6, 6, 6), dtype=np.float32)
    phase = np.linspace(0.0, np.pi, rho.size, dtype=np.float64).reshape(rho.shape)
    phi = np.exp(1j * phase).astype(np.complex64)

    T = tensor_validation.construct_T_info(rho, phi)

    assert T.shape == (3, 3, 6, 6, 6)
    assert np.isrealobj(T)
    assert np.isfinite(T).all()


def test_artifact_loader_adaptive_downsample_for_large_fields(tmp_path: Path):
    h5_path = tmp_path / "large_artifact.h5"
    psi = (np.random.rand(20, 20, 20) + 1j * np.random.rand(20, 20, 20)).astype(np.complex64)
    rho = np.random.rand(20, 20, 20).astype(np.float32)

    with h5py.File(h5_path, "w") as h5f:
        h5f.create_dataset("psi_final", data=psi)
        h5f.create_dataset("rho_final", data=rho)

    with patch.object(vp, "MAX_ARTIFACT_ELEMENTS", 1000):
        psi_out, rho_out, _ = vp.ArtifactLoader.load(str(h5_path))

    assert psi_out.shape == rho_out.shape
    assert psi_out.shape[0] < psi.shape[0]


def test_predator_db_execute_retry_recovers_from_locked_once():
    class _FakeCursor:
        def __init__(self):
            self.calls = 0

        def execute(self, *_args, **_kwargs):
            self.calls += 1
            if self.calls == 1:
                raise sqlite3.OperationalError("database is locked")
            return None

    cursor = _FakeCursor()
    predator_sweep._execute_with_retry(cursor, "SELECT 1")
    assert cursor.calls >= 2


def test_fss_read_sql_retry_recovers_from_locked_once():
    conn = sqlite3.connect(":memory:")
    expected = pd.DataFrame({"x": [1]})

    call_count = {"n": 0}

    def _side_effect(*_args, **_kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise sqlite3.OperationalError("database is locked")
        return expected

    with patch("fss_scaling_analyzer.pd.read_sql_query", side_effect=_side_effect):
        out = fss_scaling_analyzer._read_sql_with_retry("SELECT 1", conn)

    conn.close()
    assert call_count["n"] >= 2
    assert out.equals(expected)
