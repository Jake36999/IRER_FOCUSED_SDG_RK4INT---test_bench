import json
import os
import sqlite3
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np

import worker_daemon as wd
from validation_pipeline import ValidationPipeline


def test_gate1_tda_null_safety(tmp_path: Path):
    params_path = tmp_path / "params.json"
    output_dir = tmp_path / "out"
    output_dir.mkdir(parents=True, exist_ok=True)

    params_payload = {
        "config_hash": "gate1hash",
        "param_D": 1.0,
        "param_eta": 0.65,
        "param_rho_vac": 1.0,
        "param_a_coupling": 0.2,
        "param_splash_coupling": 0.2,
        "param_splash_fraction": -0.5,
    }
    params_path.write_text(json.dumps(params_payload), encoding="utf-8")

    with patch("validation_pipeline.ArtifactLoader.load", return_value=(np.zeros((8, 8, 8)), np.ones((8, 8, 8)), {})), patch(
        "validation_pipeline.SpectralFidelityEngine.run",
        return_value={
            "validation_status": "FAIL: HIGH_SSE",
            "log_prime_sse": 20.0,
            "primary_harmonic_error": 999.0,
            "missing_peak_penalty": 7.0,
            "bragg_lattice_sse": 999.0,
            "bragg_peaks_detected": 0,
        },
    ):
        pipeline = ValidationPipeline(input_path="dummy.h5", params_path=str(params_path), output_dir=str(output_dir))
        assert pipeline.run() is True

    provenance_path = output_dir / "provenance_gate1hash.json"
    assert provenance_path.exists()
    payload = json.loads(provenance_path.read_text(encoding="utf-8"))

    topology = payload.get("topology", {})
    assert topology.get("q_type") == "Transient"
    assert topology.get("betti_0") == 1
    assert topology.get("betti_1") == 0
    assert topology.get("betti_2") == 0


def test_gate2_predator_queue_atomicity(tmp_path: Path):
    db_path = tmp_path / "simulation_ledger.db"
    queue_path = tmp_path / "backlog_queue.json"

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS parameters (
            config_hash TEXT PRIMARY KEY,
            param_D REAL,
            param_eta REAL,
            param_rho_vac REAL,
            param_a_coupling REAL,
            param_splash_coupling REAL,
            param_splash_fraction REAL
        )
        """
    )
    cursor.execute(
        """
        INSERT OR REPLACE INTO parameters (
            config_hash, param_D, param_eta, param_rho_vac,
            param_a_coupling, param_splash_coupling, param_splash_fraction
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        ("golden_hash", 1.0, 0.65, 1.0, 0.2, 0.2, -0.5),
    )
    conn.commit()
    conn.close()

    queue_path.write_text("[]", encoding="utf-8")

    cmd = [
        sys.executable,
        str((Path(__file__).resolve().parent.parent / "predator_sweep.py")),
        "--target_hash",
        "golden_hash",
        "--db",
        str(db_path),
    ]

    start = time.monotonic()
    proc = subprocess.run(cmd, cwd=str(tmp_path), capture_output=True, text=True)
    elapsed = time.monotonic() - start

    assert proc.returncode == 0, proc.stderr
    assert elapsed < 2.0

    queue_payload = json.loads(queue_path.read_text(encoding="utf-8"))
    assert len(queue_payload) == 15


def test_gate3_origin_schema_parity(tmp_path: Path):
    """Validates origin column schema parity after orchestrator initializes schema."""
    from orchestrator.schema_utils import initialize_ledger_schema
    
    db_path = tmp_path / 'simulation_ledger.db'
    
    # Initialize ledger schema
    initialize_ledger_schema(str(db_path))
    
    # Now worker can write to ledger
    wd.DB_FILE = str(db_path)
    
    params_pred = {
        "generation": -1,
        "origin": "PREDATOR_SWEEP",
        "param_D": 1.0,
        "param_eta": 0.65,
        "param_rho_vac": 1.0,
        "param_a_coupling": 0.2,
        "param_splash_coupling": 0.2,
        "param_splash_fraction": -0.5,
    }
    params_fss = {
        "generation": 2,
        "origin": "FSS_PREDICTOR",
        "param_D": 1.0,
        "param_eta": 0.65,
        "param_rho_vac": 1.0,
        "param_a_coupling": 0.25,
        "param_splash_coupling": 0.18,
        "param_splash_fraction": -0.5,
    }

    assert wd.write_to_ledger("pred_hash", params_pred, "FAIL") is True
    assert wd.write_to_ledger("fss_hash", params_fss, "FAIL") is True

    conn = sqlite3.connect(wd.DB_FILE)
    rows = conn.execute(
        "SELECT origin, COUNT(*) FROM runs WHERE origin IN ('PREDATOR_SWEEP', 'FSS_PREDICTOR') GROUP BY origin"
    ).fetchall()
    conn.close()

    grouped = {row[0]: row[1] for row in rows}
    assert grouped.get("PREDATOR_SWEEP", 0) >= 1
    assert grouped.get("FSS_PREDICTOR", 0) >= 1


def test_gate4_worker_ledger_bragg_column_compatibility(tmp_path: Path):
    """Validates bragg column schema after orchestrator initializes schema."""
    from orchestrator.schema_utils import initialize_ledger_schema
    
    db_path = tmp_path / 'simulation_ledger.db'
    
    # Initialize ledger schema
    initialize_ledger_schema(str(db_path))
    
    # Now worker can write to ledger
    wd.DB_FILE = str(db_path)
    
    params = {
        "generation": 1,
        "origin": "NATURAL",
        "param_D": 1.0,
        "param_eta": 0.65,
        "param_rho_vac": 1.0,
        "param_a_coupling": 0.2,
        "param_splash_coupling": 0.2,
        "param_splash_fraction": -0.5,
    }
    provenance = {
        "spectral_fidelity": {
            "log_prime_sse": 0.42,
            "bragg_peaks_detected": 3,
            "bragg_prime_sse": 0.11,
            "collapse_event_count": 0,
        },
        "aletheia_metrics": {"pcs": 0.97},
    }

    assert wd.write_to_ledger("compat_hash", params, "SUCCESS", provenance) is True

    conn = sqlite3.connect(wd.DB_FILE)
    row = conn.execute(
        "SELECT log_prime_sse, bragg_peaks_detected, n_bragg_peaks FROM metrics WHERE config_hash = ?",
        ("compat_hash",),
    ).fetchone()
    conn.close()

    assert row is not None
    assert abs(float(row[0]) - 0.42) < 1e-9
    assert int(row[1]) == 3
    assert int(row[2]) == 3
