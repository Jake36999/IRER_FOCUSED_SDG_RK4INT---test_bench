import json
import os
import random
from pathlib import Path

import aste_hunter
from aste_hunter import Hunter


def _seed_completed_run(
    hunter: Hunter,
    *,
    config_hash: str,
    generation: int,
    origin: str,
    fitness: float,
    log_prime_sse: float,
    primary_harmonic_error: float,
    missing_peak_penalty: float,
    pcs: float,
    ic: float,
    grad_phase_var: float,
    dominant_peak_k: float,
    secondary_peak_k: float,
):
    with hunter._get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT OR REPLACE INTO runs (config_hash, generation, status, fitness, parent_1, parent_2, origin)
            VALUES (?, ?, 'completed', ?, NULL, NULL, ?)
            """,
            (config_hash, generation, fitness, origin),
        )
        cursor.execute(
            """
            INSERT OR REPLACE INTO parameters (
                config_hash, param_D, param_eta, param_rho_vac,
                param_a_coupling, param_splash_coupling, param_splash_fraction
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (config_hash, 1.0, 0.65, 1.0, 0.3, 0.3, -0.5),
        )
        cursor.execute(
            """
            INSERT OR REPLACE INTO metrics (
                config_hash, log_prime_sse, primary_harmonic_error,
                missing_peak_penalty, noise_penalty,
                sse_null_phase_scramble, sse_null_target_shuffle,
                pcs, pli, ic, c4_contrast, ablated_c4_contrast,
                j_info_mean, grad_phase_var, max_amp_peak,
                clamp_fraction_mean, omega_sat_mean, collapse_event_count,
                dominant_peak_k, secondary_peak_k, basin_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                config_hash,
                log_prime_sse,
                primary_harmonic_error,
                missing_peak_penalty,
                0.0,
                999.0,
                999.0,
                pcs,
                0.0,
                ic,
                0.0,
                0.0,
                0.0,
                grad_phase_var,
                0.0,
                0.0,
                0.0,
                0,
                dominant_peak_k,
                secondary_peak_k,
                -1,
            ),
        )
        conn.commit()


def test_valid_breeders_include_new_producer_origins_when_rank_zero(tmp_path: Path):
    hunter = Hunter(db_file=str(tmp_path / "ledger.db"))
    population = [
        {"origin": "NATURAL", "rank": 4},
        {"origin": "SGN_ENGINE", "rank": 3},
        {"origin": "PREDATOR_SWEEP", "rank": 0},
        {"origin": "FSS_PREDICTOR", "rank": 0},
        {"origin": "PREDATOR_SWEEP", "rank": 1},
        {"origin": "FSS_PREDICTOR", "rank": 2},
    ]

    valid = hunter._get_valid_breeders(population)

    assert set(valid) == {0, 1, 2, 3}


def test_generate_next_generation_marks_children_predictor_engine_for_predictor_parents(tmp_path: Path):
    hunter = Hunter(db_file=str(tmp_path / "ledger.db"))

    _seed_completed_run(
        hunter,
        config_hash="pred_1",
        generation=0,
        origin="PREDATOR_SWEEP",
        fitness=3.0,
        log_prime_sse=0.05,
        primary_harmonic_error=2.0,
        missing_peak_penalty=0.0,
        pcs=0.7,
        ic=0.2,
        grad_phase_var=0.1,
        dominant_peak_k=1.0,
        secondary_peak_k=1.2,
    )
    _seed_completed_run(
        hunter,
        config_hash="pred_2",
        generation=0,
        origin="FSS_PREDICTOR",
        fitness=2.5,
        log_prime_sse=0.06,
        primary_harmonic_error=2.1,
        missing_peak_penalty=0.0,
        pcs=0.68,
        ic=0.21,
        grad_phase_var=0.11,
        dominant_peak_k=1.1,
        secondary_peak_k=1.3,
    )

    random.seed(7)
    children = hunter.generate_next_generation(population_size=5)

    assert len(children) == 5
    assert any(child.get("origin") == "PREDICTOR_ENGINE" for child in children[1:])


def test_process_generation_results_does_not_emit_legacy_predator_queue_by_default(tmp_path: Path, monkeypatch):
    hunter = Hunter(db_file=str(tmp_path / "ledger.db"))

    job = {
        "config_hash": "legacy_gate_hash",
        "generation": 0,
        "origin": "NATURAL",
        "param_D": 1.0,
        "param_eta": 0.65,
        "param_rho_vac": 1.0,
        "param_a_coupling": 0.2,
        "param_splash_coupling": 0.2,
        "param_splash_fraction": -0.5,
    }
    hunter.register_new_jobs([job])

    provenance_dir = tmp_path / "provenance"
    provenance_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "spectral_fidelity": {
            "primary_harmonic_error": 0.001,
            "log_prime_sse": 0.2,
            "scaled_peaks": [0.9, 1.1],
            "prime_log_targets": [0.9, 1.3],
            "dominant_peak_k": 0.9,
            "secondary_peak_k": 1.1,
            "fast_energy_ratio": 5.0,
            "sse_null_phase_scramble": 0.5,
            "sse_null_target_shuffle": 0.6,
        },
        "aletheia_metrics": {
            "pcs": 0.6,
            "pli": 0.1,
            "ic": 0.2,
            "j_info_l2_mean": 0.0,
            "grad_phase_var_mean": 0.01,
            "max_amp_peak": 0.0,
            "clamp_fraction_mean": 0.0,
            "omega_sat_mean": 0.0,
        },
        "empirical_bridge": {
            "c4_interference_contrast": 0.0,
            "ablated_c4_contrast": 0.0,
        },
    }
    (provenance_dir / "provenance_legacy_gate_hash.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(aste_hunter, "ENABLE_LEGACY_PREDATOR_QUEUE", False)

    hunter.process_generation_results(str(provenance_dir), ["legacy_gate_hash"])

    assert not (tmp_path / "predator_queue.json").exists()
