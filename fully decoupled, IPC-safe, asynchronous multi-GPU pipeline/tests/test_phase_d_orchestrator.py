import json
import os
import subprocess
from unittest.mock import MagicMock
from pathlib import Path
from unittest.mock import patch

import worker_daemon as wd
from orchestrator.orchestrator_engine import OrchestratorEngine
from orchestrator.result_processor import ResultProcessor


def _engine_config(tmp_path: Path) -> dict:
    data_dir = tmp_path / "simulation_data"
    prov_dir = tmp_path / "provenance_reports"
    archive_dir = tmp_path / "archive"
    data_dir.mkdir(parents=True, exist_ok=True)
    prov_dir.mkdir(parents=True, exist_ok=True)
    archive_dir.mkdir(parents=True, exist_ok=True)
    return {
        "db_path": str(tmp_path / "ledger.db"),
        "data_dir": str(data_dir),
        "provenance_dir": str(prov_dir),
        "archive_dir": str(archive_dir),
        "generations": 3,
        "population_size": 1,
        "seeds_per_candidate": 1,
        "orchestrator_state_file": str(tmp_path / "orchestrator_state.json"),
    }


def test_timeout_fail_path_pushes_result_queue(tmp_path: Path):
    from orchestrator.schema_utils import initialize_ledger_schema
    
    db_path = tmp_path / "simulation_ledger.db"
    wd.DB_FILE = str(db_path)
    
    # Initialize ledger schema
    initialize_ledger_schema(str(db_path))

    class _FakeQueueManager:
        def __init__(self):
            self.claimed = False
            self.completed_tokens = []
            self.pushed_results = []

        def set_worker_heartbeat(self, *_args, **_kwargs):
            return None

        def claim_job(self, *_args, **_kwargs):
            if self.claimed:
                return None
            self.claimed = True
            payload = {
                "job_id": "jid-timeout",
                "generation": 0,
                "config_hash": "abc123",
                "origin": "NATURAL",
                "params": {
                    "config_hash": "abc123",
                    "generation": 0,
                    "origin": "NATURAL",
                },
            }
            return {"token": "tok-1", "payload": json.dumps(payload)}

        def push_result(self, payload):
            self.pushed_results.append(json.loads(payload))

        def complete_job(self, token):
            self.completed_tokens.append(token)

    fake_qm = _FakeQueueManager()

    with patch("worker_daemon.QueueManager", return_value=fake_qm), patch.object(
        wd.subprocess,
        "run",
        side_effect=subprocess.TimeoutExpired(cmd=["worker_cupy.py"], timeout=0.01),
    ), patch.object(wd, "write_to_ledger", return_value=True) as ledger_mock:
        cwd = os.getcwd()
        os.chdir(tmp_path)
        try:
            wd.main()
        finally:
            os.chdir(cwd)

    assert ledger_mock.called
    assert ledger_mock.call_args.args[2] == "FAIL"
    assert fake_qm.completed_tokens == ["tok-1"]
    assert fake_qm.pushed_results, "Expected timeout payload to be pushed"
    assert fake_qm.pushed_results[0]["status"] == "FAIL"
    assert fake_qm.pushed_results[0]["reason"] == "timeout"


def test_champion_update_and_checkpoint_resume(tmp_path: Path):
    cfg = _engine_config(tmp_path)
    engine = OrchestratorEngine(cfg)

    artifact = Path(cfg["data_dir"]) / "rho_history_h-best.h5"
    artifact.write_bytes(b"123")

    def _ok(result_data):
        result_data["_ingested_log_prime_sse"] = 0.77
        return True

    engine.queue_manager.get_results = lambda: [
        {
            "job_id": "j-best",
            "generation": 0,
            "config_hash": "h-best",
            "status": "SUCCESS",
            "artifact_url": str(artifact),
        }
    ]
    engine.result_processor.process_result = _ok
    engine.bleed_manager.queue_for_bleed = lambda *args, **kwargs: None

    engine._process_pending_results()

    assert engine.global_best_hash == "h-best"
    assert abs(engine.global_best_sse - 0.77) < 1e-9
    state_path = Path(cfg["orchestrator_state_file"])
    assert state_path.exists()

    restored = OrchestratorEngine(cfg)
    assert restored.global_best_hash == "h-best"
    assert abs(restored.global_best_sse - 0.77) < 1e-9


def test_async_visual_trigger_non_blocking(tmp_path: Path):
    cfg = {
        "db_path": str(tmp_path / "ledger.db"),
        "data_dir": str(tmp_path),
        "provenance_dir": str(tmp_path),
    }
    rp = ResultProcessor(cfg)

    artifact = tmp_path / "rho.h5"
    provenance = tmp_path / "provenance_h.json"
    artifact.write_bytes(b"1")
    provenance.write_text(json.dumps({"spectral_fidelity": {"log_prime_sse": 2.5}}), encoding="utf-8")

    payload = {
        "job_id": "jid",
        "generation": 0,
        "config_hash": "h",
        "status": "SUCCESS",
        "artifact_url": str(artifact),
        "provenance_path": str(provenance),
    }

    proc = MagicMock()
    with patch.object(rp, "_store_result", return_value=None), patch(
        "orchestrator.result_processor.subprocess.Popen",
        return_value=proc,
    ) as popen_mock:
        assert rp.process_result(payload) is True
        assert popen_mock.called
        assert not proc.wait.called
        assert not proc.communicate.called


def test_pre_execution_setup_failure_records_fail_and_releases_claim(tmp_path: Path):
    from orchestrator.schema_utils import initialize_ledger_schema
    
    db_path = tmp_path / "simulation_ledger.db"
    wd.DB_FILE = str(db_path)
    
    # Initialize ledger schema
    initialize_ledger_schema(str(db_path))
    
    class _FakeQueueManager:
        def __init__(self):
            self.claimed = False
            self.completed_tokens = []

        def set_worker_heartbeat(self, *_args, **_kwargs):
            return None

        def claim_job(self, *_args, **_kwargs):
            if self.claimed:
                return None
            self.claimed = True
            return {"token": "tok-pre", "payload": "{malformed-json"}

        def complete_job(self, token):
            self.completed_tokens.append(token)

    fake_qm = _FakeQueueManager()

    with patch("worker_daemon.QueueManager", return_value=fake_qm), patch.object(
        wd,
        "record_fail_result",
        return_value=None,
    ) as fail_mock:
        cwd = os.getcwd()
        os.chdir(tmp_path)
        try:
            wd.main()
        finally:
            os.chdir(cwd)

    assert fail_mock.called
    assert fake_qm.completed_tokens == ["tok-pre"]


def test_success_path_keeps_artifact_for_orchestrator_lifecycle(tmp_path: Path):
    from orchestrator.schema_utils import initialize_ledger_schema
    
    db_path = tmp_path / "simulation_ledger.db"
    wd.DB_FILE = str(db_path)
    
    # Initialize ledger schema
    initialize_ledger_schema(str(db_path))
    
    class _FakeQueueManager:
        def __init__(self):
            self.claimed = False
            self.completed_tokens = []
            self.pushed = []

        def set_worker_heartbeat(self, *_args, **_kwargs):
            return None

        def claim_job(self, *_args, **_kwargs):
            if self.claimed:
                return None
            self.claimed = True
            payload = {
                "job_id": "jid-ok",
                "generation": 0,
                "config_hash": "okhash",
                "origin": "NATURAL",
                "params": {
                    "config_hash": "okhash",
                    "generation": 0,
                    "origin": "NATURAL",
                },
            }
            return {"token": "tok-ok", "payload": json.dumps(payload)}

        def push_result(self, payload):
            self.pushed.append(payload)

        def complete_job(self, token):
            self.completed_tokens.append(token)

    fake_qm = _FakeQueueManager()

    def _run_side_effect(cmd, cwd=None, check=None, timeout=None):
        if "worker_cupy.py" in cmd:
            out_idx = cmd.index("--output") + 1
            artifact_path = tmp_path / cmd[out_idx]
            artifact_path.parent.mkdir(parents=True, exist_ok=True)
            artifact_path.write_bytes(b"ok")
        if "validation_pipeline.py" in cmd:
            out_dir = tmp_path / "provenance_reports"
            out_dir.mkdir(parents=True, exist_ok=True)
            provenance = {
                "spectral_fidelity": {"log_prime_sse": 0.5, "n_bragg_peaks": 1, "bragg_prime_sse": 0.5, "collapse_event_count": 0},
                "aletheia_metrics": {"pcs": 0.7},
            }
            (out_dir / "provenance_okhash.json").write_text(json.dumps(provenance), encoding="utf-8")
        return None

    with patch("worker_daemon.QueueManager", return_value=fake_qm), patch.object(
        wd.subprocess,
        "run",
        side_effect=_run_side_effect,
    ):
        cwd = os.getcwd()
        os.chdir(tmp_path)
        try:
            wd.main()
        finally:
            os.chdir(cwd)

    artifact = tmp_path / "simulation_data" / "rho_history_okhash.h5"
    assert artifact.exists()
    assert fake_qm.completed_tokens == ["tok-ok"]


def test_duplicate_result_replay_is_ignored_for_processing_and_accounting(tmp_path: Path):
    cfg = _engine_config(tmp_path)
    engine = OrchestratorEngine(cfg)

    artifact = Path(cfg["data_dir"]) / "rho_history_dup.h5"
    artifact.write_bytes(b"1")

    processed = {"count": 0}

    def _ok(result_data):
        processed["count"] += 1
        result_data["_ingested_log_prime_sse"] = 0.9
        return True

    engine.queue_manager.get_results = lambda: [
        {
            "job_id": "job-1",
            "generation": 0,
            "config_hash": "dup-hash",
            "status": "SUCCESS",
            "artifact_url": str(artifact),
        },
        {
            "job_id": "job-2",
            "generation": 0,
            "config_hash": "dup-hash",
            "status": "SUCCESS",
            "artifact_url": str(artifact),
        },
    ]
    engine.result_processor.process_result = _ok
    engine.bleed_manager.queue_for_bleed = lambda *args, **kwargs: None

    engine._process_pending_results()

    assert processed["count"] == 1
    assert len(engine.completed_jobs) == 1
