import json
from pathlib import Path
from unittest.mock import patch

from orchestrator.scheduling.job_dispatcher import JobDispatcher
from orchestrator.scheduling.queue_manager import QueueManager


def test_dispatch_generation_uses_single_batch_enqueue(tmp_path: Path):
    queue_path = tmp_path / "backlog_queue.json"
    result_path = tmp_path / "result_queue.json"

    config = {
        "seeds_per_candidate": 3,
        "origin": "NATURAL",
    }

    dispatcher = JobDispatcher(config)
    dispatcher.queue_manager = QueueManager(queue_file=str(queue_path), result_file=str(result_path))

    candidates = [
        {"param1": 1.0, "param2": 2.0},
        {"param1": 1.1, "param2": 2.1},
    ]

    with patch.object(dispatcher, "_pop_backlog_seed_configs", return_value=[]), patch.object(
        dispatcher.queue_manager,
        "push_jobs_batch",
        wraps=dispatcher.queue_manager.push_jobs_batch,
    ) as batch_mock:
        dispatched = dispatcher.dispatch_generation(generation=7, candidate_configs=candidates)

    assert dispatched == 6
    assert batch_mock.call_count == 1
    assert len(batch_mock.call_args.args[0]) == 6


def test_queue_manager_push_jobs_batch_appends_all_payloads(tmp_path: Path):
    queue_path = tmp_path / "queue.json"
    result_path = tmp_path / "results.json"
    qm = QueueManager(queue_file=str(queue_path), result_file=str(result_path))

    payloads = [
        json.dumps({"job_id": "a", "v": 1}),
        json.dumps({"job_id": "b", "v": 2}),
        json.dumps({"job_id": "c", "v": 3}),
    ]

    pushed = qm.push_jobs_batch(payloads)
    assert pushed == 3

    queue_data = json.loads(queue_path.read_text(encoding="utf-8"))
    assert len(queue_data) == 3
    assert [item["job_id"] for item in queue_data] == ["a", "b", "c"]


def test_backlog_seed_json_decode_retry_recovers_transient_failure(tmp_path: Path, monkeypatch):
    queue_path = tmp_path / "backlog_queue.json"
    queue_path.write_text(json.dumps([{"seed": 1}, {"seed": 2}]), encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    dispatcher = JobDispatcher(
        {
            "backlog_seed_json_decode_retries": 2,
            "backlog_seed_json_decode_backoff_seconds": 0.001,
        }
    )

    real_json_load = json.load
    call_count = {"n": 0}

    def _flaky_load(handle):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise json.JSONDecodeError("boom", "x", 0)
        return real_json_load(handle)

    with patch("orchestrator.scheduling.job_dispatcher.json.load", side_effect=_flaky_load), patch(
        "orchestrator.scheduling.job_dispatcher.time.sleep", return_value=None
    ):
        seeds = dispatcher._pop_backlog_seed_configs(1)

    assert len(seeds) == 1
    assert call_count["n"] >= 2
    assert dispatcher._backlog_decode_retry_events >= 1
    assert dispatcher._backlog_decode_retry_attempts >= 1


def test_backlog_seed_json_decode_retry_exhausted_returns_empty(tmp_path: Path, monkeypatch):
    queue_path = tmp_path / "backlog_queue.json"
    queue_path.write_text("{bad", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    dispatcher = JobDispatcher(
        {
            "backlog_seed_json_decode_retries": 1,
            "backlog_seed_json_decode_backoff_seconds": 0.001,
        }
    )

    with patch("orchestrator.scheduling.job_dispatcher.time.sleep", return_value=None):
        seeds = dispatcher._pop_backlog_seed_configs(1)

    assert seeds == []
