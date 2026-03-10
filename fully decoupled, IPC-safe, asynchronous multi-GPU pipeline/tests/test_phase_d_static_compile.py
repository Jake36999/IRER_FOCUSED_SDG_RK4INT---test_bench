import ast
import py_compile
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


TARGET_FILES = [
    ROOT / "worker_daemon.py",
    ROOT / "orchestrator" / "orchestrator_engine.py",
    ROOT / "orchestrator" / "result_processor.py",
    ROOT / "orchestrator" / "scheduling" / "job_dispatcher.py",
]


def test_phase_d_targets_compile_cleanly():
    for file_path in TARGET_FILES:
        py_compile.compile(str(file_path), doraise=True)


def test_result_processor_has_no_blocking_wait_or_communicate_calls():
    file_path = ROOT / "orchestrator" / "result_processor.py"
    module = ast.parse(file_path.read_text(encoding="utf-8"))

    forbidden = {"wait", "communicate"}
    forbidden_calls = []

    for node in ast.walk(module):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr in forbidden:
                forbidden_calls.append(node.func.attr)

    assert not forbidden_calls, f"Found blocking subprocess calls in result_processor: {forbidden_calls}"
