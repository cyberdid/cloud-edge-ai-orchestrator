"""Colab scale study: model switch and Docker-free subprocess executor."""
import pytest

from experiments.openevolve import run_pilot
from experiments.openevolve.sandbox_exec import run_candidate

DRIVER = """
import json, candidate
json.dump(candidate.main(), open('/output/result.json', 'w'))
"""


@pytest.fixture
def subprocess_executor(monkeypatch):
    from experiments.openevolve import sandbox_exec

    def no_docker(*args, **kwargs):
        raise AssertionError("Docker must not be used by the subprocess executor")

    monkeypatch.setattr(sandbox_exec, "DockerSandbox", no_docker)
    monkeypatch.setenv("OE_PILOT_EXECUTOR", "subprocess")


def test_gemma_model_default_and_env(monkeypatch):
    monkeypatch.delenv("OE_PILOT_GEMMA_MODEL", raising=False)
    assert run_pilot.gemma_model() == "gemma4:12b"
    monkeypatch.setenv("OE_PILOT_GEMMA_MODEL", "gemma4:26b")
    assert run_pilot.gemma_model() == "gemma4:26b"


def test_subprocess_executor_returns_output(subprocess_executor):
    result = run_candidate("import json\ndef main():\n    return json.load(open('payload.json'))['x'] + 1\n",
                           DRIVER, {"x": 41}, timeout_s=30)
    assert result.ok, result.error
    assert result.output == 42


def test_subprocess_timeout(subprocess_executor):
    result = run_candidate("def main():\n    while True:\n        pass\n", DRIVER, {}, timeout_s=2)
    assert not result.ok and result.error == "timeout" and result.elapsed_s < 5


def test_subprocess_output_too_large(subprocess_executor):
    result = run_candidate("def main():\n    return 'x' * (10 * 1024 * 1024)\n", DRIVER, {}, timeout_s=60)
    assert not result.ok and result.error == "output_too_large"


def test_subprocess_crash_tail(subprocess_executor):
    result = run_candidate("def main():\n    raise ValueError('boom')\n", DRIVER, {}, timeout_s=30)
    assert not result.ok and "boom" in result.error


def test_working_evaluators_run_under_subprocess(subprocess_executor, project_root):
    from experiments.openevolve.tasks.circle_packing import working_evaluator

    seed = project_root / "experiments/openevolve/upstream/circle_packing/initial_program.py"
    assert working_evaluator.evaluate(str(seed))["combined_score"] > 0
