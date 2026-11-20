import os
from types import SimpleNamespace

import pytest

from experiments.openevolve.tasks.knn import working_evaluator_v1_matched as evaluator


def test_v1_matched_uses_shared_timing_harness_and_ignores_reported_distances(tmp_path, monkeypatch):
    program = tmp_path / "candidate.py"
    program.write_text("def approximate_knn(*args): pass\n")
    monkeypatch.setattr(evaluator, "DEV_SETS", [{"seed": 1}])
    monkeypatch.setattr(evaluator, "make_set", lambda **_kwargs: ([[0.0]], [[0.0]], 1, []))
    monkeypatch.setattr(evaluator, "append_candidate", lambda *_args, **_kwargs: None)
    calls = []

    def run_candidate(_code, _driver, payload, **_kwargs):
        calls.append(payload["mode"])
        assert payload["mode"] == "working_v1_matched_timed"
        return SimpleNamespace(ok=True, output={"sets": [{
            "reference_time": 1.0,
            "reference_ids": [[0]],
            "runs": [{"time": 0.25, "ids": [[0]], "dists": [[999.0]]}] * 3,
        }]})

    monkeypatch.setattr(evaluator, "run_candidate", run_candidate)

    metrics = evaluator.evaluate(str(program))

    assert calls == ["working_v1_matched_timed"]
    assert metrics["recall"] == 1.0
    assert metrics["speed_ratio"] == 0.25
    assert metrics["combined_score"] == evaluator.speed_score(1.0, 0.25)
    assert "valid" not in metrics


@pytest.mark.skipif(os.environ.get("RUN_DOCKER_TESTS") != "1", reason="set RUN_DOCKER_TESTS=1")
def test_docker_smoke_rejects_fabricated_distances_before_timing(tmp_path, monkeypatch):
    from experiments.openevolve.tasks.knn import working_evaluator_v2 as v2
    from experiments.openevolve.sandbox_exec import run_candidate as real_run_candidate

    small_set = [dict(seed=993, n=100, d=3, k=3, centres=0, sigma=0.0, q=10, copies=0)]
    monkeypatch.setattr(evaluator, "DEV_SETS", small_set)
    monkeypatch.setattr(v2, "DEV_SETS", small_set)
    monkeypatch.delenv("OE_PILOT_EXECUTOR", raising=False)
    monkeypatch.setattr(evaluator, "append_candidate", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(v2, "append_candidate", lambda *_args, **_kwargs: None)
    modes = []

    def traced_run_candidate(code, driver, payload, **kwargs):
        modes.append(payload["mode"])
        return real_run_candidate(code, driver, payload, **kwargs)

    monkeypatch.setattr(evaluator, "run_candidate", traced_run_candidate)
    monkeypatch.setattr(v2, "run_candidate", traced_run_candidate)
    program = tmp_path / "fabricated.py"
    program.write_text(
        "import numpy as np\n"
        "def approximate_knn(vectors, queries, k):\n"
        " x=np.asarray(vectors,dtype=float); q=np.asarray(queries,dtype=float)\n"
        " d=((q[:,None,:]-x[None,:,:])**2).sum(axis=2)\n"
        " ids=np.argsort(d,axis=1)[:,:k]\n"
        " return ids,np.full(ids.shape,999.0)\n"
    )

    v1_result = evaluator.evaluate(str(program))
    assert v1_result["combined_score"] > 0
    assert modes == ["working_v1_matched_timed"]

    modes.clear()
    v2_result = v2.evaluate(str(program))
    assert v2_result["valid"] is False
    assert v2_result["combined_score"] == 0.0
    assert any("distance mismatch" in failure for failure in v2_result["validation_failures"])
    assert modes == ["working_v2_preflight"]
