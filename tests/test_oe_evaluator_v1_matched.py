from types import SimpleNamespace

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
