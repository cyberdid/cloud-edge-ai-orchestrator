import numpy as np

from experiments.openevolve.tasks.knn.data import brute_knn
from experiments.openevolve.tasks.knn.validation import validate_knn_output


def fixture_data():
    vectors = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 2.0], [2.0, 2.0]])
    queries = np.array([[0.2, 0.1], [1.4, 1.8]])
    return vectors, queries


def test_accepts_neighbor_ids_with_their_true_squared_distances():
    vectors, queries = fixture_data()
    ids, distances = brute_knn(vectors, queries, 2)

    assert validate_knn_output(vectors, queries, (ids, distances), 2) == []


def test_rejects_correct_shaped_ids_with_fabricated_distances():
    vectors, queries = fixture_data()
    ids, _ = brute_knn(vectors, queries, 2)

    errors = validate_knn_output(vectors, queries, (ids, np.zeros((2, 2))), 2)

    assert any("distance" in error for error in errors)


def test_rejects_non_integer_neighbor_ids():
    vectors, queries = fixture_data()

    errors = validate_knn_output(vectors, queries, (np.array([[0.0, 1.0], [1.0, 3.0]]),
                                                      np.zeros((2, 2))), 2)

    assert any("integer" in error for error in errors)


def test_rejects_out_of_range_and_duplicate_neighbor_ids():
    vectors, queries = fixture_data()
    ids = np.array([[0, 4], [1, 1]])
    distances = np.zeros((2, 2))

    errors = validate_knn_output(vectors, queries, (ids, distances), 2)

    assert any("range" in error for error in errors)
    assert any("duplicate" in error for error in errors)


def test_rejects_nan_distances():
    vectors, queries = fixture_data()
    ids, distances = brute_knn(vectors, queries, 2)
    distances[0, 0] = np.nan

    errors = validate_knn_output(vectors, queries, (ids, distances), 2)

    assert any("finite" in error for error in errors)


def test_v2_rejects_preflight_distance_error_before_timing(tmp_path, monkeypatch):
    from experiments.openevolve.tasks.knn import working_evaluator_v2 as v2
    from experiments.openevolve.sandbox_exec import SandboxResult

    vectors, queries = fixture_data()
    ids, _ = brute_knn(vectors, queries, 2)
    spec = {"seed": 1, "n": 4, "d": 2, "k": 2, "centres": 0, "sigma": 0.0, "q": 2, "copies": 0}
    monkeypatch.setattr(v2, "DEV_SETS", [spec])
    monkeypatch.setattr(v2, "make_set", lambda **_: (vectors, queries, 2, np.empty(0, dtype=np.int64)))
    calls = []

    def fake_run_candidate(code, driver, payload, timeout_s):
        calls.append(payload["mode"])
        return SandboxResult(True, {"sets": [{"ids": ids.tolist(), "dists": np.zeros_like(ids, dtype=float).tolist()}]}, "", 0.01)

    monkeypatch.setattr(v2, "run_candidate", fake_run_candidate)
    candidate = tmp_path / "candidate.py"
    candidate.write_text("def approximate_knn(vectors, queries, k): pass\n")

    metrics = v2.evaluate(str(candidate))

    assert metrics["valid"] is False
    assert metrics["combined_score"] == 0.0
    assert any("distance" in error for error in metrics["validation_failures"])
    assert calls == ["working_v2_preflight"]


def test_v2_scores_only_after_preflight_and_timed_outputs_validate(tmp_path, monkeypatch):
    from experiments.openevolve.tasks.knn import working_evaluator_v2 as v2
    from experiments.openevolve.sandbox_exec import SandboxResult

    vectors, queries = fixture_data()
    ids, distances = brute_knn(vectors, queries, 2)
    spec = {"seed": 1, "n": 4, "d": 2, "k": 2, "centres": 0, "sigma": 0.0, "q": 2, "copies": 0}
    monkeypatch.setattr(v2, "DEV_SETS", [spec])
    monkeypatch.setattr(v2, "make_set", lambda **_: (vectors, queries, 2, np.empty(0, dtype=np.int64)))
    calls = []

    def fake_run_candidate(code, driver, payload, timeout_s):
        calls.append(payload["mode"])
        if payload["mode"] == "working_v2_preflight":
            output = [ids.tolist(), distances.tolist()]
        else:
            output = {"reference_ids": ids.tolist(), "reference_time": 1.0,
                      "runs": [{"time": 0.2, "ids": ids.tolist(), "dists": distances.tolist()} for _ in range(3)]}
        return SandboxResult(True, {"sets": [output]}, "", 0.01)

    monkeypatch.setattr(v2, "run_candidate", fake_run_candidate)
    candidate = tmp_path / "candidate.py"
    candidate.write_text("def approximate_knn(vectors, queries, k): pass\n")

    metrics = v2.evaluate(str(candidate))

    assert metrics["valid"] is True
    assert metrics["combined_score"] > 0
    assert metrics["recall"] == 1.0
    assert calls == ["working_v2_preflight", "working_v2_timed"]
