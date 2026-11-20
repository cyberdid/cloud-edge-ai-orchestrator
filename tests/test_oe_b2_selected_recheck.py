import hashlib
import json
from pathlib import Path

import pytest

from experiments.openevolve.colab.b2_protocol import load_protocol


def spec():
    return load_protocol(Path(__file__).parents[1] / "experiments/openevolve/configs/b2_estimation_2026-11-20.json")


def create_run(root, protocol, pair, version, *, missing=False, wrong_hash=False):
    run_dir = root / "runs" / pair["pair_id"] / version
    run_dir.mkdir(parents=True)
    code = "def nearest_neighbors(x, q, k):\n    return [], []\n"
    program = run_dir / "openevolve/best/best_program.py"
    program.parent.mkdir(parents=True)
    program.write_text(code)
    digest = hashlib.sha256(code.encode()).hexdigest()
    manifest = {"status": "complete", "task": "knn", "arm": "L", "seed": pair["seed"],
        "git_sha": "b" * 40, "protocol_id": protocol["protocol_id"],
        "protocol_sha256": "a" * 64, "model_digest": protocol["model_digest"],
        "evaluator_version": version,
        "selection": {"status": "missing" if missing else "recorded",
                      "path": "openevolve/best/best_program.py",
                      "code_sha256": ("c" * 64 if wrong_hash else digest)}}
    (run_dir / "manifest.json").write_text(json.dumps(manifest))
    return run_dir


def test_recheck_uses_official_selected_artifact_and_records_hashes(tmp_path):
    from experiments.openevolve.recheck_b2_selected import recheck_selected

    protocol = spec()
    for pair in protocol["pairs"]:
        for version in ("v1_matched", "v2"):
            create_run(tmp_path, protocol, pair, version)
    seen = []

    def checker(code):
        seen.append(code)
        return {"valid": True, "score": 0.5, "holdout_recall": 1.0,
                "holdout_speed_ratio": 1.1, "reasons": []}

    report = recheck_selected(tmp_path, protocol, protocol_sha256="a" * 64,
                              source_commit="b" * 40, checker=checker)

    assert report["assigned_runs"] == 12
    assert report["checked_runs"] == 12
    assert report["invalid_selected_runs"] == 0
    assert len(seen) == 12
    assert all(row["selection_sha256"] == hashlib.sha256(seen[0].encode()).hexdigest()
               for row in report["runs"])


def test_recheck_counts_missing_or_tampered_selection_as_invalid(tmp_path):
    from experiments.openevolve.recheck_b2_selected import recheck_selected

    protocol = spec()
    for i, pair in enumerate(protocol["pairs"]):
        for version in ("v1_matched", "v2"):
            create_run(tmp_path, protocol, pair, version,
                       missing=(i == 0 and version == "v1_matched"),
                       wrong_hash=(i == 1 and version == "v2"))

    report = recheck_selected(tmp_path, protocol, protocol_sha256="a" * 64,
                              source_commit="b" * 40,
                              checker=lambda code: {"valid": True, "score": 1.0})

    assert report["assigned_runs"] == 12
    assert report["invalid_selected_runs"] == 2
    assert report["checked_runs"] == 10


def test_recheck_rejects_run_manifest_from_another_frozen_cohort(tmp_path):
    from experiments.openevolve.recheck_b2_selected import recheck_selected

    protocol = spec()
    pair = protocol["pairs"][0]
    create_run(tmp_path, protocol, pair, "v2")
    path = tmp_path / "runs" / pair["pair_id"] / "v2" / "manifest.json"
    data = json.loads(path.read_text())
    data["protocol_sha256"] = "f" * 64
    path.write_text(json.dumps(data))

    with pytest.raises(ValueError, match="protocol_sha256"):
        recheck_selected(tmp_path, protocol, protocol_sha256="a" * 64,
                         source_commit="b" * 40, checker=lambda _: {})


def test_recheck_retains_partial_run_as_incomplete_instead_of_aborting(tmp_path):
    from experiments.openevolve.recheck_b2_selected import recheck_selected

    protocol = spec()
    for pair in protocol["pairs"]:
        for version in ("v1_matched", "v2"):
            create_run(tmp_path, protocol, pair, version)
    path = tmp_path / "runs" / "P1" / "v2" / "manifest.json"
    data = json.loads(path.read_text())
    data["status"] = "interrupted"
    path.write_text(json.dumps(data))

    report = recheck_selected(tmp_path, protocol, protocol_sha256="a" * 64,
                              source_commit="b" * 40,
                              checker=lambda _: {"valid": True, "score": 1.0})

    row = next(row for row in report["runs"] if row["run_id"] == "P1:v2")
    assert row["valid"] is False
    assert row["reason"] == "run_not_complete"
