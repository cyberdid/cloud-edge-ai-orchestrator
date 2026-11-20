import json
from pathlib import Path

import pytest


def valid_protocol():
    return {
        "protocol_id": "paper2-b2-estimation-2026-11-20-v1",
        "status": "draft",
        "study_type": "estimation_feasibility",
        "task": "knn",
        "model": "gemma4:12b",
        "model_digest": "6114515d63c1",
        "calls_per_arm": 20,
        "max_total_calls": 240,
        "bucket": "oe-colab-scale-0903767759",
        "gcs_prefix": "paper2-v2/b2-estimation-2026-11-20",
        "colab_session": "paper2-b2-estimation",
        "max_account_draw_cu": 70.4,
        "minimum_balance_cu": 100.0,
        "pairs": [
            {"pair_id": f"P{i}", "seed": 20261119 + i,
             "order": order}
            for i, order in enumerate([
                ["v2", "v1_matched"], ["v1_matched", "v2"],
                ["v2", "v1_matched"], ["v1_matched", "v2"],
                ["v1_matched", "v2"], ["v2", "v1_matched"],
            ], start=1)
        ],
    }


def loader():
    try:
        from experiments.openevolve.colab.b2_protocol import load_protocol
    except ModuleNotFoundError:
        pytest.fail("B2 protocol loader is not implemented")
    return load_protocol


def write_protocol(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload))
    return path


def test_load_protocol_accepts_balanced_six_pair_estimation_design(tmp_path):
    source = write_protocol(tmp_path / "protocol.json", valid_protocol())

    loaded = loader()(source)

    assert len(loaded["pairs"]) == 6
    assert sum(pair["order"][0] == "v1_matched" for pair in loaded["pairs"]) == 3
    assert loaded["max_total_calls"] == 240


def test_committed_b2_protocol_is_frozen_with_verification_registration():
    from experiments.openevolve.colab.b2_protocol import load_protocol

    source = Path(__file__).parents[1] / "experiments/openevolve/configs/b2_estimation_2026-11-20.json"

    protocol = load_protocol(source)
    assert protocol["status"] == "frozen"
    assert protocol["verification"]["holdout_visibility"] == "source_visible_fixed"
    assert protocol["verification"]["docker_image_digest"].startswith("sha256:")


def _frozen_protocol_with_current_verification():
    from experiments.openevolve.colab.b2_protocol import (
        EXPECTED_DOCKER_IMAGE, EXPECTED_DOCKER_IMAGE_DIGEST, checker_fingerprint,
    )

    payload = valid_protocol()
    payload["status"] = "frozen"
    payload["verification"] = {
        **checker_fingerprint(Path(__file__).parents[1]),
        "docker_image": EXPECTED_DOCKER_IMAGE,
        "docker_image_digest": EXPECTED_DOCKER_IMAGE_DIGEST,
        "holdout_definition": "experiments/openevolve/tasks/knn/recheck.py::HOLDOUT_SETS",
        "holdout_seeds": [9001, 9002, 9003],
        "holdout_visibility": "source_visible_fixed",
    }
    return payload


def test_frozen_protocol_requires_checker_image_and_holdout_registration(tmp_path):
    payload = valid_protocol()
    payload["status"] = "frozen"
    source = write_protocol(tmp_path / "frozen.json", payload)

    with pytest.raises(ValueError, match="verification"):
        loader()(source)


def test_frozen_protocol_accepts_current_verification_fingerprint(tmp_path):
    source = write_protocol(tmp_path / "frozen.json", _frozen_protocol_with_current_verification())

    assert loader()(source)["status"] == "frozen"


def test_frozen_protocol_rejects_checker_component_drift(tmp_path):
    payload = _frozen_protocol_with_current_verification()
    payload["verification"]["components"]["experiments/openevolve/tasks/knn/recheck.py"] = "0" * 64
    source = write_protocol(tmp_path / "frozen.json", payload)

    with pytest.raises(ValueError, match="fingerprint"):
        loader()(source)


@pytest.mark.parametrize("mutation", [
    lambda p: p["pairs"][1].update(seed=p["pairs"][0]["seed"]),
    lambda p: p["pairs"][0].update(order=["v1_matched", "v1_matched"]),
    lambda p: p.update(max_total_calls=280),
    lambda p: p.update(gcs_prefix="paper2-v2/b1"),
    lambda p: p.update(model_digest=""),
])
def test_load_protocol_rejects_invalid_or_b1_overlapping_design(tmp_path, mutation):
    payload = valid_protocol()
    mutation(payload)
    source = write_protocol(tmp_path / "bad-protocol.json", payload)

    with pytest.raises(ValueError):
        loader()(source)
