"""Validation for the budget-capped Paper II B2 estimation protocol."""
from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Any


EXPECTED_ARMS = {"v1_matched", "v2"}
EXPECTED_SEED_COUNT = 6
EXPECTED_CALLS_PER_ARM = 20
EXPECTED_MODEL = "gemma4:12b"
EXPECTED_MODEL_DIGEST = "6114515d63c1"
EXPECTED_GCS_BUCKET = "oe-colab-scale-0903767759"
EXPECTED_GCS_PREFIX = "paper2-v2/b2-estimation-"
EXPECTED_COLAB_SESSION = "paper2-b2-estimation"
EXPECTED_DOCKER_IMAGE = "antigravity-orion:local"
EXPECTED_DOCKER_IMAGE_DIGEST = (
    "sha256:aae973535ab734c026d878c0ef1d7ec1164b83538bdcd63c192476672cb27ae2"
)
EXPECTED_HOLDOUT_DEFINITION = "experiments/openevolve/tasks/knn/recheck.py::HOLDOUT_SETS"
EXPECTED_HOLDOUT_SEEDS = [9001, 9002, 9003]
CHECKER_COMPONENTS = (
    "antigravity_hybrid_orchestrator/integrity/oracle.py",
    "antigravity_hybrid_orchestrator/runtime/sandbox.py",
    "experiments/openevolve/candidate_log.py",
    "experiments/openevolve/tasks/knn/recheck.py",
    "experiments/openevolve/tasks/knn/data.py",
    "experiments/openevolve/tasks/knn/driver.py",
    "experiments/openevolve/sandbox_exec.py",
    "experiments/openevolve/static_check.py",
    "experiments/openevolve/tasks/knn/working_evaluator.py",
    "experiments/openevolve/recheck_b2_selected.py",
    "experiments/openevolve/analyze_b2_estimation.py",
)


def checker_fingerprint(repo_root: str | Path) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    components = {}
    for relative in CHECKER_COMPONENTS:
        path = root / relative
        if not path.is_file():
            raise ValueError(f"checker component is missing: {relative}")
        components[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    joined = "".join(f"{path}\0{digest}\n" for path, digest in sorted(components.items()))
    return {"components": components, "sha256": hashlib.sha256(joined.encode()).hexdigest()}


def _validate_frozen_verification(protocol: dict[str, Any]) -> None:
    verification = protocol.get("verification")
    if not isinstance(verification, dict):
        raise ValueError("frozen protocol requires verification metadata")
    if verification.get("docker_image") != EXPECTED_DOCKER_IMAGE:
        raise ValueError("frozen verification Docker image name is not registered")
    if verification.get("docker_image_digest") != EXPECTED_DOCKER_IMAGE_DIGEST:
        raise ValueError("frozen verification Docker image digest is not registered")
    if verification.get("holdout_definition") != EXPECTED_HOLDOUT_DEFINITION:
        raise ValueError("frozen verification holdout definition is not registered")
    if verification.get("holdout_seeds") != EXPECTED_HOLDOUT_SEEDS:
        raise ValueError("frozen verification holdout seeds are not registered")
    if verification.get("holdout_visibility") != "source_visible_fixed":
        raise ValueError("frozen verification must disclose source-visible holdout status")
    expected = checker_fingerprint(Path(__file__).resolve().parents[3])
    if verification.get("components") != expected["components"] or verification.get("sha256") != expected["sha256"]:
        raise ValueError("frozen verification checker fingerprint does not match source components")


def load_protocol(path: str | Path) -> dict[str, Any]:
    """Load and validate the fixed six-pair B2 estimation configuration."""
    try:
        protocol = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read valid protocol JSON: {exc}") from exc
    if not isinstance(protocol, dict):
        raise ValueError("protocol must be a JSON object")
    required = {
        "protocol_id", "status", "study_type", "task", "model", "model_digest",
        "calls_per_arm", "max_total_calls", "bucket", "gcs_prefix", "colab_session",
        "max_account_draw_cu", "minimum_balance_cu", "pairs",
    }
    missing = sorted(required - protocol.keys())
    if missing:
        raise ValueError(f"protocol is missing required fields: {', '.join(missing)}")
    if protocol["status"] not in {"draft", "frozen"}:
        raise ValueError("protocol status must be draft or frozen")
    if protocol["status"] == "frozen":
        _validate_frozen_verification(protocol)
    if protocol["study_type"] != "estimation_feasibility":
        raise ValueError("this runner accepts estimation_feasibility protocols only")
    if protocol["task"] != "knn":
        raise ValueError("B2 estimation task must be knn")
    if protocol["model"] != EXPECTED_MODEL or protocol["model_digest"] != EXPECTED_MODEL_DIGEST:
        raise ValueError("model name and immutable digest must match the frozen Gemma 4 12B build")
    if protocol["calls_per_arm"] != EXPECTED_CALLS_PER_ARM:
        raise ValueError("calls_per_arm must remain 20 for both conditions")
    if protocol["max_total_calls"] != EXPECTED_SEED_COUNT * 2 * EXPECTED_CALLS_PER_ARM:
        raise ValueError("total call budget must equal six pairs times two arms times 20 calls")
    if protocol["bucket"] != EXPECTED_GCS_BUCKET:
        raise ValueError("B2 must use the registered research artifact bucket")
    prefix = protocol["gcs_prefix"]
    if not isinstance(prefix, str) or not prefix.startswith(EXPECTED_GCS_PREFIX) or ".." in prefix:
        raise ValueError("B2 requires a dedicated paper2-v2/b2-estimation-* GCS prefix")
    if protocol["gcs_prefix"] == "paper2-v2/b1":
        raise ValueError("B2 cannot write into the B1 artifact prefix")
    if protocol["colab_session"] != EXPECTED_COLAB_SESSION:
        raise ValueError("B2 must use its isolated Colab session name")
    if protocol["max_account_draw_cu"] != 70.4 or protocol["minimum_balance_cu"] != 100.0:
        raise ValueError("B2 account draw cap and reserve do not match the approved budget envelope")

    pairs = protocol["pairs"]
    if not isinstance(pairs, list) or len(pairs) != EXPECTED_SEED_COUNT:
        raise ValueError("protocol must contain exactly six seed pairs")
    pair_ids, seeds, first_arms = set(), set(), []
    for pair in pairs:
        if not isinstance(pair, dict) or not {"pair_id", "seed", "order"} <= pair.keys():
            raise ValueError("each pair must contain pair_id, seed, and order")
        if not isinstance(pair["pair_id"], str) or not pair["pair_id"]:
            raise ValueError("pair_id must be a non-empty string")
        if not isinstance(pair["seed"], int) or isinstance(pair["seed"], bool):
            raise ValueError("each task seed must be an integer")
        order = pair["order"]
        if not isinstance(order, list) or len(order) != 2 or set(order) != EXPECTED_ARMS:
            raise ValueError("each pair order must contain v1_matched and v2 exactly once")
        pair_ids.add(pair["pair_id"])
        seeds.add(pair["seed"])
        first_arms.append(order[0])
    if len(pair_ids) != EXPECTED_SEED_COUNT or len(seeds) != EXPECTED_SEED_COUNT:
        raise ValueError("pair IDs and seeds must each be unique")
    if first_arms.count("v1_matched") != 3 or first_arms.count("v2") != 3:
        raise ValueError("arm order must be balanced: three pairs start with each condition")
    return protocol
