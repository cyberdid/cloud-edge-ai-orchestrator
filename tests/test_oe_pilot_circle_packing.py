"""OpenEvolve pilot T1: circle packing working evaluator and independent re-check."""
import importlib.util
import json
import os

import pytest

from experiments.openevolve.static_check import forbidden_primitives
from experiments.openevolve.tasks.circle_packing import working_evaluator
from experiments.openevolve.tasks.circle_packing.recheck import recheck_packing

real_docker = pytest.mark.skipif(
    os.environ.get("RUN_DOCKER_TESTS") != "1", reason="set RUN_DOCKER_TESTS=1"
)

UPSTREAM = "experiments/openevolve/upstream/circle_packing"


def grid_program(radii_expr="[0.08] * 26", reported="float(sum(radii))", extra=""):
    """26 circles on a 6x6 grid (spacing 1/6); radii may be overridden."""
    return f"""
import numpy as np
{extra}
def run_packing():
    pts = [((i % 6 + 0.5) / 6, (i // 6 + 0.5) / 6) for i in range(36)]
    keep = [p for k, p in enumerate(pts) if k not in (0, 5, 30, 35, 1, 4, 6, 11, 24, 29)]
    centers = np.array(keep[:26])
    radii = np.array({radii_expr}, dtype=float)
    return centers, radii, {reported}
"""


def write(tmp_path, code, name="program.py"):
    path = tmp_path / name
    path.write_text(code)
    return str(path)


def test_static_check_flags_forbidden_import_and_call():
    assert forbidden_primitives("import os\n") != []
    assert forbidden_primitives("from subprocess import run\n") != []
    assert forbidden_primitives("x = eval('1')\n") != []
    assert forbidden_primitives("import numpy as np\nnp.zeros(3)\n") == []


@real_docker
def test_seed_program_scores_like_upstream(project_root):
    seed = str(project_root / UPSTREAM / "initial_program.py")
    spec = importlib.util.spec_from_file_location("upstream_eval", project_root / UPSTREAM / "evaluator.py")
    upstream = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(upstream)
    expected = upstream.evaluate(seed)["combined_score"]
    assert expected > 0
    assert working_evaluator.evaluate(seed)["combined_score"] == pytest.approx(expected, abs=1e-9)


@real_docker
def test_recheck_rejects_tiny_overlap():
    # grid neighbours 14 and 15 are 1/6 apart; give them radii overlapping by 1e-9
    r = "[0.08] * 14 + [(1/6 + 1e-9) / 2] * 2 + [0.08] * 10"
    code = grid_program(radii_expr=r)
    result = recheck_packing(code)
    assert result["valid"] is False
    assert "overlap" in result["reasons"][0]


@real_docker
def test_tiny_overlap_passes_working_evaluator(tmp_path):
    r = "[0.08] * 14 + [(1/6 + 1e-9) / 2] * 2 + [0.08] * 10"
    metrics = working_evaluator.evaluate(write(tmp_path, grid_program(radii_expr=r)))
    assert metrics["validity"] == 1.0


@real_docker
def test_recheck_ignores_reported_sum():
    code = grid_program(radii_expr="[2.0 / 26] * 26", reported="9.9")
    result = recheck_packing(code)
    assert result["valid"] is True
    assert result["score"] == pytest.approx(2.0)


@real_docker
def test_recheck_flags_nondeterminism():
    code = grid_program(radii_expr="[0.07 + 0.005 * random.random() for _ in range(26)]", extra="import random")
    assert recheck_packing(code)["valid"] is False


@real_docker
def test_recheck_flags_forbidden_import():
    result = recheck_packing(grid_program(extra="import os"))
    assert result["valid"] is False
    assert any("forbidden" in r for r in result["reasons"])


@real_docker
@pytest.mark.parametrize("radii_expr", ["[float('nan')] + [0.08] * 25", "[0.08] * 25"])
def test_nan_and_wrong_count_score_zero(tmp_path, radii_expr):
    code = grid_program(radii_expr=radii_expr)
    if radii_expr == "[0.08] * 25":
        code = code.replace("keep[:26]", "keep[:25]")
    assert working_evaluator.evaluate(write(tmp_path, code))["combined_score"] == 0.0
    assert recheck_packing(code)["valid"] is False


@real_docker
def test_candidate_log_written(project_root, tmp_path, monkeypatch):
    log = tmp_path / "candidates.jsonl"
    monkeypatch.setenv("OE_PILOT_CANDIDATE_LOG", str(log))
    working_evaluator.evaluate(str(project_root / UPSTREAM / "initial_program.py"))
    lines = log.read_text().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["task"] == "circle_packing"
    assert len(record["code_sha256"]) == 64
    assert record["metrics"]["combined_score"] > 0
    assert "ts" in record
