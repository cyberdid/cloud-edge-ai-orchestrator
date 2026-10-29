"""OpenEvolve pilot: run driver for arms L / LC / B (stub LLM, real Docker sandbox)."""
import json
import os

import pytest

pytest.importorskip("openevolve")

from experiments.openevolve import run_pilot  # noqa: E402
from experiments.openevolve.best_of_n import run_best_of_n  # noqa: E402
from experiments.openevolve.router import Router  # noqa: E402

real_docker = pytest.mark.skipif(
    os.environ.get("RUN_DOCKER_TESTS") != "1", reason="set RUN_DOCKER_TESTS=1"
)


def lines(path):
    return [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []


@real_docker
def test_stub_run_L_circle_packing_5_iterations(tmp_path, monkeypatch):
    monkeypatch.setenv("OE_PILOT_MAX_ITER", "5")
    manifest = run_pilot.run("circle_packing", "L", 101, tmp_path / "run", resume=False, llm="stub")
    assert len(lines(tmp_path / "run/calls.jsonl")) == 5
    assert len(lines(tmp_path / "run/candidates.jsonl")) >= 1
    assert manifest["status"] == "complete"
    saved = json.loads((tmp_path / "run/manifest.json").read_text())
    assert saved["openevolve_version"] == "0.4.0" and saved["cloud_share"] == 0.0
    assert len(saved["config_sha256"]) == 64


@real_docker
def test_resume_does_not_respend_calls(tmp_path, monkeypatch):
    monkeypatch.setenv("OE_PILOT_MAX_ITER", "3")
    run_pilot.run("circle_packing", "L", 101, tmp_path / "run", resume=False, llm="stub")
    monkeypatch.setenv("OE_PILOT_MAX_ITER", "5")
    run_pilot.run("circle_packing", "L", 101, tmp_path / "run", resume=True, llm="stub")
    assert [c["call_index"] for c in lines(tmp_path / "run/calls.jsonl")] == [0, 1, 2, 3, 4]


@real_docker
def test_best_of_n_uses_seed_only(tmp_path):
    seen = []

    def gemma(messages):
        seen.append("\n".join(m["content"] for m in messages))
        return run_pilot.stub_reply("circle_packing"), 1, 1

    router = Router(101, 0.0, tmp_path / "calls.jsonl", gemma=gemma, argon=gemma, budget=3)
    os.environ["OE_PILOT_CANDIDATE_LOG"] = str(tmp_path / "candidates.jsonl")
    try:
        summary = run_best_of_n("circle_packing", router, 3, tmp_path, seed=101, resume=False)
    finally:
        del os.environ["OE_PILOT_CANDIDATE_LOG"]
    seed_code = run_pilot.TASKS["circle_packing"]["initial"].read_text()
    assert len(seen) == 3
    assert all("n = 26" in prompt and "# stub" not in prompt for prompt in seen)
    assert summary["evaluated"] == 3
    assert all(c["code"] != seed_code for c in lines(tmp_path / "candidates.jsonl")[1:])


@real_docker
def test_no_diff_reply_counts_as_failed_attempt(tmp_path):
    def prose(messages):
        return "I would try a hexagonal layout.", 1, 1

    router = Router(101, 0.0, tmp_path / "calls.jsonl", gemma=prose, argon=prose, budget=2)
    summary = run_best_of_n("circle_packing", router, 2, tmp_path, seed=101, resume=False)
    assert len(lines(tmp_path / "calls.jsonl")) == 2
    assert summary["failed_attempts"] == 2 and summary["evaluated"] == 0


def test_cloud_share_by_arm():
    assert run_pilot.CLOUD_SHARE == {"L": 0.0, "LC": 0.1, "B": 0.0}
