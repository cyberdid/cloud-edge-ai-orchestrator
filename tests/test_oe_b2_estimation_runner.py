import json
from pathlib import Path

import pytest

from experiments.openevolve.colab.b2_protocol import load_protocol


PROTOCOL_PATH = Path(__file__).parents[1] / "experiments/openevolve/configs/b2_estimation_2026-11-20.json"
PROTOCOL_SHA256 = "a" * 64
SOURCE_COMMIT = "b" * 40


class MemoryMirror:
    def __init__(self):
        self.restores = 0
        self.syncs = 0

    def restore(self):
        self.restores += 1

    def sync(self):
        self.syncs += 1


def fake_run_factory(*, corrupt=None):
    calls = []

    def fake_run(task, arm, seed, out, resume, llm, *, evaluator_version):
        calls.append((seed, evaluator_version, resume, llm))
        out.mkdir(parents=True, exist_ok=True)
        rows = [{"call_index": i, "model": "gemma", "status": "ok"} for i in range(20)]
        if corrupt == (seed, evaluator_version):
            rows[-1]["call_index"] = rows[-2]["call_index"]
        (out / "calls.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
        manifest = {
            "status": "complete", "task": task, "arm": arm, "seed": seed,
            "git_sha": SOURCE_COMMIT,
            "max_iterations": 20, "protocol_id": "paper2-b2-estimation-2026-11-20-v1",
            "protocol_sha256": PROTOCOL_SHA256, "model_digest": "6114515d63c1",
            "evaluator_version": evaluator_version,
            "evaluator_sha256": "c" * 64,
            "selection": {"status": "recorded", "code_sha256": "d" * 64},
        }
        if corrupt == "wrong_protocol":
            manifest["protocol_sha256"] = "e" * 64
        (out / "manifest.json").write_text(json.dumps(manifest))
        return manifest

    return fake_run, calls


def interrupted_once_factory():
    calls = []
    interrupted = {"done": False}

    def fake_run(task, arm, seed, out, resume, llm, *, evaluator_version):
        calls.append((seed, evaluator_version, resume, llm))
        out.mkdir(parents=True, exist_ok=True)
        is_interruption = seed == 20261121 and evaluator_version == "v1_matched" and not interrupted["done"]
        count = 19 if is_interruption else 20
        rows = [{"call_index": i, "model": "gemma", "status": "ok"} for i in range(count)]
        (out / "calls.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
        manifest = {
            "status": "incomplete" if is_interruption else "complete", "task": task, "arm": arm, "seed": seed,
            "git_sha": SOURCE_COMMIT,
            "max_iterations": 20, "protocol_id": "paper2-b2-estimation-2026-11-20-v1",
            "protocol_sha256": PROTOCOL_SHA256, "model_digest": "6114515d63c1",
            "evaluator_version": evaluator_version, "evaluator_sha256": "c" * 64,
            "selection": {"status": "recorded", "code_sha256": "d" * 64},
        }
        (out / "manifest.json").write_text(json.dumps(manifest))
        if is_interruption:
            interrupted["done"] = True
        return manifest

    return fake_run, calls


def runner():
    try:
        from experiments.openevolve.colab.paper2_b2_estimation import run_cohort
    except ModuleNotFoundError:
        pytest.fail("resumable B2 cohort runner is not implemented")
    return run_cohort


def test_runner_executes_fixed_balanced_pairs_and_mirrors_terminal_state(tmp_path):
    protocol = load_protocol(PROTOCOL_PATH)
    fake_run, calls = fake_run_factory()
    mirror = MemoryMirror()

    result = runner()(protocol, PROTOCOL_SHA256, llm="stub", root=tmp_path,
                      mirror=mirror, run_one=fake_run, source_commit=SOURCE_COMMIT)

    assert result["phase"] == "complete"
    assert len(calls) == 12
    assert [version for _, version, _, _ in calls] == [
        version for pair in protocol["pairs"] for version in pair["order"]
    ]
    assert mirror.restores == 1
    assert mirror.syncs >= 12
    state = json.loads((tmp_path / "state.json").read_text())
    assert state["phase"] == "complete"
    assert state["completed_runs"] == 12


def test_real_model_calls_refuse_unfrozen_protocol_before_run(tmp_path):
    protocol = load_protocol(PROTOCOL_PATH)
    protocol["status"] = "draft"
    fake_run, calls = fake_run_factory()

    with pytest.raises(ValueError, match="frozen"):
        runner()(protocol, PROTOCOL_SHA256, llm="real", root=tmp_path,
                 mirror=MemoryMirror(), run_one=fake_run, source_commit=SOURCE_COMMIT)

    assert calls == []


def test_runner_refuses_duplicate_call_indices(tmp_path):
    protocol = load_protocol(PROTOCOL_PATH)
    fake_run, _ = fake_run_factory(corrupt=(20261120, "v2"))

    with pytest.raises(ValueError, match="call indices"):
        runner()(protocol, PROTOCOL_SHA256, llm="stub", root=tmp_path,
                 mirror=MemoryMirror(), run_one=fake_run, source_commit=SOURCE_COMMIT)


def test_runner_rejects_protocol_mismatch_in_saved_manifest(tmp_path):
    protocol = load_protocol(PROTOCOL_PATH)
    fake_run, _ = fake_run_factory(corrupt="wrong_protocol")

    with pytest.raises(ValueError, match="protocol"):
        runner()(protocol, PROTOCOL_SHA256, llm="stub", root=tmp_path,
                 mirror=MemoryMirror(), run_one=fake_run, source_commit=SOURCE_COMMIT)


def test_runner_does_not_reopen_terminal_cohort(tmp_path):
    protocol = load_protocol(PROTOCOL_PATH)
    (tmp_path / "state.json").write_text(json.dumps({"phase": "complete"}))
    fake_run, calls = fake_run_factory()

    with pytest.raises(ValueError, match="terminal"):
        runner()(protocol, PROTOCOL_SHA256, llm="stub", root=tmp_path,
                 mirror=MemoryMirror(), run_one=fake_run, source_commit=SOURCE_COMMIT)

    assert calls == []


def test_runner_rejects_source_commit_mismatch_in_run_manifest(tmp_path):
    protocol = load_protocol(PROTOCOL_PATH)
    fake_run, _ = fake_run_factory()

    def wrong_source(*args, **kwargs):
        manifest = fake_run(*args, **kwargs)
        manifest["git_sha"] = "f" * 40
        out = args[3]
        (out / "manifest.json").write_text(json.dumps(manifest))
        return manifest

    with pytest.raises(ValueError, match="git_sha"):
        runner()(protocol, PROTOCOL_SHA256, llm="stub", root=tmp_path,
                 mirror=MemoryMirror(), run_one=wrong_source, source_commit=SOURCE_COMMIT)


def test_runner_resumes_only_incomplete_arm_without_replaying_completed_runs(tmp_path):
    protocol = load_protocol(PROTOCOL_PATH)
    fake_run, calls = interrupted_once_factory()
    mirror = MemoryMirror()

    with pytest.raises(RuntimeError, match="incomplete"):
        runner()(protocol, PROTOCOL_SHA256, llm="stub", root=tmp_path,
                 mirror=mirror, run_one=fake_run, source_commit=SOURCE_COMMIT)
    interrupted_state = json.loads((tmp_path / "state.json").read_text())
    assert interrupted_state["phase"] == "interrupted"
    assert interrupted_state["completed_runs"] == 2

    completed_state = runner()(protocol, PROTOCOL_SHA256, llm="stub", root=tmp_path,
                               mirror=mirror, run_one=fake_run, source_commit=SOURCE_COMMIT)

    assert completed_state["phase"] == "complete"
    assert completed_state["completed_runs"] == 12
    assert calls[3] == (20261121, "v1_matched", True, "stub")
    assert len(calls) == 13
