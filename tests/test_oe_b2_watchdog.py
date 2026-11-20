from pathlib import Path

import pytest

from experiments.openevolve.colab.b2_protocol import load_protocol


def protocol():
    path = Path(__file__).parents[1] / "experiments/openevolve/configs/b2_estimation_2026-11-20.json"
    return load_protocol(path)


def watchdog():
    try:
        from experiments.openevolve.colab.paper2_b2_watchdog import (
            SESSION, PREFIX, can_continue, parse_usage_balance, reconcile_state,
        )
    except ModuleNotFoundError:
        pytest.fail("isolated B2 watchdog is not implemented")
    return SESSION, PREFIX, can_continue, parse_usage_balance, reconcile_state


def test_b2_watchdog_has_isolated_session_and_prefix():
    session, prefix, *_ = watchdog()
    spec = protocol()
    assert session == spec["colab_session"]
    assert prefix == spec["gcs_prefix"]
    assert "b1" not in prefix.lower()
    assert "b12" not in session.lower()


def test_rehearsal_evidence_is_versioned_by_protocol_and_source_hash():
    from experiments.openevolve.colab.paper2_b2_watchdog import _rehearsal_path

    protocol_sha = "a" * 64
    source_a = "b" * 40
    source_b = "c" * 40
    assert _rehearsal_path(protocol_sha, source_a) != _rehearsal_path(protocol_sha, source_b)
    assert _rehearsal_path(protocol_sha, source_a).name.startswith(
        f"no-call-rehearsal-{protocol_sha[:12]}-{source_a[:12]}"
    )


def test_budget_gate_preserves_100_cu_reserve_and_70_4_draw_cap():
    _, _, can_continue, *_ = watchdog()
    assert can_continue(209.95, 200.0, 70.4, 100.0)
    assert not can_continue(209.95, 139.54, 70.4, 100.0)
    assert not can_continue(150.0, 100.0, 70.4, 100.0)
    assert not can_continue(209.95, 99.99, 70.4, 100.0)


def test_usage_parser_fails_closed_when_balance_is_missing():
    *_, parse_usage_balance, _ = watchdog()
    assert parse_usage_balance("Current balance: 123.45 CU") == 123.45
    with pytest.raises(ValueError, match="balance"):
        parse_usage_balance("usage unavailable")


def test_reconcile_requires_all_twelve_complete_manifests_and_exact_ledgers():
    *_, reconcile_state = watchdog()
    spec = protocol()
    state = {"phase": "running", "protocol_id": spec["protocol_id"],
             "protocol_sha256": "a" * 64, "source_commit": "b" * 40,
             "model_digest": spec["model_digest"], "completed_runs": []}
    evidence = {}
    for pair in spec["pairs"]:
        for version in ("v1_matched", "v2"):
            run_id = f"{pair['pair_id']}:{version}"
            evidence[run_id] = {
                "manifest": {"status": "complete", "protocol_id": spec["protocol_id"],
                             "protocol_sha256": "a" * 64, "source_commit": "b" * 40,
                             "model_digest": spec["model_digest"], "seed": pair["seed"],
                             "task": "knn", "arm": "L", "evaluator_version": version,
                             "selection": {"code_sha256": "c" * 64}},
                "calls": [{"call_index": i, "model": "gemma", "status": "ok"} for i in range(20)],
            }
    assert reconcile_state(state, evidence, spec)["phase"] == "complete"
    del evidence["P6:v2"]
    assert reconcile_state(state, evidence, spec) is None


def test_reconcile_rejects_duplicate_indices_and_unexpected_model():
    *_, reconcile_state = watchdog()
    spec = protocol()
    state = {"phase": "running", "protocol_id": spec["protocol_id"],
             "protocol_sha256": "a" * 64, "source_commit": "b" * 40,
             "model_digest": spec["model_digest"], "completed_runs": []}
    evidence = {}
    for pair in spec["pairs"]:
        for version in ("v1_matched", "v2"):
            run_id = f"{pair['pair_id']}:{version}"
            rows = [{"call_index": i, "model": "gemma", "status": "ok"} for i in range(20)]
            evidence[run_id] = {"manifest": {"status": "complete", "protocol_id": spec["protocol_id"],
                "protocol_sha256": "a" * 64, "source_commit": "b" * 40, "model_digest": spec["model_digest"],
                "seed": pair["seed"], "task": "knn", "arm": "L", "evaluator_version": version,
                "selection": {"code_sha256": "c" * 64}}, "calls": rows}
    evidence["P1:v2"]["calls"][0]["call_index"] = 1
    assert reconcile_state(state, evidence, spec) is None


def test_watchdog_dry_run_is_explicitly_no_model_call():
    from experiments.openevolve.colab.paper2_b2_watchdog import build_dry_run_plan

    plan = build_dry_run_plan(protocol(), "b" * 40)
    assert plan["mode"] == "dry_run_no_model_calls"
    assert plan["model_calls"] == 0
    assert plan["colab_session"] == "paper2-b2-estimation"
    assert plan["gcs_prefix"] == "paper2-v2/b2-estimation-2026-11-20"


class MemoryBlob:
    def __init__(self):
        self.data = None

    def upload_from_string(self, data, **kwargs):
        if kwargs.get("if_generation_match") == 0 and self.data is not None:
            raise RuntimeError("sentinel already exists")
        self.data = data.encode() if isinstance(data, str) else data

    def download_as_text(self):
        return self.data.decode()

    def download_as_bytes(self):
        return self.data


class MemoryBucket:
    def __init__(self):
        self.blobs = {}

    def blob(self, name):
        return self.blobs.setdefault(name, MemoryBlob())


def test_no_call_rehearsal_requires_upload_then_restores_sentinel():
    from experiments.openevolve.colab.paper2_b2_rehearsal import run_stage

    bucket = MemoryBucket()
    uploaded = run_stage("upload", protocol_sha256="a" * 64,
                         source_commit="b" * 40, bucket=bucket)
    assert uploaded["phase"] == "uploaded"
    restored = run_stage("restore", protocol_sha256="a" * 64,
                         source_commit="b" * 40, bucket=bucket)
    assert restored["status"] == "passed"
    assert restored["model_calls"] == 0
    assert restored["fresh_runtime_restore"] is True
    assert restored["checkpoint_restore_after_colab_runtime_recreation"] is True
    assert restored["terminal_state_roundtrip"] is True


def test_colab_kernel_launcher_arguments_do_not_break_rehearsal(monkeypatch, capsys):
    import sys
    from experiments.openevolve.colab import paper2_b2_rehearsal as rehearsal

    called = {}
    monkeypatch.setenv("B2_REHEARSAL_STAGE", "upload")
    monkeypatch.setenv("B2_PROTOCOL_SHA256", "a" * 64)
    monkeypatch.setenv("PAPER2_GIT_SHA", "b" * 40)
    monkeypatch.setenv("B2_REHEARSAL_RUN_ID", "test")
    monkeypatch.setattr(sys, "argv", ["colab_kernel_launcher.py", "-f", "connection.json"])
    monkeypatch.setattr(rehearsal, "run_stage", lambda *a, **kw: called.update(stage=a[0], **kw) or {"phase": "uploaded"})

    rehearsal.main()

    assert called["stage"] == "upload"
    assert "B2_REHEARSAL_RESULT" in capsys.readouterr().out


def test_b2_systemd_unit_starts_watchdog_as_a_repo_module():
    unit = (Path(__file__).parents[1] / "scripts/colab/paper2-b2-estimation.service").read_text()

    assert "WorkingDirectory=/opt/paper2-b2/repo" in unit
    assert ("ExecStart=/opt/oe-watchdog/bin/python -m "
            "experiments.openevolve.colab.paper2_b2_watchdog --live") in unit
