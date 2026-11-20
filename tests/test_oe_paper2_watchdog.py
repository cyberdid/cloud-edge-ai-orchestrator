import pytest

from experiments.openevolve.colab import paper2_watchdog
from experiments.openevolve.colab.paper2_watchdog import (
    CONFIG,
    ROOT,
    budget_exhausted,
    parse_usage_balance,
)


def _completed_manifest(version):
    return {
        "status": "complete", "evaluator_version": version, "task": "knn", "arm": "L",
        "seed": 20261119, "git_sha": "a" * 40,
        "protocol_id": "evaluator-v2-pilot-2026-11-19", "protocol_sha256": "b" * 64,
        "model_digest": "model-digest", "wall_s": 123.0,
    }


def _complete_state():
    return {
        "git_sha": "a" * 40, "protocol_id": "evaluator-v2-pilot-2026-11-19",
        "protocol_sha256": "b" * 64, "model": "gemma4:12b", "task": "knn",
        "seed": 20261119, "calls_per_condition": 20,
        "bootstrap": {"model_digest": "model-digest"},
    }


def _call_rows():
    return [{"call_index": i, "model": "gemma", "status": "ok"} for i in range(20)]


def test_completed_manifests_and_contiguous_ledgers_reconcile_state():
    state = _complete_state()
    manifests = {version: _completed_manifest(version) for version in ("v1", "v2")}
    calls = {version: _call_rows() for version in ("v1", "v2")}

    summaries = paper2_watchdog.verify_completed_runs(state, manifests, calls)

    assert [row["evaluator_version"] for row in summaries] == ["v1", "v2"]
    assert all(row["calls"] == 20 and row["status"] == "complete" for row in summaries)


@pytest.mark.parametrize("mutate", [
    lambda manifests, calls: manifests["v2"].update(status="running"),
    lambda manifests, calls: calls["v1"].pop(),
    lambda manifests, calls: calls["v2"].__setitem__(19, {"call_index": 18, "model": "gemma"}),
    lambda manifests, calls: manifests["v1"].update(git_sha="c" * 40),
    lambda manifests, calls: calls["v1"][0].update(model="argon"),
])
def test_inconsistent_or_incomplete_records_are_not_reconciled(mutate):
    state = _complete_state()
    manifests = {version: _completed_manifest(version) for version in ("v1", "v2")}
    calls = {version: _call_rows() for version in ("v1", "v2")}
    mutate(manifests, calls)

    with pytest.raises(ValueError):
        paper2_watchdog.verify_completed_runs(state, manifests, calls)


def test_parse_usage_balance_from_colab_cli():
    output = "Current balance: 224.37 compute units\nUsage rate: 1.94/hr\nActive assignments: 1"

    assert parse_usage_balance(output) == pytest.approx(224.37)


def test_budget_stops_only_at_or_above_preregistered_cap():
    assert budget_exhausted(224.37, 209.38, 15.0) is False
    assert budget_exhausted(224.37, 209.37, 15.0) is True


def test_mutable_watchdog_control_state_is_inside_writable_root():
    assert CONFIG.is_relative_to(ROOT)


def test_prepare_session_uploads_source_and_credentials(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(paper2_watchdog, "ARCHIVE", tmp_path / "oe.tar.gz")
    monkeypatch.setattr(paper2_watchdog, "KEY", tmp_path / "sa-key.json")
    monkeypatch.setattr(paper2_watchdog, "REPO", tmp_path / "repo")
    monkeypatch.setattr(paper2_watchdog, "_run", lambda args, **kwargs: calls.append(args))

    paper2_watchdog._prepare_session("a" * 40)

    assert len(calls) == 3
    assert all(call[1] == "upload" for call in calls)
    assert calls[-1][-1] == "/content/paper2_pilot.py"


@pytest.mark.parametrize("state, expected", [(None, True), ({"phase": "error"}, True),
                                             ({"phase": "interrupted"}, True),
                                             ({"phase": "running"}, False),
                                             ({"phase": "complete"}, False)])
def test_error_states_require_fresh_colab_session(state, expected):
    assert paper2_watchdog._needs_fresh_session(state) is expected
