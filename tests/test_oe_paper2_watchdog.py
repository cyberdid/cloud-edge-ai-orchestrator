import pytest

from experiments.openevolve.colab import paper2_watchdog
from experiments.openevolve.colab.paper2_watchdog import (
    CONFIG,
    ROOT,
    budget_exhausted,
    parse_usage_balance,
)


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
