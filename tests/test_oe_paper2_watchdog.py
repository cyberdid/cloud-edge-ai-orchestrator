import pytest
import subprocess

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


def test_prepare_session_restarts_kernel_after_uploading_pinned_runner(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(paper2_watchdog, "ARCHIVE", tmp_path / "oe.tar.gz")
    monkeypatch.setattr(paper2_watchdog, "KEY", tmp_path / "sa-key.json")
    monkeypatch.setattr(paper2_watchdog, "REPO", tmp_path / "repo")
    monkeypatch.setattr(
        paper2_watchdog,
        "_run",
        lambda args, **kwargs: (calls.append(args) or subprocess.CompletedProcess(args, 0, "PAPER2_KERNEL_READY")),
    )
    monkeypatch.setattr(paper2_watchdog, "_wait_for_kernel_ready", lambda: calls.append(["kernel-ready"]))

    paper2_watchdog._prepare_session("a" * 40)

    assert calls[-2] == [paper2_watchdog.CLI, "restart-kernel", "-s", paper2_watchdog.SESSION]
    assert calls[-1] == ["kernel-ready"]


def test_wait_for_kernel_ready_requires_success_marker(monkeypatch, tmp_path):
    commands = []
    monkeypatch.setattr(paper2_watchdog, "ROOT", tmp_path)
    monkeypatch.setattr(
        paper2_watchdog,
        "_run",
        lambda args, **kwargs: (commands.append(args) or subprocess.CompletedProcess(args, 0, "PAPER2_KERNEL_READY")),
    )

    paper2_watchdog._wait_for_kernel_ready()

    assert commands == [[paper2_watchdog.CLI, "exec", "-s", paper2_watchdog.SESSION,
                         "-f", str(tmp_path / "kernel-ready-probe.py"), "--timeout", "30"]]
