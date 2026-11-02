"""Colab scale study: laptop watchdog decisions (pure function)."""
from experiments.openevolve.colab.watchdog import SESSION, decide

NOW = 1_800_000_000.0


def state(**kw):
    base = {"phase": "running", "blocked": None, "updated": NOW - 60}
    base.update(kw)
    return base


def test_cap_stops_session():
    # cap raised to 380 units on 2026-11-02 (docs/colab_scale/DEVIATIONS.md)
    assert decide(state(), [SESSION], balance=19.0, now=NOW) == "stop_capped"
    assert decide(state(), [], balance=19.0, now=NOW) == "none"
    assert decide(state(), [SESSION], balance=60.0, now=NOW) == "none"  # 339.47 spent: below the cap


def test_done_stops_session():
    assert decide(state(phase="done"), [SESSION], balance=300, now=NOW) == "stop"
    assert decide(state(phase="done"), [], balance=300, now=NOW) == "none"


def test_blocked_stops_and_does_not_restart():
    assert decide(state(phase="blocked", blocked="agy_login"), [SESSION], balance=300, now=NOW) == "stop"
    assert decide(state(phase="blocked", blocked="argon_quota"), [], balance=300, now=NOW) == "none"


def test_starting_state_is_not_restarted():
    assert decide(state(phase="starting", updated=NOW - 600), [], balance=300, now=NOW) == "none"
    assert decide(state(phase="starting", updated=NOW - 4000), [], balance=300, now=NOW) == "start"


def test_no_session_with_work_starts():
    assert decide(None, [], balance=399.47, now=NOW) == "start"
    assert decide(state(updated=NOW - 3600), [], balance=300, now=NOW) == "start"


def test_fresh_heartbeat_protects_live_vm_but_not_a_missing_vm():
    # a VM created on another host shows as "?" here; its driver's heartbeat is fresh
    assert decide(state(updated=NOW - 300), ["?"], balance=300, now=NOW) == "none"
    assert decide(state(updated=NOW - 300), [], balance=300, now=NOW) == "start"


def test_stale_session_is_stopped():
    assert decide(state(updated=NOW - 3600), [SESSION], balance=300, now=NOW) == "stop"
    assert decide(state(updated=NOW - 300), [SESSION], balance=300, now=NOW) == "none"


def test_other_sessions_are_ignored():
    assert decide(state(updated=NOW - 3600), ["someone-else"], balance=300, now=NOW) == "start"


def test_parse_sessions_ignores_cli_messages():
    from experiments.openevolve.colab.watchdog import parse_sessions

    assert parse_sessions("[colab] No active sessions found on server.\n") == []
    listing = "[oe-scale] gpu-l4-s-x | Hardware: L4 | Status: BUSY\n[other] cpu-y | Hardware: CPU\n"
    assert parse_sessions(listing) == ["oe-scale", "other"]


def test_unknown_cross_host_vm_blocks_restart_even_with_stale_heartbeat():
    assert decide(state(updated=NOW - 3600), ["?"], balance=300, now=NOW) == "none"


def test_sessions_rejects_cli_failure(monkeypatch):
    import subprocess
    import pytest
    from experiments.openevolve.colab import watchdog
    monkeypatch.setattr(watchdog, "_run", lambda *a, **k: subprocess.CompletedProcess(a, 1, stdout="", stderr="failed"))
    with pytest.raises(RuntimeError):
        watchdog.sessions()


def test_sessions_rejects_unrecognised_empty_output(monkeypatch):
    import subprocess
    import pytest
    from experiments.openevolve.colab import watchdog
    monkeypatch.setattr(watchdog, "_run", lambda *a, **k: subprocess.CompletedProcess(a, 0, stdout="unexpected", stderr=""))
    with pytest.raises(RuntimeError):
        watchdog.sessions()


def test_state_read_failure_is_not_treated_as_missing_state(monkeypatch):
    import subprocess
    import pytest
    from experiments.openevolve.colab import watchdog
    monkeypatch.setattr(watchdog, "_run", lambda *a, **k: subprocess.CompletedProcess(a, 1, stdout="", stderr="failed"))
    with pytest.raises(RuntimeError):
        watchdog.read_state()


def test_active_assignment_without_listed_session_fails_closed(monkeypatch, tmp_path):
    import sys
    import pytest
    from experiments.openevolve.colab import watchdog
    monkeypatch.setattr(watchdog, "read_state", lambda: state())
    monkeypatch.setattr(watchdog, "sessions", lambda: [])
    monkeypatch.setattr(watchdog, "usage", lambda: (300.0, 1), raising=False)
    monkeypatch.setattr(watchdog, "CACHE", tmp_path)
    monkeypatch.setattr(sys, "argv", ["watchdog"])
    with pytest.raises(RuntimeError, match="assignment"):
        watchdog.main()


def test_unknown_assignment_at_cap_reports_manual_intervention(monkeypatch, tmp_path):
    import sys
    import pytest
    from experiments.openevolve.colab import watchdog
    monkeypatch.setattr(watchdog, "read_state", lambda: state())
    monkeypatch.setattr(watchdog, "sessions", lambda: ["?"])
    monkeypatch.setattr(watchdog, "usage", lambda: (19.0, 1))  # spent 380.47 >= cap 380
    monkeypatch.setattr(watchdog, "CACHE", tmp_path)
    monkeypatch.setattr(sys, "argv", ["watchdog"])
    with pytest.raises(RuntimeError, match="manual attention"):
        watchdog.main()


def test_exec_code_restarts_kernel_once_after_a_hang(monkeypatch, tmp_path):
    import subprocess

    from experiments.openevolve.colab import watchdog

    monkeypatch.setattr(watchdog, "CACHE", tmp_path)
    calls = []

    def fake_colab(*args, timeout=1800):
        calls.append(args[0])
        if args[0] == "exec" and calls.count("exec") == 1:
            raise subprocess.TimeoutExpired("colab", timeout)
        return subprocess.CompletedProcess(args, 0, stdout="ok", stderr="")

    monkeypatch.setattr(watchdog, "colab", fake_colab)
    assert watchdog.exec_code("print(1)", timeout=5) == "ok"
    assert calls == ["exec", "restart-kernel", "exec"]


def test_start_vm_stops_allocated_vm_and_clears_starting_state_on_bootstrap_failure(monkeypatch, tmp_path):
    import subprocess
    import pytest
    from experiments.openevolve.colab import watchdog

    monkeypatch.setattr(watchdog, "CACHE", tmp_path)
    states = []
    colab_calls = []
    monkeypatch.setattr(watchdog, "write_state", lambda s: states.append(dict(s)))
    monkeypatch.setattr(watchdog, "_run", lambda *a, **k: subprocess.CompletedProcess(a, 0, stdout="", stderr=""))

    def fake_colab(*args, timeout=1800):
        colab_calls.append(args[0])
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    monkeypatch.setattr(watchdog, "colab", fake_colab)
    monkeypatch.setattr(watchdog, "exec_code", lambda code, timeout=600: "failed" if "bootstrap.sh" in code else "ok")

    with pytest.raises(RuntimeError, match="bootstrap failed"):
        watchdog.start_vm(state())

    assert "new" in colab_calls and colab_calls[-1] == "stop"
    assert states[0]["phase"] == "starting"
    assert states[-1]["phase"] == "recovering" and states[-1]["updated"] == 0
    assert decide(states[-1], [], balance=300, now=NOW) == "start"
    assert decide(states[-1], [SESSION], balance=300, now=NOW) == "stop"

