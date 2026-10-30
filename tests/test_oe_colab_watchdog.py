"""Colab scale study: laptop watchdog decisions (pure function)."""
from experiments.openevolve.colab.watchdog import SESSION, decide

NOW = 1_800_000_000.0


def state(**kw):
    base = {"phase": "running", "blocked": None, "updated": NOW - 60}
    base.update(kw)
    return base


def test_cap_stops_session():
    assert decide(state(), [SESSION], balance=60.0, now=NOW) == "stop_capped"
    assert decide(state(), [], balance=60.0, now=NOW) == "none"


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
    assert decide(state(), [], balance=300, now=NOW) == "start"


def test_stale_session_is_stopped():
    assert decide(state(updated=NOW - 3600), [SESSION], balance=300, now=NOW) == "stop"
    assert decide(state(updated=NOW - 300), [SESSION], balance=300, now=NOW) == "none"


def test_other_sessions_are_ignored():
    assert decide(state(), ["someone-else"], balance=300, now=NOW) == "start"


def test_parse_sessions_ignores_cli_messages():
    from experiments.openevolve.colab.watchdog import parse_sessions

    assert parse_sessions("[colab] No active sessions found on server.\n") == []
    listing = "[oe-scale] gpu-l4-s-x | Hardware: L4 | Status: BUSY\n[other] cpu-y | Hardware: CPU\n"
    assert parse_sessions(listing) == ["oe-scale", "other"]


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
