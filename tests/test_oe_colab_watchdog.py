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


def test_completed_original_plan_extends_in_place_but_respects_cap_and_blocks():
    old = state(phase="done", total=149, done=149)
    assert decide(old, [SESSION], balance=70, now=NOW) == "extend"
    assert decide(old, [], balance=70, now=NOW) == "start"
    assert decide(old, ["?"], balance=70, now=NOW) == "none"
    assert decide(old, [SESSION], balance=19, now=NOW) == "stop_capped"
    assert decide(dict(old, blocked="argon_quota"), [SESSION], balance=70, now=NOW) == "stop"
    assert decide(dict(old, phase="running"), [SESSION], balance=70, now=NOW) == "none"
    assert decide(state(phase="done", total=179, done=179), [SESSION], balance=70, now=NOW) == "stop"


def test_extend_uploads_only_driver_and_keeps_the_live_vm(monkeypatch, tmp_path):
    import subprocess
    from experiments.openevolve.colab import watchdog

    states, commands, cells = [], [], []
    monkeypatch.setattr(watchdog, "CACHE", tmp_path)
    monkeypatch.setattr(watchdog, "write_state", lambda s: states.append(dict(s)))

    def colab(*args, **kwargs):
        commands.append(args)
        return subprocess.CompletedProcess(args, 0, stdout="ok", stderr="")

    monkeypatch.setattr(watchdog, "colab", colab)
    monkeypatch.setattr(watchdog, "start_attached_client", lambda cell: cells.append(cell.read_text()))
    monkeypatch.setattr(watchdog, "wait_running", lambda: True)
    watchdog.extend_vm(state(phase="done", total=149, done=149))
    assert [c[0] for c in commands] == ["upload"]  # no stop/new/restart-kernel
    assert commands[0][-1] == "/root/oe/experiments/openevolve/colab/study.py"
    assert len(cells) == 1
    assert states[-1]["phase"] == "starting" and states[-1]["total"] == 179
    assert states[-1]["done"] == 149


def test_extension_upload_failure_does_not_launch_or_change_state(monkeypatch, tmp_path):
    import subprocess
    import pytest
    from experiments.openevolve.colab import watchdog

    monkeypatch.setattr(watchdog, "CACHE", tmp_path)
    monkeypatch.setattr(watchdog, "colab", lambda *a, **kw: subprocess.CompletedProcess(a, 1, stdout="", stderr="failed"))
    states, launched = [], []
    monkeypatch.setattr(watchdog, "write_state", states.append)
    monkeypatch.setattr(watchdog, "start_attached_client", launched.append)
    with pytest.raises(RuntimeError, match="upload"):
        watchdog.extend_vm(state(phase="done", total=149, done=149))
    assert states == launched == []


def test_extension_refuses_in_progress_or_incomplete_original_plan(monkeypatch):
    import pytest
    from experiments.openevolve.colab import watchdog

    for invalid in (state(total=149, done=139), state(phase="done", total=149, done=148),
                    state(phase="done", total=149, done=149, blocked="argon_quota")):
        with pytest.raises(RuntimeError, match="completed original plan"):
            watchdog.extend_vm(invalid)


def test_extension_start_failure_stops_client_and_blocks_recovery(monkeypatch, tmp_path):
    import subprocess
    import pytest
    from experiments.openevolve.colab import watchdog

    states, stopped = [], []
    monkeypatch.setattr(watchdog, "CACHE", tmp_path)
    monkeypatch.setattr(watchdog, "colab", lambda *a, **kw: subprocess.CompletedProcess(a, 0, stdout="ok", stderr=""))
    monkeypatch.setattr(watchdog, "write_state", lambda s: states.append(dict(s)))
    monkeypatch.setattr(watchdog, "start_attached_client", lambda cell: None)
    monkeypatch.setattr(watchdog, "wait_running", lambda: False)
    monkeypatch.setattr(watchdog, "_run", lambda cmd, **kw: stopped.append(cmd))
    with pytest.raises(RuntimeError, match="did not start"):
        watchdog.extend_vm(state(phase="done", total=149, done=149))
    assert stopped == [["systemctl", "--user", "stop", "oe-colab-client.service"]]
    assert states[-1]["blocked"] == "extension_start_failed"
    assert decide(states[-1], [SESSION], balance=70, now=NOW) == "stop"


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



def test_lc_reserve_holds_vm_while_lc_waits_for_agy():
    # Deviation 2026-11-04: the last 20 units are kept for MoE LC (Argon quota refreshes later).
    # 399.47 - 39.0 = 360.47 spent >= 380 - 20
    assert decide(state(waiting="agy_login"), [SESSION], balance=39.0, now=NOW, lc_reserve=20.0) == "hold_lc"
    assert decide(state(waiting="agy_login"), [], balance=39.0, now=NOW, lc_reserve=20.0) == "none"
    assert decide(state(waiting="agy_login"), [SESSION], balance=41.0, now=NOW, lc_reserve=20.0) == "none"  # 358.47 spent
    # LC running or the owner's login window: the reserve is being used as intended
    assert decide(state(waiting=None), [SESSION], balance=39.0, now=NOW, lc_reserve=20.0) == "none"
    assert decide(state(waiting="login_window"), [SESSION], balance=39.0, now=NOW, lc_reserve=20.0) == "none"
    assert decide(state(phase="blocked", blocked="lc_reserve"), [], balance=39.0, now=NOW, lc_reserve=20.0) == "none"


def test_no_lc_reserve_after_colab_lc_frozen():
    # Deviation 2026-11-10: Gemma-only runs may continue to the cap while LC waits for agy.
    assert decide(state(waiting="agy_login"), [SESSION], balance=39.0, now=NOW) == "none"
    assert decide(state(waiting="agy_login"), [SESSION], balance=19.0, now=NOW) == "stop_capped"
