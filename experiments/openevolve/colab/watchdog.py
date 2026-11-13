"""Watchdog for the Colab scale study (systemd user timer, every minute).

Reads the study state from the bucket and the live Colab sessions, enforces the
compute-unit cap, stops finished/blocked/stale VMs and starts a fresh one when work
remains. Only the session named ``SESSION`` is ever touched.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Literal

from experiments.openevolve.colab.study import LC_EXTENSION_ID, study_plan

BUCKET = "gs://oe-colab-scale-0903767759"
SESSION = "oe-scale"
START_BALANCE = 399.47
CAP_UNITS = 380.0  # raised from 330 on 2026-11-02, see docs/colab_scale/DEVIATIONS.md
# The last units are kept for MoE LC while it waits for the Argon quota (deviation 2026-11-04).
# Deviation 2026-11-10: Colab LC is frozen (no agy on Colab any more), so no units are
# held back for it; the Gemma-only extension L controls may use the cap. Was 20.0.
LC_RESERVE = 0.0
STALE_S = 1800
FIRST_MODEL = "gemma4:26b"
REPO = Path(__file__).resolve().parents[3]
KEY = Path.home() / ".config/oe-colab/sa-key.json"
CACHE = Path.home() / ".cache/oe-colab"
ARCHIVE_PATHS = ("pyproject.toml", "uv.lock", "README.md", "LICENSE", "NOTICE",
                 "antigravity_hybrid_orchestrator", "experiments")
CELL = """
import subprocess
print(subprocess.run("cd /root/oe && OE_COLAB_RUNS=/root/oe-runs .venv/bin/python -m experiments.openevolve.colab.study",
                     shell=True, capture_output=True, text=True).stdout[-2000:])
"""
CLIENT_UNIT = "oe-colab-client.service"
CELL_TIMEOUT_S = 48 * 60 * 60

Action = Literal["none", "start", "stop", "stop_capped", "hold_lc", "extend"]


def decide(state: dict | None, sessions: list[str], balance: float, now: float,
           start_balance: float = START_BALANCE, cap: float = CAP_UNITS,
           lc_reserve: float | None = None) -> Action:
    lc_reserve = LC_RESERVE if lc_reserve is None else lc_reserve
    live = SESSION in sessions
    if start_balance - balance >= cap:
        return "stop_capped" if live else "none"
    if state and state.get("blocked"):
        return "stop" if live else "none"
    # The live driver keeps its original fixed plan. Once it has persisted all 149
    # completions, hand off to the extended driver on this VM, retaining agy login.
    # Unknown sessions and the spending cap always retain their protections.
    if state and state.get("phase") == "done" and state.get("total") == 149 and state.get("done") == 149:
        if "?" in sessions:
            return "none"
        return "extend" if live else "start"
    if state and state.get("phase") == "done":
        return "stop" if live else "none"
    if state and state.get("waiting") == "agy_login" and start_balance - balance >= cap - lc_reserve:
        return "hold_lc" if live else "none"
    # Unknown assignments cannot safely be stopped or replaced by name.
    if "?" in sessions:
        return "none"
    # Session names are host-local: another host's live VM appears as "?". Protect
    # it and an in-progress bootstrap, but do not let an old "running" heartbeat
    # delay recovery after the server confirms that the VM is gone.
    if (state and now - state.get("updated", 0) < STALE_S
            and (state.get("phase") == "starting" or live or "?" in sessions)):
        return "none"
    return "stop" if live else "start"


def _run(args: list[str], timeout: float = 600, **kw) -> subprocess.CompletedProcess:
    return subprocess.run(args, capture_output=True, text=True, timeout=timeout, stdin=subprocess.DEVNULL, **kw)


def read_state() -> dict | None:
    out = _run(["gcloud", "storage", "cat", f"{BUCKET}/state.json"], timeout=120)
    if out.returncode != 0 or not out.stdout.strip():
        raise RuntimeError("cannot read study state; refusing automatic action")
    return json.loads(out.stdout)


def write_state(state: dict) -> None:
    tmp = CACHE / "state.json"
    tmp.write_text(json.dumps(state))
    out = _run(["gcloud", "storage", "cp", str(tmp), f"{BUCKET}/state.json"], timeout=120)
    if out.returncode != 0:
        raise RuntimeError("cannot persist study state; refusing automatic action")


def parse_sessions(text: str) -> list[str]:
    """Session names from ``colab sessions``; ``[colab] ...`` lines are CLI messages."""
    return [name for name in re.findall(r"^\[([^\]]+)\]", text, flags=re.M) if name != "colab"]


def sessions() -> list[str]:
    out = _run(["colab", "sessions"], timeout=120)
    if out.returncode:
        raise RuntimeError("cannot list Colab sessions; refusing automatic action")
    names = parse_sessions(out.stdout)
    if not names and "No active sessions found on server." not in out.stdout:
        raise RuntimeError("unrecognised Colab session response; refusing automatic action")
    return names


def usage() -> tuple[float, int]:
    out = _run(["colab", "usage"], timeout=120)
    balance = re.search(r"Current balance:\s*([\d.]+)", out.stdout)
    assignments = re.search(r"Active assignments:\s*(\d+)", out.stdout)
    if out.returncode or not balance or not assignments:
        raise RuntimeError("cannot read Colab usage; refusing automatic action")
    return float(balance.group(1)), int(assignments.group(1))


def log(event: dict) -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    line = json.dumps(dict(event, ts=time.time()))
    with (CACHE / "watchdog.log").open("a") as handle:
        handle.write(line + "\n")
    _run(["gcloud", "storage", "cp", str(CACHE / "watchdog.log"), f"{BUCKET}/vm_log.jsonl"], timeout=120)


def colab(*args: str, timeout: float = 1800) -> subprocess.CompletedProcess:
    return _run(["colab", *args], timeout=timeout)


def start_attached_client(cell: Path) -> None:
    """Keep the notebook connection in a unit independent of this oneshot."""
    _run(["systemctl", "--user", "stop", CLIENT_UNIT], timeout=30)
    client = _run(["systemd-run", "--user", "--collect", "--unit", CLIENT_UNIT,
                   "--property", "Type=exec", "--property", "Restart=no",
                   "--setenv", f"PATH={os.environ.get('PATH', '')}",
                   shutil.which("colab") or "colab", "exec", "-s", SESSION, "-f", str(cell),
                   "--timeout", str(CELL_TIMEOUT_S)], timeout=30)
    if client.returncode:
        raise RuntimeError(f"cannot start attached Colab client: {client.stderr[-500:]}")


def extend_vm(state: dict) -> None:
    """Replace only the completed 149-run driver; retain the VM and its login."""
    if state.get("phase") != "done" or state.get("total") != 149 or state.get("done") != 149 or state.get("blocked"):
        raise RuntimeError("extension requires the completed original plan")
    uploaded = colab("upload", "-s", SESSION,
                     str(REPO / "experiments/openevolve/colab/study.py"),
                     "/root/oe/experiments/openevolve/colab/study.py")
    if uploaded.returncode:
        raise RuntimeError("cannot upload extended driver; leaving completed VM untouched")
    write_state(dict(state, phase="starting", blocked=None, waiting=None, total=len(study_plan()),
                     plan_revision=LC_EXTENSION_ID, updated=time.time()))
    cell = CACHE / "study_cell.py"
    cell.write_text(CELL)
    try:
        start_attached_client(cell)
        if not wait_running():
            raise RuntimeError("extended study driver did not start")
    except Exception:
        _run(["systemctl", "--user", "stop", CLIENT_UNIT], timeout=30)
        write_state(dict(state, phase="blocked", blocked="extension_start_failed", updated=time.time()))
        raise


def start_vm(state: dict | None) -> None:
    write_state(dict(state or {}, phase="starting", blocked=None, updated=time.time()))
    tarball = CACHE / "oe.tar.gz"
    _run(["git", "-C", str(REPO), "archive", "--format=tar.gz", "-o", str(tarball), "HEAD", *ARCHIVE_PATHS])
    allocated = False
    try:
        if colab("new", "-s", SESSION, "--gpu", "L4", timeout=600).returncode:
            raise RuntimeError("colab new failed")
        allocated = True
        exec_code("import os; os.makedirs('/root/.oe', mode=0o700, exist_ok=True)")
        colab("upload", "-s", SESSION, str(tarball), "/root/oe.tar.gz")
        colab("upload", "-s", SESSION, str(REPO / "experiments/openevolve/colab/bootstrap.sh"), "/root/bootstrap.sh")
        colab("upload", "-s", SESSION, str(KEY), "/root/.oe/sa-key.json")
        out = exec_code("import subprocess\nr = subprocess.run('chmod 600 /root/.oe/sa-key.json && bash /root/bootstrap.sh "
                        f"/root/oe.tar.gz {FIRST_MODEL}', shell=True, capture_output=True, text=True)\n"
                        "print(r.stdout[-1500:], r.stderr[-1500:])", timeout=3600)
        if "BOOTSTRAP OK" not in out:
            raise RuntimeError(f"bootstrap failed: {out[-800:]}")
        log({"event": "bootstrap", "detail": out.strip().splitlines()[-1]})
        cell = CACHE / "study_cell.py"
        cell.write_text(CELL)
        for attempt in range(2):
            # The cell keeps the kernel busy for the whole study; the client may go away.
            # A child of this oneshot service is killed when the watchdog exits, even
            # with setsid/nohup. Give the attached client its own systemd cgroup and a
            # timeout suitable for a long-running notebook cell (CLI default: 30 s).
            start_attached_client(cell)
            if wait_running():
                return
            colab("restart-kernel", "-s", SESSION, timeout=120)
        raise RuntimeError("study driver did not start")
    except Exception:
        _run(["systemctl", "--user", "stop", CLIENT_UNIT], timeout=30)
        if allocated:
            colab("stop", "-s", SESSION, timeout=300)
        write_state(dict(state or {}, phase="recovering", blocked=None, updated=0))
        raise


def wait_running(timeout_s: float = 600, poll_s: float = 20) -> bool:
    """The driver writes phase "running" right after restoring from the bucket."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        state = read_state()
        if state and state.get("phase") == "running":
            return True
        time.sleep(poll_s)
    return False


def exec_code(code: str, timeout: float = 600) -> str:
    """`colab exec` sometimes hangs before reaching the kernel (seen twice in the spike):
    on a timeout, restart the kernel and try once more."""
    path = CACHE / "snippet.py"
    path.write_text(code)
    try:
        out = colab("exec", "-s", SESSION, "-f", str(path),
                    "--timeout", str(max(1, timeout - 30)), timeout=timeout)
    except subprocess.TimeoutExpired:
        colab("restart-kernel", "-s", SESSION, timeout=120)
        out = colab("exec", "-s", SESSION, "-f", str(path),
                    "--timeout", str(max(1, timeout - 30)), timeout=timeout)
    return out.stdout + out.stderr


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)
    with open(CACHE / "watchdog.lock", "w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        state, live = read_state(), sessions()
        units, assignment_count = usage()
        if assignment_count != len(live):
            raise RuntimeError("Colab assignment count disagrees with session listing; refusing automatic action")
        if "?" in live and (START_BALANCE - units >= CAP_UNITS or state.get("phase") == "done" or state.get("blocked")):
            raise RuntimeError("unknown Colab session requires manual attention; cannot safely stop it by name")
        action = decide(state, live, units, time.time())
        log({"event": "tick", "action": action, "balance": units, "sessions": live,
             "phase": (state or {}).get("phase"), "blocked": (state or {}).get("blocked")})
        if args.dry_run or action == "none":
            return
        if action in ("stop", "stop_capped", "hold_lc"):
            colab("stop", "-s", SESSION, timeout=300)
            if action == "stop_capped":
                write_state(dict(state or {}, phase="capped", updated=time.time()))
            if action == "hold_lc":
                write_state(dict(state or {}, phase="blocked", blocked="lc_reserve", updated=time.time()))
            log({"event": action})
        elif action == "start":
            start_vm(state)
            log({"event": "started"})
        elif action == "extend":
            extend_vm(state)
            log({"event": "extension_started", "plan_revision": LC_EXTENSION_ID})


if __name__ == "__main__":
    main()
