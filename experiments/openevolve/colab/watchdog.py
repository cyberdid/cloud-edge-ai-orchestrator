"""Laptop watchdog for the Colab scale study (systemd user timer, every 10 minutes).

Reads the study state from the bucket and the live Colab sessions, enforces the
compute-unit cap, stops finished/blocked/stale VMs and starts a fresh one when work
remains. Only the session named ``SESSION`` is ever touched.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import re
import subprocess
import time
from pathlib import Path
from typing import Literal

BUCKET = "gs://oe-colab-scale-0903767759"
SESSION = "oe-scale"
START_BALANCE = 399.47
CAP_UNITS = 330.0
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

Action = Literal["none", "start", "stop", "stop_capped"]


def decide(state: dict | None, sessions: list[str], balance: float, now: float,
           start_balance: float = START_BALANCE, cap: float = CAP_UNITS) -> Action:
    live = SESSION in sessions
    if start_balance - balance >= cap:
        return "stop_capped" if live else "none"
    if state and (state.get("phase") == "done" or state.get("blocked")):
        return "stop" if live else "none"
    fresh = bool(state) and now - state.get("updated", 0) < STALE_S
    if state and state.get("phase") == "starting" and fresh:
        return "none"
    if not live:
        return "start"
    return "none" if fresh else "stop"


def _run(args: list[str], timeout: float = 600, **kw) -> subprocess.CompletedProcess:
    return subprocess.run(args, capture_output=True, text=True, timeout=timeout, stdin=subprocess.DEVNULL, **kw)


def read_state() -> dict | None:
    out = _run(["gcloud", "storage", "cat", f"{BUCKET}/state.json"], timeout=120)
    return json.loads(out.stdout) if out.returncode == 0 and out.stdout.strip() else None


def write_state(state: dict) -> None:
    tmp = CACHE / "state.json"
    tmp.write_text(json.dumps(state))
    _run(["gcloud", "storage", "cp", str(tmp), f"{BUCKET}/state.json"], timeout=120)


def parse_sessions(text: str) -> list[str]:
    """Session names from ``colab sessions``; ``[colab] ...`` lines are CLI messages."""
    return [name for name in re.findall(r"^\[([^\]]+)\]", text, flags=re.M) if name != "colab"]


def sessions() -> list[str]:
    return parse_sessions(_run(["colab", "sessions"], timeout=120).stdout)


def balance() -> float:
    match = re.search(r"Current balance:\s*([\d.]+)", _run(["colab", "usage"], timeout=120).stdout)
    if not match:
        raise RuntimeError("cannot read Colab balance")
    return float(match.group(1))


def log(event: dict) -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    line = json.dumps(dict(event, ts=time.time()))
    with (CACHE / "watchdog.log").open("a") as handle:
        handle.write(line + "\n")
    _run(["gcloud", "storage", "cp", str(CACHE / "watchdog.log"), f"{BUCKET}/vm_log.jsonl"], timeout=120)


def colab(*args: str, timeout: float = 1800) -> subprocess.CompletedProcess:
    return _run(["colab", *args], timeout=timeout)


def start_vm(state: dict | None) -> None:
    write_state(dict(state or {}, phase="starting", blocked=None, updated=time.time()))
    tarball = CACHE / "oe.tar.gz"
    _run(["git", "-C", str(REPO), "archive", "--format=tar.gz", "-o", str(tarball), "HEAD", *ARCHIVE_PATHS])
    if colab("new", "-s", SESSION, "--gpu", "L4", timeout=600).returncode:
        raise RuntimeError("colab new failed")
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
    # The cell keeps the kernel busy for the whole study; the client may go away.
    subprocess.Popen(["setsid", "nohup", "colab", "exec", "-s", SESSION, "-f", str(cell)],
                     stdout=open(CACHE / "cell.log", "a"), stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)


def exec_code(code: str, timeout: float = 600) -> str:
    path = CACHE / "snippet.py"
    path.write_text(code)
    out = colab("exec", "-s", SESSION, "-f", str(path), timeout=timeout)
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
        state, live, units = read_state(), sessions(), balance()
        action = decide(state, live, units, time.time())
        log({"event": "tick", "action": action, "balance": units, "sessions": live,
             "phase": (state or {}).get("phase"), "blocked": (state or {}).get("blocked")})
        if args.dry_run or action == "none":
            return
        if action in ("stop", "stop_capped"):
            colab("stop", "-s", SESSION, timeout=300)
            if action == "stop_capped":
                write_state(dict(state or {}, phase="capped", updated=time.time()))
            log({"event": action})
        elif action == "start":
            start_vm(state)
            log({"event": "started"})


if __name__ == "__main__":
    main()
