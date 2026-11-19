"""e2-micro supervisor for the laptop-independent Paper II Colab pilot."""
from __future__ import annotations

import json
import os
import re
import subprocess
import tarfile
import threading
import time
from pathlib import Path

PROJECT = "gen-lang-client-0903767759"
ZONE = "us-west4-a"
INSTANCE = "paper2-colab-watchdog"
SESSION = "paper2-v2-pilot"
BUCKET = "oe-colab-scale-0903767759"
PREFIX = "paper2-v2/b1"
REPO_URL = "https://github.com/cyberdid/antigravity-codex-orchestrator.git"
ROOT = Path("/opt/paper2")
REPO = ROOT / "repo"
ARCHIVE = ROOT / "oe.tar.gz"
KEY = Path("/root/.oe/sa-key.json")
CONFIG = ROOT / "watchdog-control.json"
CLI = "/opt/oe-watchdog/bin/colab"
MAX_COMPUTE_UNITS = 15.0
MIN_BALANCE_UNITS = 100.0
RUN_TIMEOUT_S = 6 * 60 * 60


def parse_usage_balance(output: str) -> float:
    match = re.search(r"Current balance:\s*([\d.]+)", output)
    if not match:
        raise ValueError("Colab CLI usage output has no parseable balance")
    return float(match.group(1))


def budget_exhausted(start: float, current: float, cap: float) -> bool:
    return start - current >= cap


def _run(args: list[str], *, timeout: int = 600, check: bool = True) -> subprocess.CompletedProcess:
    result = subprocess.run(args, capture_output=True, text=True, timeout=timeout, stdin=subprocess.DEVNULL)
    if check and result.returncode:
        # Avoid logging command output: CLI errors can contain short-lived auth tokens.
        raise RuntimeError(f"command failed: {Path(args[0]).name} {args[1]} (exit {result.returncode})")
    return result


def _gcs_state() -> dict | None:
    from google.cloud import storage

    blob = storage.Client().bucket(BUCKET).blob(f"{PREFIX}/state.json")
    if not blob.exists():
        return None
    return json.loads(blob.download_as_text())


def _usage() -> float:
    result = _run([CLI, "usage"])
    return parse_usage_balance(result.stdout)


def _ensure_repo(commit: str) -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    if not (REPO / ".git").exists():
        _run(["git", "clone", "--filter=blob:none", REPO_URL, str(REPO)], timeout=1800)
    _run(["git", "-C", str(REPO), "fetch", "origin", "main", "--quiet"], timeout=1800)
    _run(["git", "-C", str(REPO), "checkout", "--detach", commit, "--quiet"], timeout=300)
    actual = _run(["git", "-C", str(REPO), "rev-parse", "HEAD"]).stdout.strip()
    if actual != commit:
        raise RuntimeError("the Colab source checkout did not match the pinned Git SHA")
    with tarfile.open(ARCHIVE, "w:gz") as archive:
        # Build from Git's tracked tree, never from untracked local benchmark outputs.
        result = subprocess.run(["git", "-C", str(REPO), "archive", "--format=tar", commit],
                                capture_output=True, check=True)
        import io
        with tarfile.open(fileobj=io.BytesIO(result.stdout), mode="r:") as source:
            for member in source.getmembers():
                archive.addfile(member, source.extractfile(member) if member.isfile() else None)


def _write_local_control(commit: str, start_balance: float) -> None:
    CONFIG.write_text(json.dumps({"git_sha": commit, "start_balance": start_balance,
                                  "max_compute_units": MAX_COMPUTE_UNITS}, indent=2))
    CONFIG.chmod(0o600)


def _read_local_control() -> dict | None:
    try:
        return json.loads(CONFIG.read_text())
    except FileNotFoundError:
        return None


def _session_exists() -> bool:
    status = _run([CLI, "status", "-s", SESSION], check=False)
    return status.returncode == 0 and "not found" not in status.stdout.lower()


def _session_busy() -> bool:
    status = _run([CLI, "status", "-s", SESSION], check=False)
    return status.returncode == 0 and "busy" in status.stdout.lower()


def _ensure_session() -> None:
    if _session_exists():
        return
    _run([CLI, "new", "-s", SESSION, "--gpu", "L4"], timeout=900)


def _prepare_session(commit: str) -> None:
    _run([CLI, "upload", "-s", SESSION, str(ARCHIVE), "/content/oe.tar.gz"], timeout=600)
    _run([CLI, "upload", "-s", SESSION, str(KEY), "/content/sa-key.json"], timeout=300)
    _run([CLI, "upload", "-s", SESSION, str(REPO / "experiments/openevolve/colab/paper2_pilot.py"),
          "/content/paper2_pilot.py"], timeout=300)
    # Clear cached Python modules so a resumed kernel cannot import an earlier Git revision.
    _run([CLI, "restart-kernel", "-s", SESSION], timeout=300, check=False)


def _stop_session() -> None:
    if _session_exists():
        _run([CLI, "stop", "-s", SESSION], timeout=600, check=False)


def _budget_monitor(start_balance: float, cap: float, stop_event: threading.Event,
                    limit_event: threading.Event) -> None:
    errors = 0
    while not stop_event.wait(60):
        try:
            balance = _usage()
            errors = 0
        except Exception:
            errors += 1
            if errors >= 2:
                limit_event.set()
                _stop_session()
                return
            continue
        if budget_exhausted(start_balance, balance, cap) or balance <= MIN_BALANCE_UNITS:
            limit_event.set()
            _stop_session()
            return


def main() -> None:
    local = _read_local_control()
    state = _gcs_state()
    if state and state.get("phase") == "complete":
        _stop_session()
        return

    usage = _usage()
    if local is None:
        start_balance = usage
        commit = os.environ.get("PAPER2_GIT_SHA", "")
        if not re.fullmatch(r"[0-9a-f]{40}", commit):
            raise RuntimeError("PAPER2_GIT_SHA must be a full 40-character commit SHA")
        local = {"git_sha": commit, "start_balance": start_balance,
                 "max_compute_units": MAX_COMPUTE_UNITS}
        _write_local_control(commit, start_balance)
    commit = local["git_sha"]
    if budget_exhausted(local["start_balance"], usage, local["max_compute_units"]):
        _stop_session()
        print("preregistered Colab compute-unit cap reached; GCS checkpoints preserved")
        return
    if usage <= MIN_BALANCE_UNITS:
        _stop_session()
        print("stopping with at least 100 compute units held in reserve")
        return

    _ensure_repo(commit)
    _ensure_session()
    _prepare_session(commit)

    while True:
        state = _gcs_state()
        if state and state.get("phase") == "complete":
            _stop_session()
            return
        if state and state.get("phase") == "running" and _session_exists() and _session_busy():
            time.sleep(60)
            continue
        usage = _usage()
        if budget_exhausted(local["start_balance"], usage, local["max_compute_units"]) or usage <= MIN_BALANCE_UNITS:
            _stop_session()
            print("Colab budget limit reached; GCS checkpoints retained")
            return
        stop_monitor = threading.Event()
        limit_reached = threading.Event()
        monitor = threading.Thread(target=_budget_monitor,
                                   args=(local["start_balance"], local["max_compute_units"],
                                         stop_monitor, limit_reached), daemon=True)
        monitor.start()
        try:
            result = _run([CLI, "exec", "-s", SESSION, "-f",
                           str(REPO / "experiments/openevolve/colab/paper2_pilot.py"),
                           "--env", f"PAPER2_GIT_SHA={commit}", "--timeout", str(RUN_TIMEOUT_S)],
                          timeout=RUN_TIMEOUT_S + 120, check=False)
        finally:
            stop_monitor.set()
            monitor.join(timeout=5)
        if limit_reached.is_set():
            print("compute-unit/balance guard stopped Colab; GCS checkpoints retained")
            return
        state = _gcs_state()
        if state and state.get("phase") == "complete":
            _stop_session()
            return
        if result.returncode == 0:
            raise RuntimeError("Colab runner returned without a complete GCS state")

        # If a response was lost while the kernel continued running, do not queue a
        # duplicate study cell. Wait for its durable heartbeat before recovery.
        updated = float(state.get("updated_unix", 0)) if state else 0
        if time.time() - updated < 180:
            time.sleep(180)
            continue
        if _session_exists():
            _run([CLI, "restart-kernel", "-s", SESSION], timeout=300, check=False)
        else:
            _ensure_session()
            _prepare_session(commit)


if __name__ == "__main__":
    main()
