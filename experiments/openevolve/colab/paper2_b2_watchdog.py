"""Isolated, laptop-independent watchdog for the Paper II B2 estimate cohort.

The default invocation is read-only. A live invocation additionally requires a
frozen protocol, a pinned source commit, and a passing no-call rehearsal record.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import tarfile
import threading
import time
from pathlib import Path
from typing import Any

from experiments.openevolve.colab.b2_protocol import load_protocol

PROJECT = "gen-lang-client-0903767759"
ZONE = "us-west4-a"
INSTANCE = "paper2-colab-watchdog"
SESSION = "paper2-b2-estimation"
BUCKET = "oe-colab-scale-0903767759"
PREFIX = "paper2-v2/b2-estimation-2026-11-20"
REPO_URL = "https://github.com/cyberdid/cloud-edge-ai-orchestrator.git"
ROOT = Path("/opt/paper2-b2")
REPO = ROOT / "repo"
ARCHIVE = ROOT / "oe.tar.gz"
KEY = Path("/root/.oe/sa-key.json")
PROTOCOL = Path(__file__).resolve().parents[1] / "configs/b2_estimation_2026-11-20.json"
CONTROL = ROOT / "control.json"
REHEARSAL = ROOT / "no-call-rehearsal.json"
CLI = "/opt/oe-watchdog/bin/colab"
RUN_TIMEOUT_S = 6 * 60 * 60


def parse_usage_balance(output: str) -> float:
    match = re.search(r"Current balance:\s*([\d.]+)", output)
    if not match:
        raise ValueError("Colab CLI usage output has no parseable balance")
    return float(match.group(1))


def can_continue(start: float, current: float, cap: float, reserve: float) -> bool:
    """Fail closed at either the account-wide draw cap or reserve floor."""
    return (start - current) < cap and current > reserve


def build_dry_run_plan(protocol: dict[str, Any], source_commit: str) -> dict[str, Any]:
    if not re.fullmatch(r"[0-9a-f]{40}", source_commit):
        raise ValueError("source_commit must be a full 40-character Git SHA")
    return {
        "mode": "dry_run_no_model_calls", "model_calls": 0,
        "protocol_id": protocol["protocol_id"], "protocol_status": protocol["status"],
        "source_commit": source_commit, "colab_session": protocol["colab_session"],
        "gcs_bucket": protocol["bucket"], "gcs_prefix": protocol["gcs_prefix"],
        "max_account_draw_cu": protocol["max_account_draw_cu"],
        "minimum_balance_cu": protocol["minimum_balance_cu"],
        "assigned_calls": protocol["max_total_calls"],
    }


def reconcile_state(state: dict[str, Any], evidence: dict[str, dict[str, Any]],
                    protocol: dict[str, Any]) -> dict[str, Any] | None:
    """Validate all twelve run manifests and exact call ledgers before reconciliation."""
    expected_ids = {f"{pair['pair_id']}:{version}" for pair in protocol["pairs"]
                    for version in ("v1_matched", "v2")}
    if set(evidence) != expected_ids:
        return None
    if (state.get("protocol_id") != protocol["protocol_id"] or
            state.get("protocol_sha256") is None or state.get("source_commit") is None or
            state.get("model_digest") != protocol["model_digest"]):
        return None
    by_id = {f"{pair['pair_id']}:{version}": (pair, version)
             for pair in protocol["pairs"] for version in ("v1_matched", "v2")}
    for run_id, (pair, version) in by_id.items():
        record = evidence[run_id]
        manifest, rows = record.get("manifest", {}), record.get("calls", [])
        expected = {
            "status": "complete", "protocol_id": protocol["protocol_id"],
            "protocol_sha256": state["protocol_sha256"], "source_commit": state["source_commit"],
            "model_digest": protocol["model_digest"], "task": "knn", "arm": "L",
            "seed": pair["seed"], "evaluator_version": version,
        }
        if any(manifest.get(key) != value for key, value in expected.items()):
            return None
        if [row.get("call_index") for row in rows] != list(range(protocol["calls_per_arm"])):
            return None
        if any(row.get("model") != "gemma" or row.get("status") not in {"ok", "truncated"}
               for row in rows):
            return None
        selection_sha = manifest.get("selection", {}).get("code_sha256")
        if not isinstance(selection_sha, str) or not re.fullmatch(r"[0-9a-f]{64}", selection_sha):
            return None
    return {**state, "phase": "complete", "completed_run_ids": sorted(expected_ids),
            "completed_runs": len(expected_ids), "reconciled_unix": time.time(),
            "reconciliation": "all twelve manifests and exact call ledgers verified"}


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temp.replace(path)
    path.chmod(0o600)


def _run(args: list[str], *, timeout: int = 600, check: bool = True) -> subprocess.CompletedProcess:
    result = subprocess.run(args, capture_output=True, text=True, timeout=timeout,
                            stdin=subprocess.DEVNULL)
    if check and result.returncode:
        raise RuntimeError(f"command failed: {Path(args[0]).name} {args[1]} (exit {result.returncode})")
    return result


def _usage() -> float:
    return parse_usage_balance(_run([CLI, "usage"]).stdout)


def _gcs_client():
    from google.cloud import storage
    return storage.Client()


def _read_gcs_state(protocol: dict[str, Any]) -> dict[str, Any] | None:
    blob = _gcs_client().bucket(protocol["bucket"]).blob(f"{protocol['gcs_prefix']}/state.json")
    if not blob.exists():
        return None
    return json.loads(blob.download_as_text())


def _reconcile_gcs(protocol: dict[str, Any], state: dict[str, Any]) -> dict[str, Any] | None:
    bucket = _gcs_client().bucket(protocol["bucket"])
    evidence: dict[str, dict[str, Any]] = {}
    try:
        for pair in protocol["pairs"]:
            for version in ("v1_matched", "v2"):
                run_id = f"{pair['pair_id']}:{version}"
                prefix = f"{protocol['gcs_prefix']}/runs/{pair['pair_id']}/{version}"
                manifest = json.loads(bucket.blob(f"{prefix}/manifest.json").download_as_text())
                calls_text = bucket.blob(f"{prefix}/calls.jsonl").download_as_text()
                calls = [json.loads(line) for line in calls_text.splitlines() if line.strip()]
                evidence[run_id] = {"manifest": manifest, "calls": calls}
    except Exception as exc:
        if exc.__class__.__name__ in {"NotFound", "NoSuchKey"}:
            return None
        raise
    reconciled = reconcile_state(state, evidence, protocol)
    if reconciled is None:
        return None
    blob = bucket.blob(f"{protocol['gcs_prefix']}/state.json")
    blob.upload_from_string(json.dumps(reconciled, indent=2, sort_keys=True) + "\n",
                            content_type="application/json")
    return reconciled


def _session_exists() -> bool:
    result = _run([CLI, "status", "-s", SESSION], check=False)
    return result.returncode == 0 and "not found" not in result.stdout.lower()


def _session_busy() -> bool:
    result = _run([CLI, "status", "-s", SESSION], check=False)
    return result.returncode == 0 and "busy" in result.stdout.lower()


def _stop_session() -> None:
    # This function is intentionally parameter-free: it can only stop the B2 session.
    if _session_exists():
        _run([CLI, "stop", "-s", SESSION], timeout=600, check=False)


def _ensure_repo(commit: str) -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    if not (REPO / ".git").exists():
        _run(["git", "clone", "--filter=blob:none", REPO_URL, str(REPO)], timeout=1800)
    _run(["git", "-C", str(REPO), "fetch", "origin", "main", "--quiet"], timeout=1800)
    _run(["git", "-C", str(REPO), "checkout", "--detach", commit, "--quiet"], timeout=300)
    if _run(["git", "-C", str(REPO), "rev-parse", "HEAD"]).stdout.strip() != commit:
        raise RuntimeError("B2 source checkout does not match the pinned Git SHA")
    bundle = subprocess.run(["git", "-C", str(REPO), "archive", "--format=tar", commit],
                            capture_output=True, check=True)
    import io
    with tarfile.open(ARCHIVE, "w:gz") as target:
        with tarfile.open(fileobj=io.BytesIO(bundle.stdout), mode="r:") as source:
            for member in source.getmembers():
                target.addfile(member, source.extractfile(member) if member.isfile() else None)


def _prepare_session() -> None:
    if not _session_exists():
        _run([CLI, "new", "-s", SESSION, "--gpu", "L4"], timeout=900)
    _run([CLI, "upload", "-s", SESSION, str(ARCHIVE), "/content/oe.tar.gz"], timeout=600)
    _run([CLI, "upload", "-s", SESSION, str(KEY), "/content/sa-key.json"], timeout=300)
    _run([CLI, "upload", "-s", SESSION,
          str(REPO / "experiments/openevolve/colab/paper2_b2_entrypoint.py"),
          "/content/paper2_b2_entrypoint.py"], timeout=300)
    _run([CLI, "upload", "-s", SESSION,
          str(REPO / "experiments/openevolve/colab/paper2_b2_rehearsal.py"),
          "/content/paper2_b2_rehearsal.py"], timeout=300)


def _require_rehearsal(protocol_sha256: str, source_commit: str) -> None:
    try:
        evidence = json.loads(REHEARSAL.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("passing no-call durability rehearsal is required before model calls") from exc
    required = {"status": "passed", "protocol_sha256": protocol_sha256,
                "source_commit": source_commit, "colab_session": SESSION,
                "gcs_prefix": PREFIX, "model_calls": 0,
                "resume_after_watchdog_restart": True, "terminal_state_roundtrip": True}
    if any(evidence.get(key) != value for key, value in required.items()):
        raise RuntimeError("no-call rehearsal evidence does not match this B2 protocol/source")


def _read_control(protocol: dict[str, Any], protocol_sha256: str,
                  source_commit: str) -> dict[str, Any]:
    if CONTROL.exists():
        control = json.loads(CONTROL.read_text())
        expected = {"protocol_id": protocol["protocol_id"], "protocol_sha256": protocol_sha256,
                    "source_commit": source_commit, "session": SESSION, "prefix": PREFIX}
        if any(control.get(key) != value for key, value in expected.items()):
            raise RuntimeError("saved B2 watchdog control record does not match this frozen run")
        return control
    balance = _usage()
    if balance <= protocol["minimum_balance_cu"]:
        raise RuntimeError("Colab balance is at or below the required 100-CU reserve")
    control = {"protocol_id": protocol["protocol_id"], "protocol_sha256": protocol_sha256,
               "source_commit": source_commit, "session": SESSION, "prefix": PREFIX,
               "start_balance_cu": balance, "created_unix": time.time(),
               "max_account_draw_cu": protocol["max_account_draw_cu"],
               "minimum_balance_cu": protocol["minimum_balance_cu"]}
    _atomic_json(CONTROL, control)
    return control


def _budget_monitor(protocol: dict[str, Any], control: dict[str, Any],
                    stop_event: threading.Event, limit_event: threading.Event) -> None:
    failures = 0
    while not stop_event.wait(30):
        try:
            balance = _usage()
            failures = 0
        except Exception:
            failures += 1
            if failures >= 2:
                limit_event.set()
                _stop_session()
                return
            continue
        if not can_continue(control["start_balance_cu"], balance,
                            protocol["max_account_draw_cu"],
                            protocol["minimum_balance_cu"]):
            limit_event.set()
            _stop_session()
            return


def _run_live(protocol: dict[str, Any], protocol_sha256: str, source_commit: str) -> None:
    _require_rehearsal(protocol_sha256, source_commit)
    control = _read_control(protocol, protocol_sha256, source_commit)
    current = _usage()
    if not can_continue(control["start_balance_cu"], current,
                        protocol["max_account_draw_cu"], protocol["minimum_balance_cu"]):
        _stop_session()
        print("B2 budget guard reached; durable GCS checkpoints are retained")
        return
    state = _read_gcs_state(protocol)
    if state and state.get("phase") == "complete":
        _stop_session()
        print("B2 cohort already complete; no calls started")
        return
    if state and state.get("phase") in {"error", "budget_stopped"}:
        _stop_session()
        print(f"B2 cohort is terminal ({state['phase']}); preserving artifacts for review")
        return
    if state and state.get("phase") in {"running", "interrupted"}:
        state = _reconcile_gcs(protocol, state) or state
        if state.get("phase") == "complete":
            _stop_session()
            print("B2 completion reconciled from immutable run artifacts")
            return
    if _session_exists() and _session_busy():
        print("B2 Colab execution is already active; monitoring it without queuing a duplicate")
        while _session_busy():
            try:
                current = _usage()
            except Exception:
                _stop_session()
                print("Colab usage could not be read twice; stopped only B2 session")
                return
            if not can_continue(control["start_balance_cu"], current,
                                protocol["max_account_draw_cu"],
                                protocol["minimum_balance_cu"]):
                _stop_session()
                print("B2 account guard stopped its active session; checkpoints retained")
                return
            state = _read_gcs_state(protocol)
            if state and state.get("phase") == "complete":
                _stop_session()
                print("B2 cohort completed while watchdog was recovering")
                return
            time.sleep(30)
        state = _read_gcs_state(protocol)
        if state and state.get("phase") in {"running", "interrupted"}:
            state = _reconcile_gcs(protocol, state) or state
            if state.get("phase") == "complete":
                _stop_session()
                return
    _ensure_repo(source_commit)
    _prepare_session()
    stop_monitor, limit_reached = threading.Event(), threading.Event()
    monitor = threading.Thread(target=_budget_monitor,
                               args=(protocol, control, stop_monitor, limit_reached), daemon=True)
    monitor.start()
    try:
        result = _run([CLI, "exec", "-s", SESSION, "-f",
                       str(REPO / "experiments/openevolve/colab/paper2_b2_entrypoint.py"),
                       "--env", f"PAPER2_GIT_SHA={source_commit}", "--timeout", str(RUN_TIMEOUT_S)],
                      timeout=RUN_TIMEOUT_S + 120, check=False)
    finally:
        stop_monitor.set()
        monitor.join(timeout=5)
    if limit_reached.is_set():
        print("B2 budget monitor stopped its Colab session; checkpoints retained")
        return
    state = _read_gcs_state(protocol)
    if state and state.get("phase") == "complete":
        _stop_session()
        print("B2 cohort complete")
        return
    if result.returncode:
        raise RuntimeError("B2 Colab command exited before durable complete state")
    raise RuntimeError("B2 Colab returned without a complete or budget-stopped GCS state")


def _run_rehearsal(protocol: dict[str, Any], protocol_sha256: str, source_commit: str) -> None:
    if REHEARSAL.exists():
        try:
            saved = json.loads(REHEARSAL.read_text())
        except json.JSONDecodeError as exc:
            raise RuntimeError("existing rehearsal evidence is malformed; inspect it before retry") from exc
        if (saved.get("status") == "passed" and saved.get("protocol_sha256") == protocol_sha256
                and saved.get("source_commit") == source_commit):
            print("matching no-call rehearsal already passed")
            return
        raise RuntimeError("existing rehearsal evidence belongs to a different B2 source/protocol")
    start_balance = _usage()
    if start_balance <= protocol["minimum_balance_cu"]:
        raise RuntimeError("rehearsal refused because it would cross the 100-CU reserve")
    if _session_exists():
        if _session_busy():
            raise RuntimeError("B2 session is busy; refusing to interfere with an active run")
        _stop_session()
    _ensure_repo(source_commit)
    run_id = f"rehearsal-{int(time.time())}"

    def execute(stage: str) -> dict[str, Any]:
        result = _run([CLI, "exec", "-s", SESSION, "-f",
                       str(REPO / "experiments/openevolve/colab/paper2_b2_rehearsal.py"),
                       "--env", f"B2_REHEARSAL_STAGE={stage}",
                       "--env", f"B2_PROTOCOL_SHA256={protocol_sha256}",
                       "--env", f"PAPER2_GIT_SHA={source_commit}",
                       "--env", f"B2_REHEARSAL_RUN_ID={run_id}", "--timeout", "900"],
                      timeout=1000, check=False)
        marker = "B2_REHEARSAL_RESULT="
        for line in reversed(result.stdout.splitlines()):
            if marker in line:
                return json.loads(line.split(marker, 1)[1])
        raise RuntimeError(f"B2 no-call {stage} stage failed (exit {result.returncode})")

    try:
        _prepare_session()
        uploaded = execute("upload")
        if uploaded.get("phase") != "uploaded" or uploaded.get("model_calls") != 0:
            raise RuntimeError("no-call upload stage returned unexpected evidence")
        _stop_session()
        if _session_exists():
            raise RuntimeError("Colab runtime did not stop after the checkpoint stage")
        _prepare_session()  # New named runtime; restore only from the B2 rehearsal GCS prefix.
        restored = execute("restore")
        end_balance = _usage()
        if (restored.get("status") != "passed" or restored.get("protocol_sha256") != protocol_sha256
                or restored.get("source_commit") != source_commit or restored.get("model_calls") != 0
                or restored.get("terminal_state_roundtrip") is not True):
            raise RuntimeError("fresh-runtime restore stage failed verification")
        if (end_balance <= protocol["minimum_balance_cu"] or
                start_balance - end_balance >= protocol["max_account_draw_cu"]):
            raise RuntimeError("no-call rehearsal reached the account draw cap or 100-CU reserve")
        evidence = {**restored, "balance_before_cu": start_balance,
                    "balance_after_cu": end_balance, "account_draw_cu": start_balance - end_balance,
                    "runtime_recreated": True}
        _atomic_json(REHEARSAL, evidence)
        print(json.dumps({"mode": "no_call_rehearsal", "status": "passed",
                          "balance_before_cu": start_balance, "balance_after_cu": end_balance,
                          "model_calls": 0, "gcs_prefix": protocol["gcs_prefix"]}, sort_keys=True))
    finally:
        _stop_session()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    parser.add_argument("--source-commit", default=os.environ.get("PAPER2_GIT_SHA", ""))
    parser.add_argument("--dry-run", action="store_true", help="print plan without cloud or model calls")
    parser.add_argument("--rehearse", action="store_true", help="GCS persistence check only; zero model calls")
    parser.add_argument("--live", action="store_true", help="run only after frozen-protocol rehearsal gate")
    args = parser.parse_args()
    if sum((args.dry_run, args.rehearse, args.live)) != 1:
        raise SystemExit("choose exactly one of --dry-run, --rehearse, or --live")
    protocol = load_protocol(args.protocol)
    protocol_sha256 = hashlib.sha256(args.protocol.read_bytes()).hexdigest()
    plan = build_dry_run_plan(protocol, args.source_commit)
    if args.dry_run:
        print(json.dumps(plan, sort_keys=True))
        return
    if args.rehearse:
        _run_rehearsal(protocol, protocol_sha256, args.source_commit)
        return
    if protocol["status"] != "frozen":
        raise SystemExit("refusing live B2: protocol is still draft")
    _run_live(protocol, protocol_sha256, args.source_commit)


if __name__ == "__main__":
    main()
