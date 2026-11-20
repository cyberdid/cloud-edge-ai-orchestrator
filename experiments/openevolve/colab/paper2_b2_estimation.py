"""Resumable Colab runner for the six-pair Paper II B2 estimation cohort."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable

from experiments.openevolve.colab.b2_protocol import load_protocol


class IncompleteRun(RuntimeError):
    """A resumable arm ended without the full frozen call ledger."""


def _jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSONL in {path}: {exc}") from exc


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temp.replace(path)


def _read_manifest(out: Path, protocol: dict[str, Any], protocol_sha256: str,
                   pair: dict[str, Any], version: str, source_commit: str) -> dict[str, Any]:
    manifest_path = out / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"missing run manifest for {pair['pair_id']} / {version}")
    try:
        manifest = json.loads(manifest_path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid run manifest for {pair['pair_id']} / {version}") from exc
    expected = {
        "task": "knn", "arm": "L", "seed": pair["seed"],
        "max_iterations": protocol["calls_per_arm"], "protocol_id": protocol["protocol_id"],
        "protocol_sha256": protocol_sha256, "model_digest": protocol["model_digest"],
        "evaluator_version": version, "git_sha": source_commit,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"run manifest mismatch for {key}: {pair['pair_id']} / {version}")
    return manifest


def _validate_run(out: Path, protocol: dict[str, Any], protocol_sha256: str,
                  pair: dict[str, Any], version: str, source_commit: str) -> dict[str, Any]:
    manifest = _read_manifest(out, protocol, protocol_sha256, pair, version, source_commit)
    if manifest.get("status") != "complete":
        raise IncompleteRun(f"run is resumable but incomplete: {pair['pair_id']} / {version}")
    calls_path = out / "calls.jsonl"
    if not calls_path.is_file():
        raise ValueError(f"missing call ledger for {pair['pair_id']} / {version}")
    rows = _jsonl(calls_path)
    if manifest.get("selection", {}).get("status") != "recorded":
        raise ValueError(f"official selected artifact is missing for {pair['pair_id']} / {version}")
    code_sha = manifest["selection"].get("code_sha256")
    if not isinstance(code_sha, str) or not re.fullmatch(r"[0-9a-f]{64}", code_sha):
        raise ValueError(f"official selected artifact SHA-256 is invalid for {pair['pair_id']} / {version}")
    indices = [row.get("call_index") for row in rows]
    if indices != list(range(protocol["calls_per_arm"])):
        raise ValueError(f"call indices are incomplete, duplicated, or out of order for {pair['pair_id']} / {version}")
    if any(row.get("model") != "gemma" for row in rows):
        raise ValueError(f"unexpected model in call ledger for {pair['pair_id']} / {version}")
    if any(row.get("status") not in {"ok", "truncated"} for row in rows):
        raise ValueError(f"unexpected call status for {pair['pair_id']} / {version}")
    return manifest


def run_cohort(protocol: dict[str, Any], protocol_sha256: str, *, llm: str, root: str | Path,
               mirror: Any, run_one: Callable[..., dict[str, Any]], source_commit: str,
               checkpoint_interval_s: float = 30.0) -> dict[str, Any]:
    """Execute or resume the exact frozen cohort; injected run_one supports no-model tests."""
    if llm not in {"real", "stub"}:
        raise ValueError("llm must be real or stub")
    if llm == "real":
        if protocol.get("status") != "frozen":
            raise ValueError("real model calls require a frozen B2 protocol")
        if not re.fullmatch(r"[0-9a-f]{40}", source_commit):
            raise ValueError("real model calls require the pinned 40-character source commit")
        credential = os.environ.get("OE_PILOT_GCS_CREDENTIALS")
        if not credential:
            raise ValueError("GCS credentials must be supplied by the watchdog, not embedded in the protocol")
    if not re.fullmatch(r"[0-9a-f]{64}", protocol_sha256):
        raise ValueError("protocol_sha256 must be a 64-character lowercase hex digest")
    if not re.fullmatch(r"[0-9a-f]{40}", source_commit):
        raise ValueError("source_commit must be a 40-character Git SHA")

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    mirror.restore()
    state_path = root / "state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    if state.get("phase") in {"complete", "error", "budget_stopped"}:
        raise ValueError(f"B2 cohort is terminal ({state['phase']}); refusing to reopen its prefix")
    if state and (state.get("protocol_id") != protocol["protocol_id"] or
                  state.get("protocol_sha256") != protocol_sha256 or
                  state.get("source_commit") != source_commit):
        raise ValueError("saved B2 state does not match the protocol or source commit")
    expected_run_ids = {f"{pair['pair_id']}:{version}" for pair in protocol["pairs"]
                        for version in pair["order"]}
    saved_completed = state.get("completed_run_ids", [])
    if (not isinstance(saved_completed, list) or
            any(not isinstance(run_id, str) for run_id in saved_completed) or
            len(saved_completed) != len(set(saved_completed)) or
            not set(saved_completed) <= expected_run_ids):
        raise ValueError("saved B2 completed-run inventory is malformed or outside the frozen cohort")

    env_values = {
        "OE_PILOT_MAX_ITER": str(protocol["calls_per_arm"]),
        "OE_PILOT_PROTOCOL_ID": protocol["protocol_id"],
        "OE_PILOT_PROTOCOL_SHA256": protocol_sha256,
        "OE_PILOT_GIT_SHA": source_commit,
        "OE_PILOT_MODEL_DIGEST": protocol["model_digest"],
        "OE_PILOT_GCS_BUCKET": protocol["bucket"],
        "OE_PILOT_GCS_PREFIX": protocol["gcs_prefix"],
        "OE_PILOT_GCS_ROOT": str(root),
        "OE_PILOT_GEMMA_MODEL": protocol["model"],
        "OE_PILOT_EXECUTOR": "subprocess",
    }
    os.environ.update(env_values)
    completed = list(state.get("completed_run_ids", []))
    lock = threading.Lock()
    terminal = threading.Event()

    def write_state(phase: str, **extra: Any) -> None:
        with lock:
            if phase == "running" and terminal.is_set():
                return
            if phase in {"complete", "error", "budget_stopped"}:
                terminal.set()
            snapshot = {
                "phase": phase, "updated_unix": time.time(), "study_type": protocol["study_type"],
                "protocol_id": protocol["protocol_id"], "protocol_sha256": protocol_sha256,
                "source_commit": source_commit, "model": protocol["model"],
                "model_digest": protocol["model_digest"], "task": protocol["task"],
                "calls_per_arm": protocol["calls_per_arm"], "max_total_calls": protocol["max_total_calls"],
                "gcs_bucket": protocol["bucket"], "gcs_prefix": protocol["gcs_prefix"],
                "completed_runs": len(completed), "completed_run_ids": list(completed), **extra,
            }
            _atomic_json(state_path, snapshot)
            mirror.sync()

    stop_checkpoint = threading.Event()
    def checkpoint_loop() -> None:
        while not stop_checkpoint.wait(checkpoint_interval_s):
            if terminal.is_set():
                return
            try:
                write_state("running", current_run=current_run[0])
            except Exception as exc:
                print(f"B2 checkpoint sync failed: {type(exc).__name__}: {exc}", flush=True)

    current_run: list[str | None] = [None]
    thread = threading.Thread(target=checkpoint_loop, daemon=True)
    thread.start()
    total_calls = len(completed) * protocol["calls_per_arm"]
    try:
        write_state("running", current_run=None)
        for pair in protocol["pairs"]:
            for version in pair["order"]:
                run_id = f"{pair['pair_id']}:{version}"
                current_run[0] = run_id
                out = root / "runs" / pair["pair_id"] / version
                manifest_path = out / "manifest.json"
                already_accounted = run_id in completed
                if already_accounted and not manifest_path.is_file():
                    raise ValueError(f"saved state claims a completed run with no manifest: {run_id}")
                if (already_accounted and manifest_path.is_file() and
                        json.loads(manifest_path.read_text()).get("status") != "complete"):
                    raise ValueError(f"saved state claims a completed run whose manifest is not complete: {run_id}")
                if not already_accounted and total_calls + protocol["calls_per_arm"] > protocol["max_total_calls"]:
                    raise ValueError("cohort would exceed the frozen total call budget")
                if manifest_path.exists() and json.loads(manifest_path.read_text()).get("status") == "complete":
                    manifest = _validate_run(out, protocol, protocol_sha256, pair, version, source_commit)
                else:
                    if manifest_path.exists():
                        _read_manifest(out, protocol, protocol_sha256, pair, version, source_commit)
                    manifest = run_one("knn", "L", pair["seed"], out, out.exists(), llm,
                                       evaluator_version=version)
                    manifest = _validate_run(out, protocol, protocol_sha256, pair, version, source_commit)
                if run_id not in completed:
                    completed.append(run_id)
                    total_calls += protocol["calls_per_arm"]
                mirror.sync()
                write_state("running", current_run=run_id, total_calls=total_calls)
        if total_calls > protocol["max_total_calls"]:
            raise ValueError("cohort exceeded the frozen total call budget")
        write_state("complete", current_run=None, total_calls=total_calls)
        return json.loads(state_path.read_text())
    except Exception as exc:
        phase = "interrupted" if not isinstance(exc, ValueError) else "error"
        try:
            write_state(phase, current_run=current_run[0], error=f"{type(exc).__name__}: {exc}"[-1200:])
        except Exception:
            pass
        raise
    finally:
        stop_checkpoint.set()
        thread.join(timeout=5)
        mirror.sync()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=Path(
        "/content/oe/experiments/openevolve/configs/b2_estimation_2026-11-20.json"))
    parser.add_argument("--dry-run", action="store_true", help="validate and print plan; never start Gemma")
    args = parser.parse_args()
    protocol = load_protocol(args.protocol)
    protocol_sha256 = hashlib.sha256(args.protocol.read_bytes()).hexdigest()
    if args.dry_run:
        print(json.dumps({"mode": "dry_run_no_model_calls", "protocol_id": protocol["protocol_id"],
                          "status": protocol["status"], "pairs": len(protocol["pairs"]),
                          "calls": protocol["max_total_calls"], "gcs_prefix": protocol["gcs_prefix"]},
                         sort_keys=True))
        return
    if protocol["status"] != "frozen":
        raise SystemExit("refusing model calls: protocol status is not frozen")

    from experiments.openevolve.colab.paper2_pilot import bootstrap, project_site_packages
    meta = bootstrap()
    if meta.get("model_digest") != protocol["model_digest"]:
        raise SystemExit("refusing model calls: Colab model digest differs from frozen protocol")
    if meta.get("gpu", "unavailable") == "unavailable":
        raise SystemExit("refusing model calls: CUDA GPU is unavailable")
    repo = Path("/content/oe")
    key = Path("/content/.oe/sa-key.json")
    root = Path("/content/paper2-v2/b2-estimation")
    os.environ.update({"OE_PILOT_CANDIDATE_PYTHON": "/opt/oe-cand/bin/python",
                       "OE_PILOT_GCS_CREDENTIALS": str(key)})
    sys.path.insert(0, str(repo))
    sys.path.insert(0, str(project_site_packages(repo)))
    from experiments.openevolve.colab.checkpoint import GcsMirror
    from experiments.openevolve.run_pilot import run
    mirror = GcsMirror(protocol["bucket"], protocol["gcs_prefix"], root, credentials_path=key)
    run_cohort(protocol, protocol_sha256, llm="real", root=root, mirror=mirror,
               source_commit=os.environ.get("PAPER2_GIT_SHA", ""), run_one=run)


if __name__ == "__main__":
    main()
