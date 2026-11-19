"""Bootstrap and run the preregistered, resumable V1/V2 kNN feasibility pilot."""
from __future__ import annotations

import json
import hashlib
import os
import shutil
import subprocess
import sys
import tarfile
import threading
import time
from pathlib import Path

BUCKET = "oe-colab-scale-0903767759"
PREFIX = "paper2-v2/b1"
ROOT = Path("/content/paper2-v2/b1")
REPO = Path("/content/oe")
KEY_SOURCE = Path("/content/sa-key.json")
KEY = Path("/content/.oe/sa-key.json")
CALLS_PER_CONDITION = 20
SEED = 20261119
MODEL = "gemma4:12b"
ORDER = ("v1", "v2")


def run_checked(args: list[str], *, env=None, cwd=None, timeout: int = 3600) -> str:
    result = subprocess.run(args, env=env, cwd=cwd, capture_output=True, text=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError(f"command failed ({result.returncode}): {' '.join(args)}\n{result.stderr[-1200:]}")
    return result.stdout


def ensure_system_package(package: str) -> None:
    if shutil.which(package) is not None:
        return
    run_checked(["apt-get", "update", "-qq"], timeout=1200)
    run_checked(["apt-get", "install", "-y", package], timeout=1200)


def bootstrap() -> dict:
    """Install pinned research dependencies and the local Gemma inference server."""
    if KEY_SOURCE.exists():
        KEY.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        os.replace(KEY_SOURCE, KEY)
        KEY.chmod(0o600)
    if not KEY.exists():
        raise RuntimeError("GCS credential is missing; watchdog must upload it before starting")
    KEY.parent.chmod(0o700)

    if not (REPO / "pyproject.toml").exists():
        archive = Path("/content/oe.tar.gz")
        if not archive.exists():
            raise RuntimeError("source archive /content/oe.tar.gz is missing")
        REPO.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive, "r:gz") as bundle:
            bundle.extractall(REPO, filter="data")

    try:
        run_checked(["uv", "--version"])
    except (FileNotFoundError, RuntimeError):
        run_checked([sys.executable, "-m", "pip", "install", "-q", "uv"], timeout=1200)

    env = dict(os.environ, UV_PROJECT_ENVIRONMENT=str(REPO / ".venv"),
               UV_PYTHON_INSTALL_DIR="/opt/uv-python", PYTHONDONTWRITEBYTECODE="1")
    run_checked(["uv", "sync", "--locked", "--extra", "evolve", "--extra", "research"],
                cwd=REPO, env=env, timeout=2400)
    run_checked(["uv", "python", "install", "3.12"], env=env, timeout=1200)
    candidate_python = Path("/opt/oe-cand/bin/python")
    if not candidate_python.exists():
        run_checked(["uv", "venv", "/opt/oe-cand", "--python", "3.12"], env=env, timeout=1200)
        run_checked(["uv", "pip", "install", "--python", str(candidate_python), "-r",
                     str(REPO / "antigravity_hybrid_orchestrator/runtime/requirements.lock")],
                    env=env, timeout=1800)
    run_checked(["chmod", "-R", "a+rX", "/opt/oe-cand", "/opt/uv-python"])

    if shutil.which("ollama") is None:
        ensure_system_package("zstd")
        installer_env = dict(os.environ, OLLAMA_VERSION="0.40.13")
        run_checked(["bash", "-lc", "curl -fsSL https://ollama.com/install.sh | sh"],
                    env=installer_env, timeout=1800)
    try:
        run_checked(["curl", "-fsS", "http://127.0.0.1:11434/api/version"], timeout=10)
    except (FileNotFoundError, RuntimeError, subprocess.TimeoutExpired):
        server_log = open("/content/ollama-paper2.log", "a")
        subprocess.Popen(["ollama", "serve"], env=dict(os.environ, OLLAMA_HOST="127.0.0.1:11434",
                                                         OLLAMA_NUM_PARALLEL="1", OLLAMA_KEEP_ALIVE="-1"),
                         stdout=server_log, stderr=subprocess.STDOUT, start_new_session=True)
        for _ in range(60):
            try:
                run_checked(["curl", "-fsS", "http://127.0.0.1:11434/api/version"], timeout=5)
                break
            except (RuntimeError, subprocess.TimeoutExpired):
                time.sleep(1)
        else:
            raise RuntimeError("ollama serve did not become ready")
    run_checked(["ollama", "pull", MODEL], timeout=3600)
    return {"model": MODEL, "model_digest": model_digest(), "candidate_python": str(candidate_python),
            "gpu": gpu_name(), "network_namespace": network_namespace_available()}


def model_digest() -> str | None:
    rows = run_checked(["ollama", "list"]).splitlines()[1:]
    for line in rows:
        parts = line.split()
        if len(parts) > 1 and parts[0] == MODEL:
            return parts[1]
    return None


def gpu_name() -> str:
    result = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                            capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def network_namespace_available() -> bool:
    result = subprocess.run(["unshare", "-n", "true"], capture_output=True, text=True)
    return result.returncode == 0


def main() -> None:
    meta = bootstrap()
    protocol_path = REPO / "experiments/openevolve/configs/evaluator_v2_pilot.json"
    protocol_text = protocol_path.read_text()
    protocol = json.loads(protocol_text)
    if (protocol["model"], protocol["task"], protocol["seed"], protocol["calls_per_condition"],
            tuple(protocol["evaluator_order"])) != (MODEL, "knn", SEED, CALLS_PER_CONDITION, ORDER):
        raise RuntimeError("runner constants do not match the frozen preregistration config")
    protocol_sha256 = hashlib.sha256(protocol_text.encode()).hexdigest()
    env = dict(os.environ, OE_PILOT_GEMMA_MODEL=MODEL, OE_PILOT_EXECUTOR="subprocess",
               OE_PILOT_CANDIDATE_PYTHON="/opt/oe-cand/bin/python", OE_PILOT_MAX_ITER=str(CALLS_PER_CONDITION),
               OE_PILOT_GIT_SHA=os.environ.get("PAPER2_GIT_SHA", "unknown"),
               OE_PILOT_MODEL_DIGEST=meta["model_digest"], OE_PILOT_PROTOCOL_ID=protocol["protocol_id"],
               OE_PILOT_PROTOCOL_SHA256=protocol_sha256,
               OE_PILOT_GCS_BUCKET=BUCKET, OE_PILOT_GCS_PREFIX=PREFIX,
               OE_PILOT_GCS_ROOT=str(ROOT), OE_PILOT_GCS_CREDENTIALS=str(KEY))
    os.environ.update(env)
    os.chdir(REPO)
    sys.path.insert(0, str(REPO))
    from experiments.openevolve.colab.checkpoint import GcsMirror
    from experiments.openevolve.run_pilot import run

    ROOT.mkdir(parents=True, exist_ok=True)
    mirror = GcsMirror(BUCKET, PREFIX, ROOT, credentials_path=KEY)
    mirror.restore()

    state_path = ROOT / "state.json"
    def write_state(phase: str, **extra) -> None:
        state = {"phase": phase, "updated_unix": time.time(), "git_sha": env["OE_PILOT_GIT_SHA"],
                 "protocol_id": protocol["protocol_id"], "protocol_sha256": protocol_sha256,
                 "model": MODEL, "task": "knn", "seed": SEED, "calls_per_condition": CALLS_PER_CONDITION,
                 "order": list(ORDER), "bootstrap": meta, **extra}
        tmp = state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, indent=2, sort_keys=True))
        tmp.replace(state_path)
        mirror.sync()

    stop_sync = threading.Event()
    def checkpoint_loop() -> None:
        while not stop_sync.wait(30):
            try:
                write_state("running", current_condition=current_condition[0])
            except Exception as exc:
                print(f"checkpoint_error={type(exc).__name__}: {exc}", flush=True)

    current_condition = [None]
    sync_thread = threading.Thread(target=checkpoint_loop, daemon=True)
    sync_thread.start()
    summaries = []
    try:
        write_state("running", current_condition=None)
        for version in ORDER:
            current_condition[0] = version
            out = ROOT / "runs" / f"knn_{version}_s{SEED}"
            manifest_path = out / "manifest.json"
            existing = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
            if existing.get("status") == "complete":
                summary = {"evaluator_version": version, "status": "complete", "resumed_as_existing": True}
            else:
                summary = run("knn", "L", SEED, out, resume=out.exists(), llm="real",
                              evaluator_version=version)
            summaries.append(summary)
            mirror.sync()
            write_state("running", completed_conditions=[s.get("evaluator_version") for s in summaries],
                        current_condition=version)
        write_state("complete", summaries=[{k: s.get(k) for k in ("status", "evaluator_version", "wall_s")}
                                            for s in summaries])
        print(json.dumps({"phase": "complete", "summaries": summaries}, sort_keys=True), flush=True)
    except BaseException as exc:
        try:
            write_state("interrupted" if isinstance(exc, KeyboardInterrupt) else "error",
                        current_condition=current_condition[0], error=f"{type(exc).__name__}: {exc}"[-1200:])
        finally:
            raise
    finally:
        stop_sync.set()
        sync_thread.join(timeout=5)
        mirror.sync()


if __name__ == "__main__":
    main()
