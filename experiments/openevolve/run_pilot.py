"""Run one pilot run: ``--task {circle_packing,knn} --arm {L,LC,B} --seed N --out DIR``.

Writes into DIR: ``calls.jsonl`` (router), ``candidates.jsonl`` (working evaluator),
``openevolve/`` (checkpoints, arms L/LC), ``best_of_n/`` (arm B) and ``manifest.json``.
``--resume`` continues from the logged calls and the latest checkpoint; the router's
budget keeps the run at exactly ``OE_PILOT_MAX_ITER`` (default 100) logged calls.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import multiprocessing
import os
import subprocess
import threading
import time
from pathlib import Path

from openevolve import OpenEvolve
from openevolve.config import load_config

from experiments.openevolve.best_of_n import run_best_of_n
from experiments.openevolve.candidate_log import ENV_VAR
from experiments.openevolve.router import ARGON_MODEL, PAUSE_FILE, Router, agy_argon, ollama_gemma, serve
from experiments.openevolve.tasks import TASKS

CLOUD_SHARE = {"L": 0.0, "LC": 0.1, "B": 0.0}
GEMMA_MODEL = "gemma4:12b"  # the pilot's model; OE_PILOT_GEMMA_MODEL selects another Gemma


def gemma_model() -> str:
    return os.environ.get("OE_PILOT_GEMMA_MODEL", GEMMA_MODEL)


def evaluator_path(task: str, version: str = "v1") -> Path:
    if version not in {"v1", "v1_matched", "v2"}:
        raise ValueError(f"unknown evaluator version: {version}")
    if task not in TASKS:
        raise ValueError(f"unknown task: {task}")
    if version in {"v1_matched", "v2"}:
        if task != "knn":
            raise ValueError(f"evaluator {version} is only available for knn")
        name = "working_evaluator_v2.py" if version == "v2" else "working_evaluator_v1_matched.py"
        return TASKS[task]["evaluator"].with_name(name)
    return TASKS[task]["evaluator"]
STUB_DIFFS = {
    "circle_packing": ("    n = 26\n", "    n = 26  # stub\n"),
    "knn": ('    """Baseline: exact search (recall 1.0, no speed-up)."""\n',
            '    """Baseline: exact search (stub)."""\n'),
}


def stub_reply(task: str) -> str:
    search, replace = STUB_DIFFS[task]
    return f"<<<<<<< SEARCH\n{search}=======\n{replace}>>>>>>> REPLACE\n"


def _lines(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []


def _latest_checkpoint(output_dir: Path) -> tuple[str | None, int]:
    found = [(int(p.name.split("_")[1]), p) for p in (output_dir / "checkpoints").glob("checkpoint_*")]
    if not found:
        return None, 0
    iteration, path = max(found)
    return str(path), iteration


def _router_process(router: Router, ports) -> None:
    server = serve(router, 0)
    ports.put(server.server_address[1])
    threading.Event().wait()


def _serve_in_process(router: Router):
    """Run the router in its own forked process, started before OpenEvolve forks its
    workers, so no worker is forked from a process with a live server thread."""
    context = multiprocessing.get_context("fork")
    ports = context.Queue()
    process = context.Process(target=_router_process, args=(router, ports), daemon=True)
    process.start()
    return process, ports.get(timeout=30)


def run_coroutine_synchronously(coroutine):
    """Run an async OpenEvolve job from both CLI and notebook event loops."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coroutine)
    import nest_asyncio

    nest_asyncio.apply()
    return asyncio.run(coroutine)


def _git_sha() -> str:
    return os.environ.get("OE_PILOT_GIT_SHA") or subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()


def _call_checkpoint_callback():
    """Lazily build the GCS client after the router process forks, if configured."""
    bucket = os.environ.get("OE_PILOT_GCS_BUCKET")
    prefix = os.environ.get("OE_PILOT_GCS_PREFIX")
    root = os.environ.get("OE_PILOT_GCS_ROOT")
    credentials = os.environ.get("OE_PILOT_GCS_CREDENTIALS")
    if not (bucket and prefix and root and credentials):
        return None
    holder = {"mirror": None}

    def sync_call_log() -> None:
        if holder["mirror"] is None:
            from experiments.openevolve.colab.checkpoint import GcsMirror

            holder["mirror"] = GcsMirror(bucket, prefix, Path(root), credentials_path=Path(credentials))
        holder["mirror"].sync()

    return sync_call_log


def run(task: str, arm: str, seed: int, out: Path, resume: bool, llm: str = "real",
        backends: tuple | None = None, evaluator_version: str = "v1") -> dict:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    calls_path, manifest_path = out / "calls.jsonl", out / "manifest.json"
    if not resume and _lines(calls_path):
        raise RuntimeError(f"{out} already has logged calls; use --resume")
    max_iter = int(os.environ.get("OE_PILOT_MAX_ITER", "100"))
    num_predict = int(os.environ.get("OE_PILOT_NUM_PREDICT", "2048"))
    spec = TASKS[task]
    selected_evaluator = evaluator_path(task, evaluator_version)
    evaluator_sha256 = hashlib.sha256(selected_evaluator.read_bytes()).hexdigest()
    manifest = json.loads(manifest_path.read_text()) if resume and manifest_path.exists() else {}
    if resume and manifest.get("evaluator_version", evaluator_version) != evaluator_version:
        raise RuntimeError(
            f"cannot resume with evaluator {evaluator_version}; existing run used "
            f"{manifest.get('evaluator_version')}"
        )
    previous_evaluator_sha = manifest.get("evaluator_sha256")
    if resume and previous_evaluator_sha and previous_evaluator_sha != evaluator_sha256:
        raise RuntimeError("cannot resume because the evaluator content hash changed")
    if resume and evaluator_version in {"v1_matched", "v2"} and not previous_evaluator_sha:
        raise RuntimeError("cannot safely resume this evaluator version without its content hash")
    if backends:
        gemma, argon = backends
    elif llm == "stub":
        gemma = argon = lambda messages: (stub_reply(task), 1, 1)  # noqa: E731
    else:
        gemma = ollama_gemma(base_url=os.environ.get("OE_PILOT_OLLAMA_URL", "http://localhost:11434"),
                             model=gemma_model(), num_predict=num_predict)
        argon = agy_argon()
    router = Router(seed, CLOUD_SHARE[arm], calls_path, gemma=gemma, argon=argon,
                    start_index=len(_lines(calls_path)), budget=max_iter,
                    on_call_logged=_call_checkpoint_callback())
    os.environ[ENV_VAR] = str(out / "candidates.jsonl")
    pause_path = out / PAUSE_FILE
    if pause_path.exists():  # resuming after a quota pause: keep the event, clear the flag
        manifest["quota_pauses"] = manifest.get("quota_pauses", []) + [json.loads(pause_path.read_text())]
        pause_path.unlink()
    manifest.update({
        "task": task, "arm": arm, "seed": seed, "cloud_share": CLOUD_SHARE[arm], "max_iterations": max_iter,
        "git_sha": _git_sha(), "openevolve_version": importlib.metadata.version("openevolve"),
        "config_sha256": hashlib.sha256(spec["config"].read_bytes()).hexdigest(),
        "models": {"gemma": gemma_model(), "argon": ARGON_MODEL}, "num_predict": num_predict, "llm": llm,
        "evaluator_version": evaluator_version,
        "evaluator_sha256": evaluator_sha256,
        "model_digest": os.environ.get("OE_PILOT_MODEL_DIGEST"),
        "protocol_id": os.environ.get("OE_PILOT_PROTOCOL_ID"),
        "protocol_sha256": os.environ.get("OE_PILOT_PROTOCOL_SHA256"),
        "status": "running", "starts": manifest.get("starts", []) + [time.time()],
    })
    manifest_path.write_text(json.dumps(manifest, indent=2))
    started = time.time()
    if arm == "B":
        manifest.update(run_best_of_n(task, router, max_iter, out, seed, resume))
    else:
        server, port = _serve_in_process(router)
        try:
            config = load_config(spec["config"])
            config.llm.api_base = f"http://127.0.0.1:{port}/v1"
            config.llm.update_model_params({"api_base": config.llm.api_base, "api_key": "unused"}, overwrite=True)
            config.random_seed = config.database.random_seed = seed
            config.max_iterations = max_iter
            checkpoint = _latest_checkpoint(out / "openevolve")[0] if resume else None
            # Remaining work is counted in logged calls, not checkpoint iterations: iterations
            # after a quota pause or a kill made no (or already-logged) calls.
            remaining = max_iter - router.call_index
            if remaining > 0:
                controller = OpenEvolve(str(spec["initial"]), str(selected_evaluator), config,
                                        output_dir=str(out / "openevolve"))
                run_coroutine_synchronously(
                    controller.run(iterations=remaining, checkpoint_path=checkpoint)
                )
        finally:
            server.terminate()
            server.join(10)
    calls = _lines(calls_path)
    manifest.update({
        "status": "paused_quota" if pause_path.exists() else ("complete" if len(calls) >= max_iter else "incomplete"),
        "wall_s": manifest.get("wall_s", 0.0) + time.time() - started,
        "llm_latency_s": {m: sum(c["latency_s"] for c in calls if c["model"] == m) for m in ("gemma", "argon")},
        "calls_by_status": {s: sum(c["status"] == s for c in calls) for s in {c["status"] for c in calls}},
    })
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=sorted(TASKS), required=True)
    parser.add_argument("--arm", choices=sorted(CLOUD_SHARE), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--llm", choices=("real", "stub"), default="real")
    parser.add_argument("--evaluator-version", choices=("v1", "v1_matched", "v2"), default="v1")
    args = parser.parse_args()
    manifest = run(args.task, args.arm, args.seed, args.out, args.resume, args.llm,
                   evaluator_version=args.evaluator_version)
    print(json.dumps({k: manifest[k] for k in ("task", "arm", "seed", "status")}))
    raise SystemExit(3 if manifest["status"] == "paused_quota" else 0)


if __name__ == "__main__":
    main()
