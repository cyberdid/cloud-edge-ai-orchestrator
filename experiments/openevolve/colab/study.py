"""VM-side driver of the Colab scale study
(docs/superpowers/specs/2026-10-30-colab-moe-study-design.md).

Walks the fixed run list two runs at a time (one model at a time), resumes unfinished
runs, mirrors run folders to the bucket and writes ``state.json`` for the laptop
watchdog. The final Argon phase needs the owner's ``agy`` login; without it, or on
quota exhaustion, the driver records why it stopped and exits.
"""
from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from dataclasses import dataclass
from itertools import groupby
from pathlib import Path
from typing import Callable, Iterator

BUCKET = "gs://oe-colab-scale-0903767759"
MODELS = ("gemma4:26b", "gemma4:12b", "gemma4:e4b", "gemma4:31b")
MOE = "gemma4:26b"
TASKS = ("circle_packing", "knn")
SEEDS = tuple(101 * i for i in range(1, 11))
LC_SEEDS = SEEDS[:5]
# 31B dense runs ~10 tokens/s on the L4 (spike 2026-10-30), ~6x slower than the MoE:
# 5 seeds keep the study under the compute-unit cap.
SEEDS_BY_MODEL = {"gemma4:31b": SEEDS[:5]}
LONG_SEEDS = SEEDS[:3]
LONG_ITERATIONS = {"circle_packing": 1000, "knn": 500}
CANDIDATE_PYTHON = "/opt/oe-cand/bin/python"
QUOTA_EXIT = 3


@dataclass(frozen=True)
class RunSpec:
    model: str
    task: str
    arm: str
    seed: int
    max_iter: int = 100
    long: bool = False

    @property
    def slug(self) -> str:
        return self.model.replace(":", "-")

    @property
    def dirname(self) -> str:
        return f"{self.task}_{self.arm}_s{self.seed}" + ("_long" if self.long else "")

    def path(self, root: Path) -> Path:
        return Path(root) / self.slug / self.dirname


def study_plan() -> list[RunSpec]:
    plan = [RunSpec(model, task, arm, seed)
            for model in MODELS for arm in ("L", "B") for task in TASKS
            for seed in SEEDS_BY_MODEL.get(model, SEEDS)]
    plan += [RunSpec(MOE, task, "L", seed, LONG_ITERATIONS[task], long=True) for task in TASKS for seed in LONG_SEEDS]
    plan += [RunSpec(MOE, task, "LC", seed) for task in TASKS for seed in LC_SEEDS]
    return plan


def run_status(path: Path) -> str:
    manifest = Path(path) / "manifest.json"
    if not manifest.exists():
        return "missing"
    return json.loads(manifest.read_text()).get("status", "missing")


def batches(plan: list[RunSpec], root: Path, concurrency: int = 2) -> Iterator[list[RunSpec]]:
    pending = [r for r in plan if run_status(r.path(root)) != "complete"]
    for _, group in groupby(pending, key=lambda r: (r.model, r.arm == "LC")):
        group = list(group)
        for start in range(0, len(group), concurrency):
            yield group[start:start + concurrency]


def launch_args(spec: RunSpec, root: Path) -> list[str]:
    args = [sys.executable, "-m", "experiments.openevolve.run_pilot", "--task", spec.task, "--arm", spec.arm,
            "--seed", str(spec.seed), "--out", str(spec.path(root))]
    return args + (["--resume"] if spec.path(root).exists() else [])


def launch(spec: RunSpec, root: Path) -> subprocess.Popen:
    env = dict(os.environ, OE_PILOT_GEMMA_MODEL=spec.model, OE_PILOT_MAX_ITER=str(spec.max_iter),
               OE_PILOT_EXECUTOR="subprocess", OE_PILOT_CANDIDATE_PYTHON=CANDIDATE_PYTHON)
    args = launch_args(spec, root)
    spec.path(root).mkdir(parents=True, exist_ok=True)
    log = open(spec.path(root) / "driver.log", "a")
    return subprocess.Popen(args, env=env, stdout=log, stderr=subprocess.STDOUT)


def _ollama_up() -> bool:
    try:
        with urllib.request.urlopen("http://127.0.0.1:11434/api/version", timeout=5):
            return True
    except OSError:
        return False


def _start_ollama() -> None:
    env = dict(os.environ, OLLAMA_HOST="127.0.0.1:11434", OLLAMA_NUM_PARALLEL="2", OLLAMA_KEEP_ALIVE="-1")
    subprocess.Popen(["ollama", "serve"], env=env, stdout=open("/root/.oe/ollama.log", "a"),
                     stderr=subprocess.STDOUT, start_new_session=True)


def _wait_ollama() -> bool:
    for _ in range(60):
        if _ollama_up():
            return True
        time.sleep(1)
    return False


def ensure_ollama(is_up: Callable[[], bool] = _ollama_up, start: Callable[[], None] = _start_ollama,
                  wait: Callable[[], bool] = _wait_ollama) -> None:
    """The server dies with the kernel (it is a child of the cell); start it again if needed."""
    if not is_up():
        start()
        if not wait():
            raise RuntimeError("ollama serve did not come up")


def model_digest(model: str) -> str | None:
    """Short digest of the local ``model`` build, as ``ollama list`` prints it."""
    lines = subprocess.run(["ollama", "list"], capture_output=True, text=True).stdout.splitlines()[1:]
    for line in lines:
        parts = line.split()
        if len(parts) > 1 and parts[0] == model:
            return parts[1]
    return None


def prepare_model(model: str) -> str | None:
    """Unload other models (24 GB holds one), make sure ``model`` is pulled and return
    the digest of the pulled build (a fresh VM pulls the registry's current build)."""
    ensure_ollama()
    loaded = subprocess.run(["ollama", "ps"], capture_output=True, text=True).stdout.splitlines()[1:]
    for line in loaded:
        name = line.split()[0]
        if name != model:
            subprocess.run(["ollama", "stop", name], capture_output=True)
    subprocess.run(["ollama", "pull", model], capture_output=True, check=True)
    return model_digest(model)


def record_digest(root: Path, model: str, digest: str, sync) -> None:
    """Append the build to ``<model>/model_digest.json`` whenever it differs from the last one
    (registry tags can change between VMs; the history shows which runs used which build)."""
    path = Path(root) / model.replace(":", "-") / "model_digest.json"
    history = json.loads(path.read_text()) if path.exists() else []
    if history and history[-1]["digest"] == digest:
        return
    history.append({"digest": digest, "recorded": time.time(), "host": socket.gethostname()})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(history, indent=1))
    sync.push_file(path)


def agy_ready() -> bool:
    with tempfile.TemporaryDirectory(prefix="agy-check-") as empty:
        try:
            out = subprocess.run(["agy", "-p", "Reply with OK", "--output-format", "json"], cwd=empty,
                                 capture_output=True, text=True, timeout=300)
        except (OSError, subprocess.TimeoutExpired):
            return False
    try:
        return json.loads(out.stdout).get("status") != "ERROR"
    except json.JSONDecodeError:
        return False


class Sync:
    """Mirror run folders and the study state to the bucket with the service account."""

    def __init__(self, bucket: str, local: Path, gcloud_config: Path = Path("/root/.oe/gcloud")):
        self.bucket, self.local = bucket, Path(local)
        self.env = dict(os.environ, CLOUDSDK_CONFIG=str(gcloud_config))

    def _gcloud(self, *args: str) -> None:
        subprocess.run(["gcloud", "storage", *args, "--quiet"], env=self.env, capture_output=True)

    def restore(self) -> None:
        self.local.mkdir(parents=True, exist_ok=True)
        self._gcloud("rsync", "-r", f"{self.bucket}/runs", str(self.local))

    def push(self, path: Path) -> None:
        relative = Path(path).relative_to(self.local).as_posix()
        self._gcloud("rsync", "-r", str(path), f"{self.bucket}/runs/{relative}")

    def push_file(self, path: Path) -> None:
        relative = Path(path).relative_to(self.local).as_posix()
        self._gcloud("cp", str(path), f"{self.bucket}/runs/{relative}")

    def write_state(self, state: dict) -> None:
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
            json.dump(state, handle)
        self._gcloud("cp", handle.name, f"{self.bucket}/state.json")
        os.unlink(handle.name)


def main(root: Path, sync, plan: list[RunSpec] | None = None, runner: Callable = launch,
         prepare_model: Callable[[str], str | None] = prepare_model, agy_check: Callable[[], bool] = agy_ready,
         sleep: Callable[[float], None] = time.sleep, push_interval: float = 600) -> str:
    plan = plan if plan is not None else study_plan()
    root = Path(root)
    state = {"phase": "running", "blocked": None, "host": socket.gethostname(), "total": len(plan)}

    def write(**fields) -> None:
        state.update(fields, updated=time.time(),
                     done=sum(run_status(r.path(root)) == "complete" for r in plan))
        sync.write_state(dict(state))

    write()
    sync.restore()
    write()
    current_model, agy_checked = None, False
    for batch in batches(plan, root):
        if batch[0].arm == "LC" and not agy_checked:
            if not agy_check():
                write(phase="blocked", blocked="agy_login")
                return "blocked_agy"
            agy_checked = True
        if batch[0].model != current_model:
            digest = prepare_model(batch[0].model)
            current_model = batch[0].model
            if digest:
                state.setdefault("digests", {})[current_model] = digest
                record_digest(root, current_model, digest, sync)
            write(model=current_model)
        procs = [(spec, runner(spec, root)) for spec in batch]
        last_push = time.monotonic()
        while any(proc.poll() is None for _, proc in procs):
            sleep(30)
            if time.monotonic() - last_push >= push_interval:
                for spec, _ in procs:
                    sync.push(spec.path(root))
                write()
                last_push = time.monotonic()
        for spec, _ in procs:
            sync.push(spec.path(root))
        if any(proc.returncode == QUOTA_EXIT for _, proc in procs):
            write(phase="blocked", blocked="argon_quota")
            return "blocked_quota"
        write()
    write(phase="done")
    return "done"


if __name__ == "__main__":
    runs = Path(os.environ.get("OE_COLAB_RUNS", "/root/oe-runs"))
    print(main(runs, Sync(BUCKET, runs)))
