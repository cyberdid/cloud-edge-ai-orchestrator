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
BIG = "gemma4:31b"
TASKS = ("circle_packing", "knn")
SEEDS = tuple(101 * i for i in range(1, 11))
LC_SEEDS = SEEDS[:5]
# 31B dense runs ~10 tokens/s on the L4 (spike 2026-10-30), ~6x slower than the MoE.
# Deviation 2026-11-05 (docs/colab_scale/DEVIATIONS.md): 31B cut to circle packing,
# 3 seeds, so it fits the cap.
SEEDS_BY_MODEL = {"gemma4:31b": SEEDS[:3]}
TASKS_BY_MODEL = {"gemma4:31b": ("circle_packing",)}
LONG_SEEDS = SEEDS[:3]
LONG_ITERATIONS = {"circle_packing": 1000, "knn": 500}
# Deviation 2026-11-05, exploratory: best-of-N with the long-run call budget, the
# matching control for the MoE circle-packing long runs.
LONG_B_TASKS = ("circle_packing",)
CANDIDATE_PYTHON = "/opt/oe-cand/bin/python"
QUOTA_EXIT = 3
# Runs at once / Ollama slots per model. Two 31B streams got 3.4 tokens/s each (6.8 in
# total) against 9.7 for one stream in the spike, so 31B runs alone on a one-slot server.
SERVING_PARALLEL = {BIG: 1}
DEFAULT_PARALLEL = 2


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
            for model in MODELS for arm in ("L", "B") for task in TASKS_BY_MODEL.get(model, TASKS)
            for seed in SEEDS_BY_MODEL.get(model, SEEDS)]
    # Deviation 2026-11-04 (docs/colab_scale/DEVIATIONS.md): 31B runs ~3x slower than the
    # spike estimate, so the MoE long runs and MoE LC go before the 31B block, which takes
    # what is left of the compute-unit cap.
    late = [r for r in plan if r.model == BIG]
    plan = [r for r in plan if r.model != BIG]
    plan += [RunSpec(MOE, task, "L", seed, LONG_ITERATIONS[task], long=True) for task in TASKS for seed in LONG_SEEDS]
    plan += [RunSpec(MOE, task, "B", seed, LONG_ITERATIONS[task], long=True) for task in LONG_B_TASKS for seed in LONG_SEEDS]
    plan += [RunSpec(MOE, task, "LC", seed) for task in TASKS for seed in LC_SEEDS]
    return plan + late


def run_status(path: Path) -> str:
    manifest = Path(path) / "manifest.json"
    if not manifest.exists():
        return "missing"
    return json.loads(manifest.read_text()).get("status", "missing")


def batches(plan: list[RunSpec], root: Path) -> Iterator[list[RunSpec]]:
    pending = [r for r in plan if run_status(r.path(root)) != "complete"]
    for (model, _), group in groupby(pending, key=lambda r: (r.model, r.arm == "LC")):
        group = list(group)
        concurrency = SERVING_PARALLEL.get(model, DEFAULT_PARALLEL)
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


def _start_ollama(parallel: int = DEFAULT_PARALLEL) -> None:
    env = dict(os.environ, OLLAMA_HOST="127.0.0.1:11434", OLLAMA_NUM_PARALLEL=str(parallel), OLLAMA_KEEP_ALIVE="-1")
    subprocess.Popen(["ollama", "serve"], env=env, stdout=open("/root/.oe/ollama.log", "a"),
                     stderr=subprocess.STDOUT, start_new_session=True)


def _wait_ollama() -> bool:
    for _ in range(60):
        if _ollama_up():
            return True
        time.sleep(1)
    return False


def ensure_ollama(is_up: Callable[[], bool] = _ollama_up, start: Callable[[], None] = None,
                  wait: Callable[[], bool] = _wait_ollama) -> None:
    """The server dies with the kernel (it is a child of the cell); start it again if needed."""
    if not is_up():
        (start or (lambda: _start_ollama(_serving["parallel"] or DEFAULT_PARALLEL)))()
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


_serving = {"parallel": None}  # slots of the server this driver started; None = bootstrap's


def serve_with_parallel(parallel: int) -> None:
    """(Re)start ``ollama serve`` with ``parallel`` slots; Ollama sizes the KV cache for
    all slots when it loads a model, so a 2-slot server can push 31B layers off the GPU."""
    if _serving["parallel"] == parallel and _ollama_up():
        return
    subprocess.run(["pkill", "-x", "ollama"], capture_output=True)
    for _ in range(30):
        if not _ollama_up():
            break
        time.sleep(1)
    _start_ollama(parallel)
    if not _wait_ollama():
        raise RuntimeError("ollama serve did not come up")
    _serving["parallel"] = parallel


def prepare_model(model: str) -> str | None:
    """Unload other models (24 GB holds one), make sure ``model`` is pulled and return
    the digest of the pulled build (a fresh VM pulls the registry's current build)."""
    serve_with_parallel(SERVING_PARALLEL.get(model, DEFAULT_PARALLEL))
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

    def login_window(self) -> bool:
        """True while the object ``agy_login_window`` exists in the bucket."""
        out = subprocess.run(["gcloud", "storage", "ls", f"{self.bucket}/agy_login_window"],
                             env=self.env, capture_output=True)
        return out.returncode == 0

    def write_state(self, state: dict) -> None:
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
            json.dump(state, handle)
        self._gcloud("cp", handle.name, f"{self.bucket}/state.json")
        os.unlink(handle.name)


def main(root: Path, sync, plan: list[RunSpec] | None = None, runner: Callable = launch,
         prepare_model: Callable[[str], str | None] = prepare_model, agy_check: Callable[[], bool] = agy_ready,
         sleep: Callable[[float], None] = time.sleep, push_interval: float = 600,
         login_window: Callable[[], bool] = lambda: False, login_wait_s: float = 3600) -> str:
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
    current = {"model": None}
    agy_ok, deferred = False, []

    def run_batch(batch: list[RunSpec]) -> str | None:
        if batch[0].model != current["model"]:
            digest = prepare_model(batch[0].model)
            current["model"] = batch[0].model
            if digest:
                state.setdefault("digests", {})[current["model"]] = digest
                record_digest(root, current["model"], digest, sync)
            write(model=current["model"])
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
        return None

    # LC needs the owner's agy login on the live VM. Without it, LC batches wait and the
    # remaining Gemma batches run; after each batch the login is checked again and the
    # waiting LC batches go next. Gemma is never substituted for Argon.
    for batch in batches(plan, root):
        if batch[0].arm == "LC" and not agy_ok:
            agy_ok = agy_check()
            if not agy_ok and login_window():
                # The owner asked for a login slot (flag object in the bucket): keep the
                # VM and wait for the agy login in its terminal instead of moving on.
                for _ in range(max(1, int(login_wait_s // 60))):
                    write(waiting="login_window")
                    sleep(60)
                    agy_ok = agy_check()
                    if agy_ok:
                        break
            if agy_ok:
                write(waiting=None)
            else:
                deferred.append(batch)
                write(waiting="agy_login")
                continue
        stop = run_batch(batch)
        if stop:
            return stop
        if deferred and agy_check():
            agy_ok = True
            write(waiting=None)
            for lc_batch in deferred:
                stop = run_batch(lc_batch)
                if stop:
                    return stop
            deferred = []
    if deferred:
        if not agy_check():
            write(phase="blocked", blocked="agy_login")
            return "blocked_agy"
        write(waiting=None)
        for lc_batch in deferred:
            stop = run_batch(lc_batch)
            if stop:
                return stop
    write(phase="done")
    return "done"


if __name__ == "__main__":
    runs = Path(os.environ.get("OE_COLAB_RUNS", "/root/oe-runs"))
    sync = Sync(BUCKET, runs)
    print(main(runs, sync, login_window=sync.login_window))
