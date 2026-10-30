# Colab Scale Study Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the OpenEvolve pilot protocol on Colab L4 for Gemma 4 E4B/12B/26B-A4B/31B (MoE in focus, Argon arm last), autonomously across VM restarts, with results in Cloud Storage and re-checks on the laptop.

**Architecture:** The pilot code (`experiments/openevolve/`) gains two switches: the Gemma model tag and a subprocess executor for VMs without Docker. A VM-side driver walks a fixed run list (two runs at a time, one model at a time), syncing run folders to a GCS bucket with a bucket-only service account. A laptop watchdog (systemd user timer) restarts the VM from the bucket state, enforces the compute-unit cap and stops finished or blocked sessions.

**Tech Stack:** Python 3.13, uv, OpenEvolve 0.4.0, Ollama on Colab, `colab` CLI 0.7.5, `gcloud storage`, systemd user timers, pytest.

**Spec:** `docs/superpowers/specs/2026-10-30-colab-moe-study-design.md` (pilot spec and pre-registration: `docs/superpowers/specs/2026-10-29-openevolve-pilot-design.md`, `docs/openevolve_pilot/PREREGISTRATION.md`)

## Global Constraints

- **Never modify the main checkout while the laptop pilot runs.** All work in worktree `../antigravity-colab-scale` on branch `feat/colab-scale` (from `feat/openevolve-pilot`). New switches default to the pilot's behaviour (`gemma4:12b`, Docker executor), so pilot results are reproducible from either branch.
- Models (bare tags): `gemma4:e4b`, `gemma4:12b`, `gemma4:26b`, `gemma4:31b`. Generation identical to the pilot: `think=false`, `num_ctx=16384`, `num_predict=2048`, T 0.7, top_p 0.95.
- Argon: `gemini-4-argon-medium` via `agy` on the VM, LC phase last, never substituted.
- Run list: per model, 2 tasks × arms L, B × seeds 101,202,…,1010 × 100 calls; MoE long runs arm L seeds 101,202,303 (circle packing 1000, kNN 500); MoE LC 2 tasks × seeds 101…505 × 100 calls. Order: MoE L/B → 12B → E4B → 31B → MoE long → MoE LC.
- Concurrency 2 runs per VM, same model only.
- GCS: project `gen-lang-client-0903767759`, bucket `gs://oe-colab-scale-0903767759` (region `us-west4`), service account `oe-colab-sync` with `roles/storage.objectAdmin` on that bucket only; key at `~/.config/oe-colab/sa-key.json` (mode 600) on the laptop, `/root/.oe/sa-key.json` (600) on VMs.
- Compute-unit cap: stop when spent ≥ 330 units since the study's start balance (399.47).
- Candidates on VMs run as uid/gid 65534 with no read access to `/root`; network cut with `unshare -n` when available.
- Results: VM `/root/oe-runs/<model-slug>/<task>_<arm>_s<seed>[_long]/`, mirrored to `gs://…/runs/`; on the laptop `benchmarks/colab_scale/`.

## Review Focus

1. VM dies mid-run → next VM restores from GCS and resumes; logged calls never re-spent (Task 4 test `test_restore_then_resume_skips_complete`).
2. Two runs of different models at once would load two models into 24 GB → driver never mixes models in one batch (Task 4 test `test_batches_never_mix_models`).
3. LC phase reached without `agy` login or with quota exhausted → state `blocked_agy`/`blocked_quota`, VM released, no Gemma substitution (Task 4 test `test_lc_blocked_on_auth_writes_state`).
4. Laptop asleep for hours, then wakes → watchdog acts once (lock file), does not create a second VM while one is starting (Task 5 test `test_starting_state_is_not_restarted`).
5. Balance falls past the cap while a VM runs → watchdog stops the session and marks the study `capped` (Task 5 test `test_cap_stops_session`).

---

### Task 1: Worktree, model switch and subprocess executor

**Files:**
- Modify: `experiments/openevolve/run_pilot.py` (model from `OE_PILOT_GEMMA_MODEL`, Ollama URL from `OE_PILOT_OLLAMA_URL`)
- Modify: `experiments/openevolve/sandbox_exec.py` (executor from `OE_PILOT_EXECUTOR`: `docker` default | `subprocess`)
- Test: `tests/test_oe_colab_switches.py`

**Interfaces:**
- Produces: `run_pilot.gemma_model() -> str` (env or `"gemma4:12b"`); manifest `models.gemma` records it. `run_candidate(...)` unchanged signature; with `OE_PILOT_EXECUTOR=subprocess` it runs `python3 driver.py` (interpreter from `OE_PILOT_CANDIDATE_PYTHON`, default `sys.executable`) in a fresh temp workspace whose `/output` path is replaced by an `output/` subdirectory (driver sources are rewritten from `/output/` to the absolute temp output path), with `RLIMIT_AS` 4 GiB, `RLIMIT_FSIZE` 16 MiB, the same timeout and output-size rules; when `os.geteuid() == 0` it drops to 65534:65534 (`os.setgid`/`os.setuid` in `preexec_fn`) and prefixes `unshare -n` if `unshare -n true` succeeded once.

- [ ] **Step 1:** `git worktree add ../antigravity-colab-scale -b feat/colab-scale feat/openevolve-pilot`; copy the design + this plan into it and commit `docs(colab): design and plan`.
- [ ] **Step 2: Failing tests** (no root, no Docker needed):
  - `test_gemma_model_default_and_env` — default `"gemma4:12b"`; env `gemma4:26b` → `"gemma4:26b"`.
  - `test_subprocess_executor_returns_output` / `test_subprocess_timeout` / `test_subprocess_output_too_large` / `test_subprocess_crash_tail` — same assertions as `tests/test_oe_pilot_sandbox.py`, with `OE_PILOT_EXECUTOR=subprocess`.
  - `test_working_evaluators_run_under_subprocess` — circle-packing seed via `working_evaluator.evaluate` under `subprocess` → `combined_score > 0`.
- [ ] **Step 3:** Run `uv run --extra evolve pytest tests/test_oe_colab_switches.py -q` → FAIL. **Step 4:** implement. **Step 5:** same command → PASS, and `RUN_DOCKER_TESTS=1 … pytest tests/test_oe_pilot_*.py` still 54+ pass. **Step 6:** commit `feat(colab): gemma model switch and subprocess executor`.

### Task 2: GCS bucket and bucket-only service account

**Files:**
- Create: `scripts/colab/create_bucket.sh` (idempotent; prints what exists/was created)
- Create: `docs/colab_scale/INFRA.md` (names, roles, cost note, how to revoke: delete key, delete SA, delete bucket)

- [ ] **Step 1:** Script runs, against project `gen-lang-client-0903767759`: `gcloud storage buckets create gs://oe-colab-scale-0903767759 --location=us-west4 --uniform-bucket-level-access` (skip if exists); `gcloud iam service-accounts create oe-colab-sync`; `gcloud storage buckets add-iam-policy-binding … --member=serviceAccount:oe-colab-sync@….iam.gserviceaccount.com --role=roles/storage.objectAdmin`; `gcloud iam service-accounts keys create ~/.config/oe-colab/sa-key.json` (only if the file is absent; `chmod 600`).
- [ ] **Step 2:** Verify with the key, not user creds: `CLOUDSDK_CONFIG=$(mktemp -d) gcloud auth activate-service-account --key-file … && gcloud storage cp <tmpfile> gs://…/probe.txt && gcloud storage rm gs://…/probe.txt` → success; and `gcloud storage ls gs://` with the SA lists only what it may see (no other buckets readable).
- [ ] **Step 3:** Commit script + INFRA.md (never the key).

### Task 3: VM bootstrap

**Files:**
- Create: `experiments/openevolve/colab/__init__.py`, `experiments/openevolve/colab/bootstrap.sh`
- Test: `tests/test_oe_colab_bootstrap.py` (static checks of the script)

**Interfaces:**
- Produces: `bootstrap.sh <tarball> <model-tag>` on a VM (root): unpacks the repo to `/root/oe` (`chmod 700 /root`); `uv sync --locked --extra evolve --extra research`; creates `/opt/oe-cand` venv (world-readable) from `antigravity_hybrid_orchestrator/runtime/requirements.lock`; installs Ollama, starts `ollama serve` with `OLLAMA_NUM_PARALLEL=2 OLLAMA_KEEP_ALIVE=-1`; pulls the model; activates the service account from `/root/.oe/sa-key.json` into `CLOUDSDK_CONFIG=/root/.oe/gcloud`; restores `gs://…/runs/` into `/root/oe-runs/`; prints `BOOTSTRAP OK <ollama version> <model digest>`.
- [ ] **Step 1: Failing tests:** `test_bootstrap_pins_versions` (script text contains `--locked`, `requirements.lock`, `OLLAMA_NUM_PARALLEL=2`, `chmod 700 /root`, `/opt/oe-cand`), `test_bootstrap_never_echoes_key` (no `cat` of the key path), `bash -n` syntax check.
- [ ] **Step 2–4:** FAIL → implement → PASS (`uv run pytest tests/test_oe_colab_bootstrap.py -q`). Commit `feat(colab): VM bootstrap`.

### Task 4: VM driver (run list, batching, sync, LC gate)

**Files:**
- Create: `experiments/openevolve/colab/study.py`
- Test: `tests/test_oe_colab_study.py`

**Interfaces:**
- Consumes: `run_pilot.run(...)` via subprocess `python -m experiments.openevolve.run_pilot` with env `OE_PILOT_GEMMA_MODEL`, `OE_PILOT_MAX_ITER`, `OE_PILOT_EXECUTOR=subprocess`, `OE_PILOT_CANDIDATE_PYTHON=/opt/oe-cand/bin/python`; exit code 3 = quota pause.
- Produces:
  - `@dataclass(frozen=True) class RunSpec: model: str; task: str; arm: str; seed: int; max_iter: int; long: bool` and `RunSpec.dirname -> str` (`{task}_{arm}_s{seed}` + `_long`), `RunSpec.slug -> str` (`gemma4-26b` etc.).
  - `study_plan() -> list[RunSpec]` in the Global Constraints order (MoE L/B: 40, 12B: 40, E4B: 40, 31B: 40, long: 6, LC: 10 → 176 runs).
  - `batches(plan, done: set[str], concurrency=2) -> Iterator[list[RunSpec]]` — skips done runs, yields up to 2 runs of one model; LC runs only in the LC phase.
  - `run_status(path) -> str` from `manifest.json` (`complete|incomplete|paused_quota|missing`).
  - `class Sync: __init__(bucket: str, local: Path, gcloud_config: Path)`; `.push(path)`, `.restore()` — `gcloud storage rsync -r` with `CLOUDSDK_CONFIG`; `.write_state(dict)` → `gs://…/state.json` (`phase`, `model`, `vm`, `updated`, `blocked`).
  - `agy_ready() -> bool` — one `agy -p "Reply with OK" --output-format json` call in an empty dir; false on auth error.
  - `main()` loop: restore → for each batch: run in parallel, push each finished run, push in-progress runs every 10 min → on entering LC: if not `agy_ready()` write `blocked: "agy_login"` and exit; on exit code 3 write `blocked: "argon_quota"` and exit → at the end write `phase: "done"`.
- [ ] **Step 1: Failing tests** (fake runner, fake sync, no network):
  - `test_plan_order_and_counts` — 176 runs; first 40 are `gemma4:26b` L/B; last 10 are LC; long runs have `max_iter` 1000 (circle_packing) / 500 (knn).
  - `test_batches_never_mix_models` — across a plan with 1 remaining 26b run and 12b runs, no batch contains two models.
  - `test_restore_then_resume_skips_complete` — runs marked complete in the restored tree are not yielded; an `incomplete` one is yielded and invoked with `--resume`.
  - `test_lc_blocked_on_auth_writes_state` — `agy_ready` false → state `blocked: "agy_login"`, no LC run invoked, no Gemma run substituted.
  - `test_quota_exit_blocks_lc` — fake run exits 3 → state `blocked: "argon_quota"`, loop exits.
- [ ] **Step 2–4:** FAIL → implement → PASS (`uv run --extra evolve pytest tests/test_oe_colab_study.py -q`). Commit `feat(colab): VM study driver`.

### Task 5: Laptop watchdog

**Files:**
- Create: `experiments/openevolve/colab/watchdog.py`, `scripts/colab/watchdog.service`, `scripts/colab/watchdog.timer`, `scripts/colab/install_watchdog.sh`
- Test: `tests/test_oe_colab_watchdog.py`

**Interfaces:**
- Produces:
  - `decide(state: dict | None, sessions: list[str], balance: float, start_balance: float = 399.47, cap: float = 330, now: float) -> Literal["none","start","stop","stop_capped"]` — pure:
    - spent ≥ cap → `stop_capped` (and no start);
    - `phase == "done"` or `blocked` set → `stop` if a session exists else `none`;
    - `phase == "starting"` and `now - updated < 1800` → `none`;
    - no session and work remains → `start`; session exists and state fresh (`now - updated < 1800`) → `none`; session exists but state stale ≥ 1800 s → `stop` (next tick starts a fresh VM).
  - `start_vm()` — `colab new -s oe-scale --gpu L4`; `colab upload` tarball (`git archive HEAD` of the worktree), `bootstrap.sh`, SA key to `/root/.oe/sa-key.json`; `colab exec` bootstrap; then launches `colab exec -f study_cell.py` detached (`setsid nohup`), writing `phase: "starting"` first.
  - `main()` — flock `~/.cache/oe-colab/watchdog.lock`; reads state with the laptop user's gcloud; logs every decision to `~/.cache/oe-colab/watchdog.log` and appends VM starts/stops to `gs://…/vm_log.jsonl`.
  - Timer: every 10 min, `Persistent=true`; `install_watchdog.sh` copies units to `~/.config/systemd/user/` and enables the timer.
- [ ] **Step 1: Failing tests** for `decide`: `test_cap_stops_session`, `test_done_stops_session`, `test_blocked_stops_and_does_not_restart`, `test_starting_state_is_not_restarted`, `test_no_session_with_work_starts`, `test_stale_session_is_stopped`.
- [ ] **Step 2–4:** FAIL → implement → PASS (`uv run pytest tests/test_oe_colab_watchdog.py -q`). Commit `feat(colab): laptop watchdog`.

### Task 6: Spike on L4 (throwaway) and pre-registration

**Files:**
- Create: `docs/colab_scale/spike_notes.md`, `docs/colab_scale/PREREGISTRATION.md`

- [ ] **Step 1:** `python -m experiments.openevolve.colab.watchdog --start-only --spike` creates a VM and bootstraps with `gemma4:26b`; then on the VM: 10 iterations of `circle_packing` L with concurrency 2 vs 1 (seed 1, into `/root/spike`, not synced to `runs/`). Record s/iteration both ways; keep concurrency 2 only if throughput (iterations/hour) improves by ≥ 20%.
- [ ] **Step 2:** Isolation checks as uid 65534: cannot read `/root/oe`, `/root/.oe/sa-key.json`; `unshare -n` available or not (record); network blocked if available.
- [ ] **Step 3:** `ollama pull gemma4:31b`; one mutation with `num_ctx=16384`: record whether it is 100% GPU and s/call. If not fully on GPU, drop 31B from the plan (spec rule) — edit `study_plan()` with a ruling and test update.
- [ ] **Step 4:** Stop the VM. Write the pre-registration (frozen run list, versions: Ollama on VM, model digests, OpenEvolve 0.4.0, executor and isolation facts, cap, endpoints identical to the pilot, deviation policy incl. VM restarts and executor difference). Commit and tag `colab-scale-prereg`.

### Task 7: Launch and monitor

- [ ] **Step 1:** `bash scripts/colab/install_watchdog.sh`; first tick starts the VM; confirm `state.json` phase advances and the first MoE runs produce `calls.jsonl` in the bucket.
- [ ] **Step 2:** On `blocked: "agy_login"` (end of study): tell the owner; run `colab url -s oe-scale --open` for the owner's `agy` login in the Colab terminal, then clear `blocked` and let the watchdog restart the driver.
- [ ] **Step 3:** On completion: download `gs://…/runs/` to `benchmarks/colab_scale/`, run the pilot re-check (`analyze.main` with the Docker sandbox) per model, write `docs/COLAB_SCALE_RESULTS.md`; disable the timer; owner decides whether to delete the SA key and bucket.
