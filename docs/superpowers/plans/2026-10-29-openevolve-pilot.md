# OpenEvolve Pilot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run the OpenEvolve pilot (gemma4:12b local mutator, rare Gemini 4 Argon proposer via `agy`, independent re-check of every candidate) on circle packing n=26 and ORION approximate kNN, arms L / L+C / B.

**Architecture:** OpenEvolve 0.4.0 talks to one local OpenAI-compatible *router* that, per call, picks Gemma (Ollama native API) or Argon (`CloudArchitectClient(backend="agy").complete`) with a seeded RNG and logs every call. Each task has a *working evaluator* (drives selection; logs every candidate) and a separate *re-check* (never visible to the model). All candidate code runs in the existing Docker sandbox image. Post-hoc scripts attribute candidates to calls, re-check them, and aggregate.

**Tech Stack:** Python 3.12, uv, OpenEvolve 0.4.0, numpy/scipy, Ollama (`gemma4:12b`), Antigravity CLI `agy`, Docker image `antigravity-orion:local`, pytest, matplotlib.

**Spec:** `docs/superpowers/specs/2026-10-29-openevolve-pilot-design.md`

## Global Constraints

- Local model `gemma4:12b` only; Ollama `think=false`, `num_predict=2048` until the spike (Task 9) freezes the final value.
- Cloud model `gemini-4-argon-medium` only, only through `agy`, always in an empty cwd (use `CloudArchitectClient.complete`). Never substitute another model; on quota exhaustion, pause.
- Budget: exactly 100 LLM calls per run; arms L, LC (cloud share 0.1), B; seeds 101, 202, 303; tasks `circle_packing`, `knn` → 18 runs.
- A failed cloud call counts as a failed mutation, consumes its slot, is never retried with Gemma.
- Candidate code executes only inside `antigravity-orion:local` via `runtime/sandbox.DockerSandbox` (no network, read-only workspace, non-root).
- Evolution edits only the `EVOLVE-BLOCK`; diff-based evolution on; LLM retries 0; one evaluation at a time.
- Re-check code, holdout data and their seeds are never placed in a prompt, a candidate workspace, or the working evaluator.
- Primary endpoint: verified score of the working-best ("shipped") program; failing re-check → seed's score.
- Code in `experiments/openevolve/`, tests in `tests/test_oe_pilot_*.py`, raw data in `benchmarks/openevolve_pilot/`, branch `feat/openevolve-pilot`. Commit after every task; never commit to `main`.
- GPU-free work first: Tasks 1–8 must not need Ollama/GPU (stub LLM in tests). Tasks 9–12 wait for the owner's "GPU is free".

## Review Focus

1. A candidate that hangs, prints megabytes, or allocates huge memory → sandbox timeout/limit, scored as failed, run continues (tests in Tasks 2, 3).
2. Gemma reply with no valid diff, truncated output, or empty Argon reply → counted as a failed iteration, logged with status, budget slot consumed, run does not crash (tests in Tasks 5, 6).
3. Run interrupted (power, owner stops it) → `run_pilot.py --resume` continues from the last checkpoint without re-spending calls already logged (test in Task 6).
4. Candidate returns numpy types, NaN/inf, wrong shapes or extra keys → working evaluator scores 0 without exception; re-check reports `invalid` (tests in Tasks 3, 4).
5. Candidate tries to read the evaluator, holdout or network → nothing to read in the sandbox; static check flags forbidden primitives in the re-check (tests in Tasks 2, 3).

---

### Task 1: Branch, dependency pin, package skeleton, upstream task files

**Files:**
- Modify: `pyproject.toml` (add extra `evolve = ["openevolve==0.4.0"]`), `uv.lock`
- Create: `experiments/__init__.py`, `experiments/openevolve/__init__.py`
- Create: `experiments/openevolve/upstream/circle_packing/initial_program.py`, `experiments/openevolve/upstream/circle_packing/evaluator.py`, `experiments/openevolve/upstream/circle_packing/config.yaml`, `experiments/openevolve/upstream/NOTICE.md`
- Test: `tests/test_oe_pilot_upstream.py`

**Interfaces:**
- Produces: importable `openevolve` 0.4.0; vendored upstream circle-packing files byte-identical to tag `v0.4.0` of `algorithmicsuperintelligence/openevolve` (`examples/circle_packing/`), with source URL + SHA-256 in `NOTICE.md` (Apache-2.0, same as this repo).

- [ ] **Step 1:** `git switch -c feat/openevolve-pilot`; `git add docs/superpowers/specs/2026-10-29-openevolve-pilot-design.md docs/superpowers/plans/2026-10-29-openevolve-pilot.md`; commit `docs(oe-pilot): design and implementation plan`.
- [ ] **Step 2: Write the failing test** `tests/test_oe_pilot_upstream.py`:

```python
def test_openevolve_pinned_and_api_present():
    import importlib.metadata as md
    assert md.version("openevolve") == "0.4.0"
    from openevolve import OpenEvolve                      # noqa: F401
    from openevolve.config import load_config              # noqa: F401
    from openevolve.utils.code_utils import extract_diffs, apply_diff  # noqa: F401

def test_vendored_files_match_notice(project_root):
    # NOTICE.md lists lines "<path relative to upstream/>  <sha256>" for all 3 files
    # assert: exactly 3 such lines; for each, hashlib.sha256(file bytes).hexdigest() == listed hash
```

- [ ] **Step 2b:** If any import name differs in 0.4.0, locate the equivalent in the installed package, use it in the test, and list the mapping in `NOTICE.md` under "API names used" — later tasks import only names listed there.
- [ ] **Step 3:** `uv add --optional evolve openevolve==0.4.0`; download the three upstream files from the `v0.4.0` tag; write `NOTICE.md`.
- [ ] **Step 4:** Run `uv run --extra evolve pytest tests/test_oe_pilot_upstream.py -v` → PASS.
- [ ] **Step 5:** Commit `feat(oe-pilot): pin openevolve 0.4.0 and vendor circle-packing example`.

### Task 2: Sandboxed candidate execution

**Files:**
- Create: `experiments/openevolve/sandbox_exec.py`
- Test: `tests/test_oe_pilot_sandbox.py`

**Interfaces:**
- Produces:
  - `@dataclass(frozen=True) class SandboxResult: ok: bool; output: dict | None; error: str; elapsed_s: float`
  - `run_candidate(program_code: str, driver_code: str, payload: dict, timeout_s: float, image: str = "antigravity-orion:local") -> SandboxResult`
  - Contract: writes `candidate.py`, `driver.py`, `payload.json` into a fresh temp workspace; runs `python3 driver.py` via `DockerSandbox(workspace, image, output_directory=out).build_command(..., read_only=True)` with `Popen(shell=False)`; the driver writes `/output/result.json`; on timeout kills and calls `cleanup()`; result JSON larger than 8 MiB → `ok=False, error="output_too_large"`.

- [ ] **Step 1: Write failing tests** (mark the Docker ones with the existing `real_docker` pattern, `RUN_DOCKER_TESTS=1`):
  - `test_returns_driver_output` — candidate `def f(): return 41`, driver writes `{"v": f()+1}` → `ok and output == {"v": 42}`.
  - `test_timeout_is_failure` — candidate loops forever, `timeout_s=3` → `not ok`, `error == "timeout"`, `elapsed_s < 6`.
  - `test_no_network_and_no_host_files` — candidate tries `urllib.request.urlopen("http://example.com")` and `open("/home")`; driver records exceptions → both raised inside sandbox.
  - `test_crash_reports_stderr_tail` — candidate raises `ValueError("boom")` → `not ok`, `"boom" in error`.
  - `test_output_too_large` — driver writes a 10 MiB JSON string → `not ok`, `error == "output_too_large"`.
- [ ] **Step 2:** `RUN_DOCKER_TESTS=1 uv run pytest tests/test_oe_pilot_sandbox.py -v` → FAIL (module missing).
- [ ] **Step 3:** Implement `run_candidate` in `sandbox_exec.py` (model on `scripts/orion3/ground_truth.run_in_container`, but through `DockerSandbox`).
- [ ] **Step 4:** Same command → PASS.
- [ ] **Step 5:** Commit `feat(oe-pilot): sandboxed candidate execution`.

### Task 3: Circle-packing task — working evaluator, candidate log, re-check

**Files:**
- Create: `experiments/openevolve/candidate_log.py`, `experiments/openevolve/static_check.py` (`forbidden_primitives(program_code: str) -> list[str]`, empty = clean), `experiments/openevolve/tasks/__init__.py`, `experiments/openevolve/tasks/circle_packing/__init__.py`, `.../circle_packing/working_evaluator.py`, `.../circle_packing/recheck.py`, `.../circle_packing/driver.py` (driver source as a string constant is fine)
- Test: `tests/test_oe_pilot_circle_packing.py`

**Interfaces:**
- Consumes: `run_candidate`, `SandboxResult` (Task 2); vendored upstream evaluator (Task 1).
- Produces:
  - `candidate_log.append_candidate(record: dict) -> None` — appends one JSON line to the path in env `OE_PILOT_CANDIDATE_LOG` (no-op if unset); record always gets `ts` (time.time()) and `code_sha256`.
  - `working_evaluator.evaluate(program_path: str) -> dict` — OpenEvolve evaluator entry point. Runs the candidate's entry function (the one the upstream evaluator calls, `run_packing()` in current upstream) in the sandbox (timeout 60 s), then applies the **upstream validation and scoring unchanged** (import the upstream functions; do not copy-edit their tolerances). Returns upstream's metric dict incl. `combined_score`; any failure → `{"combined_score": 0.0, "validity": 0.0, "error": ...}`. Logs `{task, program_path, code, metrics, sandbox_ok, output}` via `append_candidate`.
  - `recheck.recheck_packing(program_code: str) -> dict` with keys `valid: bool`, `score: float` (recomputed sum of radii; 0.0 if invalid), `reasons: list[str]`.
  - Re-check rules (exact values): exactly 26 circles; all coordinates/radii finite; every radius > 0; containment `x - r >= -1e-12`, `x + r <= 1 + 1e-12` (same for y); no overlap `dist(ci, cj) >= ri + rj - 1e-12`; two sandbox runs give identical output; AST of the program contains none of `os, sys, subprocess, socket, inspect, importlib, builtins, ctypes, pathlib, shutil` imports and no calls to `open, eval, exec, compile, __import__, getattr, setattr, globals, locals, vars`; reported sum ignored.

- [ ] **Step 1: Write failing tests:**
  - `test_seed_program_scores_like_upstream` — upstream initial program → `combined_score` equals what upstream's own `evaluate` returns on the same file (run upstream evaluator on host in the test; it is our own vendored seed, not untrusted) within 1e-9.
  - `test_recheck_rejects_tiny_overlap` — a program returning two circles overlapping by 1e-9 plus 24 valid ones → `valid is False`, `"overlap" in reasons[0]`.
  - `test_recheck_ignores_reported_sum` — program whose returned `sum_radii` is 9.9 but radii sum to 2.0 → `score == pytest.approx(2.0)`.
  - `test_recheck_flags_nondeterminism` — program using `random.random()` without seed in radii → `valid is False`.
  - `test_recheck_flags_forbidden_import` — `import os` in program → `valid is False`.
  - `test_nan_and_wrong_count_score_zero` — NaN radius / 25 circles → working `combined_score == 0.0`, re-check invalid, no exception.
  - `test_candidate_log_written` — set `OE_PILOT_CANDIDATE_LOG` to tmp file, evaluate seed → one JSON line with `code_sha256` and `metrics`.
- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3:** Implement the three modules.
- [ ] **Step 4:** `RUN_DOCKER_TESTS=1 uv run --extra evolve pytest tests/test_oe_pilot_circle_packing.py -v` → PASS.
- [ ] **Step 5:** Commit `feat(oe-pilot): circle-packing working evaluator and independent re-check`.

### Task 4: kNN task — seed program, data, working evaluator, re-check

**Files:**
- Create: `experiments/openevolve/tasks/knn/__init__.py`, `.../knn/initial_program.py`, `.../knn/data.py`, `.../knn/working_evaluator.py`, `.../knn/recheck.py`, `.../knn/driver.py`
- Test: `tests/test_oe_pilot_knn.py`

**Interfaces:**
- Consumes: Tasks 2–3 (`run_candidate`, `append_candidate`); `antigravity_hybrid_orchestrator.integrity.oracle.check_neighbors` (takes Python lists).
- Produces:
  - `initial_program.py`: module docstring = the retrieval section of `scripts/orion3/PUBLIC_SPEC.md`; `exact_knn` outside the block; `approximate_knn(vectors, queries, k)` inside `# EVOLVE-BLOCK-START/END`, initially numpy brute force.
  - `data.Challenge = TypedDict("Challenge", {"vectors": list, "queries": list, "k": int})`
  - `data.dev_sets() -> list[Challenge]`: 3 sets, uniform `[0,1)^d`, `(N, d, Q, k)` = `(20000, 32, 200, 10)`, `(10000, 64, 200, 10)`, `(30000, 16, 200, 5)`, numpy seeds 11, 12, 13.
  - `data.fresh_challenges(seed: int, count: int = 6) -> list[Challenge]`: N∈[2000,20000], d∈[2,64], k∈[1,10], Q=50, uniform; each set's queries include 5 exact copies of indexed vectors.
  - `data.holdout_sets() -> list[Challenge]`: 3 sets of Gaussian blobs (20 centres, σ=0.05) with `(N, d, Q, k)` = `(25000, 24, 200, 10)`, `(15000, 48, 200, 8)`, `(40000, 12, 200, 10)`, seeds 9001–9003. Lives only here; never imported by working evaluator or prompts (test enforces).
  - `SPEED_GATE = 0.5` — candidate time / brute-force time, both best of 3 repeats, timed inside the same sandbox call by the driver.
  - `working_evaluator.evaluate(program_path) -> dict`: per dev set recall@k vs exact ids (shape/distinct/int checks only; distances **not** checked — this is the typical evaluator). `combined_score = mean_recall if max_ratio <= SPEED_GATE else 0.1 * mean_recall`; also returns `recall`, `speed_ratio`. Errors → `combined_score 0.0`. Logs via `append_candidate`.
  - `recheck.recheck_knn(program_code: str) -> dict` keys `valid`, `score` (holdout mean recall if valid and holdout `speed_ratio <= SPEED_GATE`, else 0.0), `reasons`, `holdout_recall`, `holdout_speed_ratio`. Validity = `check_neighbors(...)` empty on every fresh challenge (seed = `int(code_sha256[:8], 16)`) AND exact-copy queries return that index first at distance 0 AND `static_check.forbidden_primitives(program_code) == []` (Task 3).
- [ ] **Step 1: Write failing tests:**
  - `test_seed_is_valid_but_slow` — seed: working `recall == 1.0`, `combined_score == pytest.approx(0.1)`; re-check `valid` True, `score == 0.0` (fails speed gate).
  - `test_euclidean_distances_pass_working_fail_recheck` — candidate returning `sqrt` distances with correct ids → working recall 1.0, re-check `valid` False with `"distance not derived"` in reasons.
  - `test_query_independent_result_scores_low` — returns ids `0..k-1` for every query → working recall < 0.05.
  - `test_ivf_reference_passes_both` — copy the IVF `approximate_knn` from `scripts/orion3/sources.py` → re-check `valid` True and `holdout_recall > 0.5`.
  - `test_holdout_not_reachable_from_working_path` — `working_evaluator` and `initial_program` source text contain neither `holdout` nor `9001`.
  - `test_numpy_int_ids_accepted` — ids as `np.int64` arrays are converted with `.tolist()` before `check_neighbors` → valid.
- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** `RUN_DOCKER_TESTS=1 uv run --extra evolve pytest tests/test_oe_pilot_knn.py -v` → PASS.
- [ ] **Step 5:** Commit `feat(oe-pilot): kNN task with speed-gated working evaluator and RIE re-check`.

### Task 5: LLM router (Gemma / Argon, seeded, logged)

**Files:**
- Create: `experiments/openevolve/router.py`
- Test: `tests/test_oe_pilot_router.py`

**Interfaces:**
- Consumes: `antigravity_hybrid_orchestrator.architect.client.CloudArchitectClient` (`backend="agy"`, `model="gemini-4-argon-medium"`, `.complete(prompt) -> (text, in_tokens, out_tokens)`).
- Produces:
  - `choose_model(seed: int, call_index: int, cloud_share: float) -> Literal["gemma", "argon"]` — `random.Random(f"{seed}:{call_index}").random() < cloud_share` → argon.
  - `class Router: __init__(self, seed: int, cloud_share: float, log_path: Path, *, gemma: Callable[[list[dict]], tuple[str, int, int]], argon: Callable[[list[dict]], tuple[str, int, int]], start_index: int = 0)`; `.complete(messages: list[dict]) -> str` (never raises: backend exception → returns `""`, status `error:<type>`).
  - `serve(router: Router, port: int) -> ThreadingHTTPServer` — stdlib; `POST /v1/chat/completions` returns OpenAI-shaped JSON (`choices[0].message.content`, `usage`); handles one request at a time (lock).
  - `ollama_gemma(base_url="http://localhost:11434", model="gemma4:12b", num_predict=2048, timeout_s=900)` → callable using Ollama native `/api/chat` with `think: false`, `options.num_predict`, `stream: false`; records `done_reason=="length"` as status `truncated` (text still returned).
  - `agy_argon(model="gemini-4-argon-medium")` → callable flattening messages to one prompt (`"{role}:\n{content}"` blocks) and calling `.complete`.
  - Log line per call: `call_index, model, ts_start, latency_s, prompt_tokens, completion_tokens, status (ok|truncated|empty|error:<T>), response_sha256, response`.
- [ ] **Step 1: Write failing tests** (fake backends, no network):
  - `test_choose_model_is_deterministic_and_near_share` — same seed → same sequence; seed 101, 1000 calls, share 0.1 → argon count in [70, 130].
  - `test_share_zero_never_calls_argon` — argon fake raises if called; 100 calls OK.
  - `test_backend_error_returns_empty_and_logs` — argon fake raises `RuntimeError` → `complete()` returns `""`, log status `error:RuntimeError`, call_index advances.
  - `test_http_roundtrip_openai_shape` — `serve` on port 0, POST with `openai`-style body via `urllib` → `choices[0]["message"]["content"] == "stub"`.
  - `test_resume_start_index` — `start_index=37` → first log `call_index == 37` and model equals `choose_model(seed, 37, share)`.
- [ ] **Step 2:** Run `uv run pytest tests/test_oe_pilot_router.py -v` → FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Same → PASS.
- [ ] **Step 5:** Commit `feat(oe-pilot): seeded logging LLM router for gemma and argon`.

### Task 6: Run driver for arms L / LC / B, configs, resume

**Files:**
- Create: `experiments/openevolve/configs/circle_packing.yaml`, `experiments/openevolve/configs/knn.yaml`, `experiments/openevolve/best_of_n.py`, `experiments/openevolve/run_pilot.py`
- Test: `tests/test_oe_pilot_run.py`

**Interfaces:**
- Consumes: Tasks 1–5.
- Produces:
  - Configs: start from vendored upstream `config.yaml`; set `max_iterations: 100`, `diff_based_evolution: true`, one LLM model `{name: "pilot-router", api_base: "http://127.0.0.1:<port>/v1", api_key: "unused", weight: 1.0}`, `llm.retries: 0`, `llm.timeout: 900`, `evaluator.parallel_evaluations: 1`, `evaluator.timeout: 180`, `checkpoint_interval: 5`, LLM feedback off, `random_seed` overwritten per run. Key names as in 0.4.0's default config (verify against the installed package).
  - `best_of_n.run_best_of_n(task: str, router_url: str, n: int, out_dir: Path, seed: int, resume: bool) -> None` — each attempt: the 0.4.0 prompt for diff evolution built from the seed program with **empty** history/top programs/inspirations (use the package's prompt sampler, names per Task 1 `NOTICE.md`); apply diff with `extract_diffs`/`apply_diff`; write candidate file; call the task's `working_evaluator.evaluate`. No diff → no evaluation, logged as failed attempt. Resume skips attempts already present in `calls.jsonl`.
  - `run_pilot.py` CLI: `--task {circle_packing,knn} --arm {L,LC,B} --seed INT --out DIR [--resume] [--llm {real,stub}]`. Writes into `DIR`: `calls.jsonl` (router), `candidates.jsonl` (env `OE_PILOT_CANDIDATE_LOG`), `openevolve/` (checkpoints), `manifest.json` (`git_sha`, `openevolve_version`, config SHA-256, models, `num_predict`, start/end, `gpu_wall_s`, `status`). Cloud share: L=0.0, LC=0.1, B=0.0. Resume: `start_index = len(calls.jsonl)`, OpenEvolve restarts from the latest checkpoint; remaining iterations = 100 − completed iterations in checkpoint. `--llm stub` = a router whose both backends return a fixed valid diff for the task (for tests).
  - Run dir naming: `benchmarks/openevolve_pilot/{task}_{arm}_s{seed}/`.
- [ ] **Step 1: Write failing tests** (stub LLM, real Docker, `RUN_DOCKER_TESTS=1`):
  - `test_stub_run_L_circle_packing_5_iterations` — with `max_iterations` overridden to 5 via env `OE_PILOT_MAX_ITER=5` → `calls.jsonl` has 5 lines, `candidates.jsonl` ≥ 1 line, `manifest.status == "complete"`.
  - `test_best_of_n_uses_seed_only` — B with n=3 → every prompt sent contains the seed program text and no other candidate's code.
  - `test_resume_does_not_respend_calls` — run 3 iterations, kill, resume with total 5 → `calls.jsonl` has exactly 5 lines with indexes 0..4.
  - `test_no_diff_reply_counts_as_failed_attempt` — stub returns prose without a diff → B run with n=2 completes, 2 calls, 0 candidates, manifest counts `failed_attempts == 2`.
- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3:** Implement configs, `best_of_n.py`, `run_pilot.py`.
- [ ] **Step 4:** `RUN_DOCKER_TESTS=1 uv run --extra evolve pytest tests/test_oe_pilot_run.py -v` → PASS.
- [ ] **Step 5:** Commit `feat(oe-pilot): run driver for arms L/LC/B with resume`.

### Task 7: Post-hoc re-check and analysis

**Files:**
- Create: `experiments/openevolve/recheck_all.py`, `experiments/openevolve/analyze.py`
- Test: `tests/test_oe_pilot_analysis.py`

**Interfaces:**
- Consumes: run directories from Task 6; `recheck_packing`, `recheck_knn`.
- Produces:
  - `attribute(calls: list[dict], candidates: list[dict]) -> list[dict]` — each candidate gets the `call_index` and `model` of the latest call with `ts_start <= candidate.ts` (valid because evaluation is serialised).
  - `recheck_run(run_dir: Path) -> dict` → writes `rechecked.jsonl` (candidate + re-check result, cached by `code_sha256`) and returns run summary: `seed_score`, `shipped_working_score`, `shipped_verified_score` (re-check score of max-working candidate; seed's verified score if invalid), `best_verified_score`, `above_seed_count`, `false_improvements` (above-seed working score but re-check invalid or verified ≤ seed verified), `false_improvement_rate`, `shipped_is_false`, `curve` (list of `(call_index, best_working, shipped_verified_so_far)`), `argon_calls`, `argon_tokens`, `argon_wall_s`, `gemma_wall_s`, `failed_calls_by_status`.
  - `analyze.main(root: Path) -> None` → `benchmarks/openevolve_pilot/summary.json` (per task × arm: median/min/max of the primary and secondary endpoints, false-improvement rates, cloud cost) and figures `docs/openevolve_pilot/fig_curves_{task}.png`, `fig_false_improvements.png` (use `dataviz` conventions already used in `docs/paper/figures`).
- [ ] **Step 1: Write failing tests** on synthetic run dirs (no Docker; inject a fake `recheck` callable):
  - `test_attribution_by_timestamp` — calls at t=1,5,9; candidates at t=2,6,10 → indexes 0,1,2.
  - `test_shipped_false_falls_back_to_seed` — best-working candidate re-checks invalid → `shipped_verified_score == seed_score`, `shipped_is_false is True`.
  - `test_false_improvement_rate` — 4 above-seed candidates, 1 invalid → rate 0.25.
  - `test_recheck_cache_by_hash` — two candidates with the same code → fake recheck called once.
- [ ] **Step 2:** Run → FAIL. **Step 3:** Implement. **Step 4:** `uv run pytest tests/test_oe_pilot_analysis.py -v` → PASS.
- [ ] **Step 5:** Commit `feat(oe-pilot): post-hoc re-check, attribution and analysis`.

### Task 8: Campaign script and Argon cost probe (no GPU needed)

**Files:**
- Create: `scripts/run_openevolve_pilot.sh`, `experiments/openevolve/argon_probe.py`
- Test: `tests/test_oe_pilot_campaign_script.py`

**Interfaces:**
- Produces:
  - `run_openevolve_pilot.sh {check|run|recheck}` — `check`: `ollama ps` empty except our model, `nvidia-smi` shows no other compute process, `docker image inspect antigravity-orion:local`, `agy` present; prints PASS/FAIL lines. `run`: pauses `archive-index-sessions.timer archive-index.timer` (as in `run_campaign_b.sh`), runs the 18 runs in order (all L and B runs first: tasks × seeds; then LC), each with `--resume` if its dir exists and is not `complete`; stops the whole campaign if a run's manifest ends `status == "paused_quota"`; restores timers; logs to `benchmarks/openevolve_pilot/campaign.log`. `recheck`: runs `recheck_all` + `analyze`.
  - `run_pilot.py` sets `manifest.status = "paused_quota"` and exits non-zero when 3 consecutive Argon calls fail with errors mentioning quota/limit (case-insensitive `quota|limit|exhaust`).
  - `argon_probe.py --calls 3` — sends 3 real mutation-sized prompts (the circle-packing seed prompt from Task 6) through `agy_argon`, prints latency, tokens and reply length; owner reads quota % in `agy` before/after.
- [ ] **Step 1: Write failing test** `test_run_order_L_B_before_LC` — run the script with env `OE_PILOT_DRY_RUN=1` (prints the planned commands only) → the first 12 lines are L/B runs, last 6 are LC; and `test_quota_pause_stops_campaign` with a fake `run_pilot.py` that writes `paused_quota` on its first LC run → no further runs.
- [ ] **Step 2–4:** FAIL → implement → PASS (`uv run pytest tests/test_oe_pilot_campaign_script.py -v`).
- [ ] **Step 5:** Run `uv run --extra evolve python -m experiments.openevolve.argon_probe --calls 3` (owner notes `agy` quota % before/after). Record results in `docs/openevolve_pilot/spike_notes.md`. Commit `feat(oe-pilot): campaign script and argon probe`.

---

**GPU gate.** Tasks 9–12 start only after the owner says the GPU is free.

### Task 9: Spike with Gemma (throwaway data)

**Files:**
- Modify: `docs/openevolve_pilot/spike_notes.md`; possibly `experiments/openevolve/configs/*.yaml` (`num_predict`, prompt system message only)

- [ ] **Step 1:** `bash scripts/run_openevolve_pilot.sh check` → all PASS.
- [ ] **Step 2:** `OE_PILOT_MAX_ITER=10 uv run --extra evolve python -m experiments.openevolve.run_pilot --task circle_packing --arm L --seed 1 --out temp/oe_spike/cp` and the same for `--task knn`.
- [ ] **Step 3:** From `calls.jsonl`/`candidates.jsonl` record: share of replies with an applicable diff, `truncated` count, median s/call, median eval s, any crash. Decision rules: if `truncated` > 2/10 → set `num_predict` 4096 and repeat Step 2 once; if applicable-diff share < 3/10 → add the upstream example's explicit diff-format instructions to the system message and repeat once. Record the final values.
- [ ] **Step 4:** Delete `temp/oe_spike/` data from consideration (never analysed). Commit config changes + notes: `chore(oe-pilot): spike settings frozen`.

### Task 10: Pre-registration

**Files:**
- Create: `docs/openevolve_pilot/PREREGISTRATION.md`

- [ ] **Step 1:** Write it from the spec + spike notes: RQ1–RQ3; tasks, seeds, arms, budgets; exact versions (`openevolve` 0.4.0, Ollama version, `gemma4:12b` digest from `ollama list`, `gemini-4-argon-medium`, `agy --version`, image digest); frozen `num_predict` and prompts (config SHA-256s); re-check rules and tolerances; primary/secondary endpoints and descriptive analysis; deviation policy (quota pause, crashes, reruns logged, never deleted).
- [ ] **Step 2:** Commit `docs(oe-pilot): pre-registration` and tag `oe-pilot-prereg`. No campaign run before this commit.

### Task 11: Campaign

- [ ] **Step 1:** `setsid nohup bash scripts/run_openevolve_pilot.sh run &`; monitor `benchmarks/openevolve_pilot/campaign.log` (don't run other GPU/CPU-heavy jobs).
- [ ] **Step 2:** On `paused_quota`: wait for `agy` refresh, then re-launch the same command (resume). Record every pause/restart in `docs/openevolve_pilot/DEVIATIONS.md`.
- [ ] **Step 3:** When all 18 manifests are `complete`, commit raw data: `data(oe-pilot): campaign complete (18 runs)`.

### Task 12: Re-check, analysis, report

**Files:**
- Create: `docs/OPENEVOLVE_PILOT_RESULTS.md`

- [ ] **Step 1:** `bash scripts/run_openevolve_pilot.sh recheck` → `summary.json` and figures.
- [ ] **Step 2:** Spot-verify: for each arm × task, open the shipped program of one run and re-run its re-check by hand; numbers must match `rechecked.jsonl`.
- [ ] **Step 3:** Write the report: RQ1–RQ3 answers with medians/ranges, curves, false-improvement examples (with diffs), cloud cost, deviations, limitations (3 seeds, one local model, one cloud model, designed tasks). No claim beyond the data.
- [ ] **Step 4:** Commit `docs(oe-pilot): results`.
