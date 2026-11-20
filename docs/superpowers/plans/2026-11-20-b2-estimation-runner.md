# B2 Estimation Runner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Run the budget-capped six-pair B2 estimation cohort from an isolated Colab L4 session that persists through laptop shutdown and never touches B1 or B-12 state.

**Architecture:** A frozen protocol JSON supplies six task seeds and balanced arm order. A Colab runner executes 20 logged calls for each `v1_matched`/`v2` arm, mirrors state and immutable run folders to a dedicated GCS prefix, and resumes only matching partial runs. A separate e2-micro systemd watchdog owns a distinct named Colab session and enforces the aggregate account-draw cap and 100-CU reserve. Independent holdout checks run offline after selected program hashes are frozen.

**Tech Stack:** Python 3.12, existing `experiments.openevolve.run_pilot.run`, Ollama Gemma 4 12B digest `6114515d63c1`, Colab CLI 0.7.5, Google Cloud Storage, systemd, pytest, Docker.

**Spec:** `docs/evaluator_v2/B2_ESTIMATION_DRAFT.md` and `docs/evaluator_v2/B2_DESIGN.md`.

## Global Constraints

- Never write B2 artifacts under `paper2-v2/b1/` or the B-12 study prefix.
- Never stop, resume, or overwrite the existing B-12 service or its Colab session.
- Maximum six pairs, 20 calls per arm, 240 model calls; no interim outcome-based sample-size changes.
- Stop at 70.4 CU aggregate account draw from the launch snapshot or 100 CU remaining, whichever comes first.
- No cloud-model fallback; preserve every call index and selected-program SHA.
- Label the cohort estimation/feasibility only; no confirmatory p-value or efficacy claim.
- Before model calls, pass a no-call Colab/e2-micro durability rehearsal and freeze the exact protocol/source hashes.

---

### Task 1: Freeze and validate the estimation protocol

**Files:**
- Create: `experiments/openevolve/configs/b2_estimation_2026-11-20.json`
- Create: `experiments/openevolve/colab/b2_protocol.py`
- Test: `tests/test_oe_b2_protocol.py`
- Modify: `docs/evaluator_v2/B2_ESTIMATION_DRAFT.md`

**Interfaces:** `load_protocol(path) -> dict` validates exactly six unique seeds, exactly three `v1_matched`-first and three `v2`-first pairs, 20 calls per arm, task `knn`, model digest `6114515d63c1`, a unique B2 GCS prefix, and the 70.4-CU / 100-CU limits. Invalid values raise `ValueError` before bootstrap or model access.

- [ ] Write tests for valid fixed protocol and rejection of duplicate seeds, unbalanced arm order, wrong call cap, B1 prefix, and missing model digest.
- [ ] Run those tests to confirm validation fails because `load_protocol` is absent.
- [ ] Add the exact six-seed allocation and minimal validator.
- [ ] Run focused tests and record the protocol SHA-256.

### Task 2: Implement resumable B2 Colab execution

**Files:**
- Create: `experiments/openevolve/colab/paper2_b2_estimation.py`
- Test: `tests/test_oe_b2_estimation_runner.py`

**Interfaces:** `run_cohort(protocol, *, llm, root, mirror, run_one)` runs pair arms in the frozen order, names each run `knn_{evaluator}_s{seed}`, resumes only a matching protocol/evaluator/model manifest, checkpoints `state.json`, and refuses a complete or mismatched prefix. `llm="stub"` is dry-run only and must never call Ollama. The production entrypoint bootstraps the pinned model and passes `llm="real"` only after the watchdog invokes it for the frozen protocol.

- [ ] Write tests proving distinct GCS prefix, fixed order, call cap propagation, manifest hash mismatch refusal, terminal-state protection, and zero model calls for stub mode.
- [ ] Run tests to observe the expected missing-runner failures.
- [ ] Implement sequential paired runs using the existing `run_pilot.run(..., evaluator_version="v1_matched"|"v2")`, `GcsMirror`, official selection hashes, and atomic state replacement.
- [ ] Run focused tests with stub replies and small temporary directories; inspect that no B1 path or Ollama request is touched.

### Task 3: Add a separate e2-micro watchdog and service

**Files:**
- Create: `experiments/openevolve/colab/paper2_b2_watchdog.py`
- Create: `deploy/systemd/paper2-b2-estimation.service`
- Test: `tests/test_oe_b2_watchdog.py`

**Interfaces:** watchdog configuration uses service name `paper2-b2-estimation`, Colab session `paper2-b2-estimation`, and the protocol's B2-only GCS prefix. It pins a full Git SHA, starts no model calls in `--dry-run`, polls account usage, stops its own B2 session at either cap, and never issues stop/exec against the B-12 session. Reconciliation marks completion only when all 12 manifests and call ledgers match the frozen protocol and every ledger has unique indices `0..19`.

- [ ] Write tests for separate session/prefix, B-12 session isolation, CU stopping, incomplete/malformed ledgers, and terminal-state reconciliation.
- [ ] Run tests to confirm expected failures before implementation.
- [ ] Implement the isolated watchdog with atomic local control state, append-only GCS run records, and bounded Colab command timeouts.
- [ ] Run the service locally in dry-run mode and inspect that it never invokes a model or changes cloud state.

### Task 4: Deploy and pass the no-call Colab durability rehearsal

**Files:**
- Modify remotely: `/etc/systemd/system/paper2-b2-estimation.service` on `paper2-colab-watchdog` only.
- Create remotely: `/opt/paper2-b2/` control directory and a dedicated B2 session/config.
- Evidence: `docs/evaluator_v2/B2_NO_CALL_REHEARSAL.json`.

- [ ] Pin the committed source SHA, create a separate B2 service without stopping or editing `b12-orchestrator.service`, and verify the unit diff before starting it.
- [ ] Start a fresh B2 Colab runtime with `--dry-run`; verify upload/restore of a sentinel checkpoint, runner resume after a watchdog restart, unchanged call count, and completion reconciliation.
- [ ] Stop only the B2 rehearsal Colab session; record CU balance before and after, service logs, GCS prefix, SHA values, and all pass/fail checks.
- [ ] Proceed to model calls only if the rehearsal passes and the shared account remains above the 100-CU reserve with the 70.4-CU aggregate cap still available.

### Task 5: Freeze and launch the six-pair estimation cohort

**Files:**
- Create: `docs/evaluator_v2/B2_ESTIMATION_REGISTRATION.json`
- Create: `docs/evaluator_v2/B2_ESTIMATION_RESULTS.md`
- Modify: `docs/evaluator_v2/B2_ESTIMATION_DRAFT.md`

- [ ] After rehearsal, freeze protocol, source, evaluator, Docker image, independent checker, holdout generator/seeds, and analysis hashes; commit before the first Gemma call.
- [ ] Capture the pre-run balance and start the B2-specific service without contacting the existing Colab assignment.
- [ ] Monitor only B2 state and aggregate CU; preserve the 100-CU reserve and stop under the frozen rules.
- [ ] After all assigned runs or a budget stop, independently re-check selected programs offline, publish the full assigned-pair inventory, and update results with descriptive estimates only.
