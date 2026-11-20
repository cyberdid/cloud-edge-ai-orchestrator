# B2 estimation study — frozen registration and launch gates

**Status:** frozen estimation cohort is running independently on the e2-micro VM. The B2 systemd service is enabled; systemd currently reports `activating` while its supervisor is running, and the dedicated Colab session is busy. The current run state is recorded in `B2_LIVE_STATUS_2026-11-20.json`. This defines the lower-cost, estimation-only option and does not replace the confirmatory design in `B2_DESIGN.md`.

- Frozen protocol SHA-256: `8f990fd64458c8060c79d71655a9712eb9f9f5963a653482e96452ff1c762719`
- Checker fingerprint SHA-256: `6eeeda09a311cfdfde83f2aac519037e606382db6580430193376b6831e53e08`
- Independent re-check image: `antigravity-orion:local` (`sha256:aae973535ab734c026d878c0ef1d7ec1164b83538bdcd63c192476672cb27ae2`)
- Holdout: fixed source-visible seeds 9001–9003; reproducible independent re-check, not a secret or sealed holdout.

## Why this is not the 715-CU study

The 715.5-CU figure is a planning scenario for 61 paired seeds, assuming 25% of pairs favor V2 only and 5% favor V1-matched only. That scenario targets at least 80% power for the exact paired-binary test, uses 40 model calls per pair, and applies B1's rough rate of 0.293 CU per call. Those discordance probabilities are illustrative assumptions, not estimates from B1.

The frozen protocol instead runs a small **estimation/feasibility cohort**, with descriptive outcomes and no confirmatory hypothesis test.

## Proposed frozen design

- **Question:** In this fixed approximate-kNN task and configuration, what selected-program validity outcomes and compute cost are observed for V1-matched and V2?
- **Pairs:** 6 preregistered task-seed pairs.
- **Calls:** 20 per arm per pair; 40 per pair; maximum 240 logged Gemma calls.
- **Conditions:** V1-matched and V2, same candidate sandbox, prompts, model digest, seed program, and call budget.
- **Order:** the frozen config fixes six seeds (`20261120`–`20261125`) and balanced order (P1 V2-first, P2 V1-matched-first, P3 V2-first, P4 V1-matched-first, P5 V1-matched-first, P6 V2-first). The protocol, checker, visible holdout, and re-check image are fingerprinted in `experiments/openevolve/configs/b2_estimation_2026-11-20.json`.
- **Model:** local Ollama Gemma 4 12B, frozen by immutable digest; no cloud-model fallback.
- **Primary descriptive outcome:** per-arm count and proportion of selected artifacts that fail the independent holdout correctness check. Publish the full six-pair table and the paired 2×2 table; do not claim a statistically established treatment effect or report a confirmatory p-value.
- **Secondary descriptive outcomes:** selected holdout score, false-improvement rate with explicit candidate and hash denominators, gate rejection reasons, wall time, model tokens, GPU time, and Colab CU.
- **Selection rule:** use the official OpenEvolve selected artifact hash. Never substitute the best candidate-log score.
- **Independent check and summary:** after both arms finish, `experiments/openevolve/recheck_b2_selected.py` loads only each arm's official selected artifact, verifies its recorded SHA-256 and run manifest, and invokes the existing independent kNN checker. Missing or interrupted runs/selections stay in the assigned denominator as invalid/incomplete; they are not replaced. `experiments/openevolve/analyze_b2_estimation.py` produces descriptive rates and the paired table only; it emits no inferential p-value. The three holdout set seeds are committed in checker source and are not passed to the candidate prompt or working evaluator. They are source-visible, so this is a reproducible independent re-check, **not a secret/sealed holdout**; do not use this cohort for confirmatory claims.
- **Missingness:** a missing arm or selected artifact is retained as an incomplete pair. Report assigned pairs, completed pairs, and missing reasons; do not replace a failed pair with a new seed.
- **Stopping:** stop at 6 complete pairs, at 240 model calls, if aggregate account draw reaches 70.4 CU from the recorded launch snapshot, or if the global Colab balance reaches 100 CU, whichever comes first. Since the current B-12 L4 assignment shares the account balance, aggregate draw cannot be attributed exactly to B2; treat the cap conservatively and report concurrent usage. A budget stop leaves the cohort incomplete; it does not permit a new sample size or a confirmatory claim.

## Compute envelope

At B1's observed rough rate, `240 calls × 0.293 CU/call = 70.32 CU` (rounded planning estimate: 70.4 CU). The actual launch snapshot was 206.93 CU with one pre-existing assignment active; the 70.4-CU cap therefore left 36.53 CU above the 100-CU reserve for concurrent account use and estimation error. The latest snapshot in `B2_LIVE_STATUS_2026-11-20.json` is 202.90 CU with two active assignments, an observed pooled change of 4.03 CU since launch. The account meter is shared, so no exact B2-only cost can be inferred. The estimate is not a guarantee or a vendor quote.

## Required launch gates

1. Keep the existing B-12 project and its active `b12eval` L4 assignment untouched. The e2-micro read-only check currently reports both `b12-orchestrator.service` and the Paper II B1 watchdog as inactive; the account CLI still reports one active L4 assignment. Do not infer that this assignment is B2 or try to attach to it.
2. **Completed in the frozen protocol:** exact six-seed list, balanced arm order, protocol JSON, selected-artifact re-check script, checker component fingerprint, source-visible holdout definition, and Docker image digest.
3. Implement a B2-specific runner and watchdog with a separate Colab session name, GCS prefix, state file, and budget counter. Do not reuse B1's hard-coded `paper2-v2/b1` prefix or the active B-12 session.
4. **Passed for exact frozen hashes:** the detached systemd no-call rehearsal recreated the Colab runtime, restored the GCS sentinel, and round-tripped terminal state; the B2 session was stopped afterward. Evidence is in `B2_FROZEN_NO_CALL_REHEARSAL.json`.
5. **Passed at launch:** balance was 206.93 CU with one other assignment; the watchdog enforces the 70.4-CU aggregate draw cap and 100-CU floor.
6. **Started:** the persistent systemd unit was enabled and launched the six-pair cohort. It runs independently of the laptop and resumes from its dedicated GCS checkpoints. Do not stop the service or B2 session while it is running; inspect the GCS state and systemd unit for progress.

## Implementation checkpoint — 2026-11-20

- Added the resumable B2 runner, isolated watchdog, Colab entrypoint, and systemd unit template. The runner refuses live calls unless this JSON is `frozen`; the dry-run reports 240 assigned calls and zero model calls.
- Run manifests must match the pinned source Git SHA, protocol SHA, model digest, evaluator version, seed, and exact 20-call ledger. Resume skips validated completed arms and continues only a matching incomplete arm.
- The watchdog uses only `paper2-b2-estimation` and `paper2-v2/b2-estimation-2026-11-20`; account-wide guard is 70.4 CU from a persisted fresh start snapshot or 100 CU remaining, whichever comes first.
- Local focused verification: 35 B2 tests passed; Python compilation, whitespace check, runner/watchdog no-call dry-runs, and systemd unit contract checks passed. The selected-artifact re-check retains interrupted runs as incomplete, and the summary is descriptive-only with no p-value.
- Exact-hash no-call GCS rehearsal passed on 2026-11-20 with source `3e603e426930215838ce9a2aa475959bb241ff87`; evidence is in `B2_FROZEN_NO_CALL_REHEARSAL.json`. The account meter's immediate reading lagged its follow-up by 0.33 pooled CU, which is not attributable to B2 alone.
- Live launch started on the e2-micro VM with launch balance 206.93 CU. The P1 pair is complete (2 of 12 arms): each arm has 20 ledger entries and an official selected artifact; both selection files exist and their byte-level SHA-256 matches the same manifest hash, `40133bf46bf9146fbe14c962ff6983ffd6f40faf7b2c0e5f9ab2183ffd6c9619`. P1 V2 has 18 candidate records (12 working-evaluator valid, six invalid); P1 V1-matched has 10 candidate records, with no final validity values present in their current records. P1 V1-matched had 15 `ok` and five `truncated` calls; P1 V2 had 16 `ok` and four `truncated` calls. All nine truncations reached the configured 2048 completion-token limit. P2 V1-matched is running with four `ok` calls and five candidate records; candidate validity has not yet been recorded. The frozen configuration remains unchanged. Independent re-check remains scheduled after all arms finish.
- The current main branch also contains a systemd unit fix placing the restart limiter in the correct section; it was reloaded without interrupting the active run. The execution source remains pinned to `3e603e426930215838ce9a2aa475959bb241ff87`.

## Interpretation boundary

With six pairs, results are descriptive and intended to estimate operational feasibility, timing variability, and preliminary paired outcome frequencies. They cannot establish V2 efficacy, equivalence, generalization beyond this task/configuration, or the 80%-power claim associated with the 715.5-CU scenario.
