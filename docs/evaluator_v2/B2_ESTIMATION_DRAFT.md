# B2 estimation study — registration draft

**Status:** draft only; no B2 model call has been made. This defines the lower-cost option for the requested B2 launch and requires an operational freeze before model calls. It does not replace the confirmatory design in `B2_DESIGN.md`.

## Why this is not the 715-CU study

The 715.5-CU figure is a planning scenario for 61 paired seeds, assuming 25% of pairs favor V2 only and 5% favor V1-matched only. That scenario targets at least 80% power for the exact paired-binary test, uses 40 model calls per pair, and applies B1's rough rate of 0.293 CU per call. Those discordance probabilities are illustrative assumptions, not estimates from B1.

This draft instead proposes a small **estimation/feasibility cohort**, with weaker claims and no confirmatory hypothesis test.

## Proposed frozen design

- **Question:** In this fixed approximate-kNN task and configuration, what selected-program validity outcomes and compute cost are observed for V1-matched and V2?
- **Pairs:** 6 preregistered task-seed pairs.
- **Calls:** 20 per arm per pair; 40 per pair; maximum 240 logged Gemma calls.
- **Conditions:** V1-matched and V2, same candidate sandbox, prompts, model digest, seed program, and call budget.
- **Order:** the current draft config fixes six seeds (`20261120`–`20261125`) and a balanced order (P1 V2-first, P2 V1-matched-first, P3 V2-first, P4 V1-matched-first, P5 V1-matched-first, P6 V2-first). This is saved in `experiments/openevolve/configs/b2_estimation_2026-11-20.json`; the protocol is still a draft until source, checker, holdout, and watchdog hashes are frozen.
- **Model:** local Ollama Gemma 4 12B, frozen by immutable digest; no cloud-model fallback.
- **Primary descriptive outcome:** per-arm count and proportion of selected artifacts that fail the independent holdout correctness check. Publish the full six-pair table and the paired 2×2 table; do not claim a statistically established treatment effect or report a confirmatory p-value.
- **Secondary descriptive outcomes:** selected holdout score, false-improvement rate with explicit candidate and hash denominators, gate rejection reasons, wall time, model tokens, GPU time, and Colab CU.
- **Selection rule:** use the official OpenEvolve selected artifact hash. Never substitute the best candidate-log score.
- **Independent check:** after both arms finish, download the selected artifacts and run the separately versioned checker offline. Do not expose the holdout dataset or seeds to the candidate-generation process. Record checker, holdout, program, and Docker image hashes.
- **Missingness:** a missing arm or selected artifact is retained as an incomplete pair. Report assigned pairs, completed pairs, and missing reasons; do not replace a failed pair with a new seed.
- **Stopping:** stop at 6 complete pairs, at 240 model calls, if aggregate account draw reaches 70.4 CU from the recorded launch snapshot, or if the global Colab balance reaches 100 CU, whichever comes first. Since the current B-12 L4 assignment shares the account balance, aggregate draw cannot be attributed exactly to B2; treat the cap conservatively and report concurrent usage. A budget stop leaves the cohort incomplete; it does not permit a new sample size or a confirmatory claim.

## Compute envelope

At B1's observed rough rate, `240 calls × 0.293 CU/call = 70.32 CU` (rounded planning estimate: 70.4 CU). The latest read-only balance check on 2026-11-20 was 208.49 CU; one pre-existing L4 assignment remained active, and the 100-CU reserve leaves 108.49 CU of total headroom. A 70.4-CU draw leaves 38.09 CU for concurrent usage and cost uncertainty. The no-call rehearsal itself used 0.16 CU by the account meter. The account meter is shared, so this is a pooled budget—not a precise B2-only cost. Re-read balance immediately before launch and stop if the projected budget no longer fits. The estimate is not a guarantee or a vendor quote.

## Required launch gates

1. Keep the existing B-12 project and its active `b12eval` L4 assignment untouched. The e2-micro read-only check currently reports both `b12-orchestrator.service` and the Paper II B1 watchdog as inactive; the account CLI still reports one active L4 assignment. Do not infer that this assignment is B2 or try to attach to it.
2. Commit the exact six-seed list, randomized arm order, protocol JSON, analysis script, and checker/holdout hashes before any model call.
3. Implement a B2-specific runner and watchdog with a separate Colab session name, GCS prefix, state file, and budget counter. Do not reuse B1's hard-coded `paper2-v2/b1` prefix or the active B-12 session.
4. Run the B2 watchdog and persistence path in a fresh Colab runtime with model calls disabled. The GCS sentinel upload, stop/recreate/restore, and terminal-state roundtrip passed; the B2 session was stopped afterward. The runner unit test also verifies resuming only the incomplete arm. A detached systemd/laptop-disconnect test remains outstanding.
5. Re-check global balance and ensure the 100-CU reserve can be maintained even if the existing active assignment continues.
6. Only then freeze the preregistration SHA and start the 6-pair estimation cohort. If any gate fails, preserve the draft and run no model calls.

## Implementation checkpoint — 2026-11-20

- Added the resumable B2 runner, isolated watchdog, Colab entrypoint, and systemd unit template. The runner refuses live calls while this JSON remains `draft`; its dry-run reports 240 assigned calls and zero model calls.
- Run manifests must match the pinned source Git SHA, protocol SHA, model digest, evaluator version, seed, and exact 20-call ledger. Resume skips validated completed arms and continues only a matching incomplete arm.
- The watchdog uses only `paper2-b2-estimation` and `paper2-v2/b2-estimation-2026-11-20`; account-wide guard is 70.4 CU from a persisted fresh start snapshot or 100 CU remaining, whichever comes first.
- Local focused verification: 23 B2 tests passed; Python compilation, whitespace check, and watchdog dry-run passed. No Gemma/Argon calls were made.
- No-call GCS persistence rehearsal passed on 2026-11-20 with source `34d663e2992876a00c179e79adbbc82886797a32`; evidence is in `B2_NO_CALL_REHEARSAL.json`. The Colab runtime was stopped and recreated, the sentinel restored from the dedicated prefix, the synthetic terminal-state record round-tripped, and the B2 session was stopped. The measured account draw was 0.16 CU; the pre-existing L4 assignment remained untouched.
- Still not launch-ready: the protocol is draft; independent checker/holdout scope and hashes, frozen source/protocol registration, detached systemd/laptop-disconnect test, service installation, and immediate pre-run balance gate remain outstanding. No B2 model call has been made.

## Interpretation boundary

With six pairs, results are descriptive and intended to estimate operational feasibility, timing variability, and preliminary paired outcome frequencies. They cannot establish V2 efficacy, equivalence, generalization beyond this task/configuration, or the 80%-power claim associated with the 715.5-CU scenario.
