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
- **Order:** pair order is fixed before the first model call; three pairs run V1-matched first and three V2 first. The exact allocation and seed list must be generated, committed, and hashed before launch.
- **Model:** local Ollama Gemma 4 12B, frozen by immutable digest; no cloud-model fallback.
- **Primary descriptive outcome:** per-arm count and proportion of selected artifacts that fail the independent holdout correctness check. Publish the full six-pair table and the paired 2×2 table; do not claim a statistically established treatment effect or report a confirmatory p-value.
- **Secondary descriptive outcomes:** selected holdout score, false-improvement rate with explicit candidate and hash denominators, gate rejection reasons, wall time, model tokens, GPU time, and Colab CU.
- **Selection rule:** use the official OpenEvolve selected artifact hash. Never substitute the best candidate-log score.
- **Independent check:** after both arms finish, download the selected artifacts and run the separately versioned checker offline. Do not expose the holdout dataset or seeds to the candidate-generation process. Record checker, holdout, program, and Docker image hashes.
- **Missingness:** a missing arm or selected artifact is retained as an incomplete pair. Report assigned pairs, completed pairs, and missing reasons; do not replace a failed pair with a new seed.
- **Stopping:** stop at 6 complete pairs, at 240 model calls, if aggregate account draw reaches 70.4 CU from the recorded launch snapshot, or if the global Colab balance reaches 100 CU, whichever comes first. Since the current B-12 L4 assignment shares the account balance, aggregate draw cannot be attributed exactly to B2; treat the cap conservatively and report concurrent usage. A budget stop leaves the cohort incomplete; it does not permit a new sample size or a confirmatory claim.

## Compute envelope

At B1's observed rough rate, `240 calls × 0.293 CU/call = 70.32 CU` (rounded planning estimate: 70.4 CU). Account balance observed on 2026-11-20 was 209.95 CU; one L4 assignment was active, and the 100-CU reserve leaves 109.95 CU of total headroom. A 70.4-CU draw leaves 39.55 CU for concurrent B-12 usage, startup/rehearsal, and cost uncertainty. The account meter is shared, so this is a pooled budget—not a precise B2-only cost. Re-read balance immediately before launch and stop if the projected budget no longer fits. The estimate is not a guarantee or a vendor quote.

## Required launch gates

1. Keep the existing B-12 orchestrator and its active L4 assignment untouched. The account CLI shows one active assignment, while the local CLI could not resolve its session name; treat its CU use as concurrent pooled consumption.
2. Commit the exact six-seed list, randomized arm order, protocol JSON, analysis script, and checker/holdout hashes before any model call.
3. Implement a B2-specific runner and watchdog with a separate Colab session name, GCS prefix, state file, and budget counter. Do not reuse B1's hard-coded `paper2-v2/b1` prefix or the active B-12 session.
4. Run the B2 watchdog and persistence path in a fresh Colab runtime with model calls disabled. Verify checkpoint upload, laptop-disconnect recovery, terminal-state behavior, and no duplicate call indices. Stop that rehearsal session after validation.
5. Re-check global balance and ensure the 100-CU reserve can be maintained even if the existing active assignment continues.
6. Only then freeze the preregistration SHA and start the 6-pair estimation cohort. If any gate fails, preserve the draft and run no model calls.

## Interpretation boundary

With six pairs, results are descriptive and intended to estimate operational feasibility, timing variability, and preliminary paired outcome frequencies. They cannot establish V2 efficacy, equivalence, generalization beyond this task/configuration, or the 80%-power claim associated with the 715.5-CU scenario.
