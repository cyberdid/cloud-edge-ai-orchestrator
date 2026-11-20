# B2 design memo: confirmatory comparison of kNN evaluators V1 and V2

**Status:** design memo only; not a preregistration and not authorization to start model calls.  
**B1 evidence:** `B1_RESULTS.md`; raw artifacts at `gs://oe-colab-scale-0903767759/paper2-v2/b1/`.  
**Purpose:** specify what must be decided and validated before B2 can be frozen.

## 1. What B1 established

B1 established that both evaluator versions can run for 20 logged Gemma calls and leave resumable artifacts in GCS. It did not estimate an effect: it used one seed, V1 had six runtime starts, and no V2 candidate was recorded as rejected specifically for fabricated distances. It also exposed a watchdog completion-state failure, now corrected in the source and covered by focused tests. B1 must remain a feasibility cohort and must not be pooled into B2's confirmatory estimate.

## 2. Research question and bounded claim

**Question:** Under one fixed local Gemma model, one approximate-kNN task, and a fixed OpenEvolve budget, does adding a pre-fitness correctness gate reduce the probability that the run's selected candidate fails a sealed independent correctness re-check, and what compute and search-quality cost does that gate add?

Any result generalizes only to the frozen kNN task, model digest, prompt, OpenEvolve version, call budget, and hardware/timing policy. It does not establish that evaluator gates generally improve program evolution.

## 3. Design decisions that must be frozen

### 3.1 Matched evaluator conditions

Implement `V1-matched` and V2 on the same candidate sandbox, generated datasets, reference implementation, number/order of timed repeats, warm-up policy, thread affinity, timeout, and score formula. V1-matched omits the correctness gate but keeps the same timing harness. The historical V1 evaluator may be reported as a descriptive reference only; it must not be the B2 control if its timing path differs.

Keep candidate feedback format and prompt identical. Do not pass V2's detailed rejection reason to the model unless feedback itself is an explicit third experimental condition. The current B1 runner and evaluator need an audit to prove which evaluator fields OpenEvolve exposes in the next prompt.

### 3.2 Pairing and order

Each pair uses the same preregistered task seed, seed program, model digest, task specification, and call budget. Randomize V1-matched/V2 order within pairs using a recorded allocation file generated before any model calls. Do not adapt the allocation or sample size from interim outcome values. Use an isolated Colab runtime/session for the full pair where practical; log restarts and resume without duplicating call indices.

If the backend cannot set an inference seed, say so. Pairing then controls task and configuration, not stochastic model responses. Record actual model digest, Ollama version, OpenEvolve version, CUDA/GPU, prompt/config hashes, and all sampling parameters for every run.

### 3.3 Independent selection check

After both runs in a pair finish, freeze the final selected program hash from each arm. Apply one independent checker to both. Its inputs must be generated from a separately committed holdout seed list and must not be visible to the working evaluator, candidate, or model. The checker validates output shape, integer/range/uniqueness constraints, true squared distances, exact-copy behavior and ties under the declared contract, then calculates recall and independently measured runtime.

The independent-check implementation and dataset generator must be versioned and hashed before the first B2 model call. A failed or missing selected artifact counts as a selection failure in the primary endpoint; report a sensitivity analysis that distinguishes operational missingness from a candidate that ran and failed correctness.

## 4. Outcomes

### Primary endpoint

For each run, `selected_holdout_invalid = 1` if the program actually selected by that run's working evaluator fails the sealed independent correctness checker, or if no selected artifact can be independently checked; otherwise 0. The treatment contrast is the paired difference in this run-level binary endpoint, V2 minus V1-matched. Report the full paired 2×2 table, absolute paired risk difference, confidence interval, and exact McNemar test only if the sample-size plan supports the planned test.

The endpoint is about the evaluator-selected program, not the best candidate found anywhere in the run. Do not conflate these.

### Secondary outcomes

- Selected holdout verified score and paired difference, with missingness shown explicitly.
- Selection regret only when the full candidate set has independent re-check coverage; otherwise label any estimate as a lower bound and state the covered candidate-pool rule.
- False-improvement rate among candidates the working evaluator ranks above the seed, with the candidate/run denominators and hash-level duplication policy specified.
- Valid proposal rate; V2 preflight rejection count by structural, distance, exact-copy, timeout, and execution-failure reason.
- Gate CPU time, query latency, one-time index/build time if any, end-to-end wall time, total model tokens, GPU time, and Colab CU. Report cost as a separate outcome, not folded into the correctness score.

Use task-seed pair as the independent unit for run-level inference. Candidate-level rows are nested repeated observations and cannot be counted as independent samples. Declare familywise correction for secondary outcomes before collection.

## 5. Sample size and compute feasibility

Do not select `n` from B1's one pair. First calculate exact paired-binary power for a declared smallest meaningful absolute reduction in selected-holdout-invalid risk. The assumed discordant-pair probabilities must be justified from independent historical run-level data or treated as scenarios; report sensitivity across conservative values. If no credible planning values exist, B2 cannot be called confirmatory and should be explicitly registered as an estimation study.

The B1 account balance fell from 223.12 CU to 211.39 CU for 40 calls across V1/V2, including the six V1 runtime starts and pilot bootstrap. This gives a crude observed rate of 0.293 CU/call, not a stable unit cost. At that rate:

| Paired seeds | Calls (20 per arm per pair) | Crude CU projection |
|---:|---:|---:|
| 6 | 240 | 70.4 CU |
| 8 | 320 | 93.8 CU |
| 20 | 800 | 234.6 CU |

This is a rough linear upper-budget scenario, not a Colab quote: shared model downloads, runtime reuse, infrastructure retries, and CU billing make the true cost uncertain. The current observed balance leaves about 111.39 CU above a 100-CU reserve. Therefore 20 pairs do not fit this planning envelope under the observed rate. Do not silently reduce a powered confirmatory sample to six or eight pairs. Either (a) secure/accumulate a preregistered compute budget that covers the powered sample plus contingency, or (b) explicitly change the study to an estimation/feasibility design with appropriately weaker claims.

Before any B2 call, run a no-model-call cost rehearsal: provision one Colab L4 session, verify that the patched watchdog marks completion and does not duplicate work, execute the evaluator harness on saved candidate programs, and measure compute-unit cost. The rehearsal must use no new LLM calls and be recorded separately from B2.

## 6. Required implementation and verification gates

1. **Implemented in source:** add `v1_matched` using the same kNN driver, development sets, exact-reference timing, three candidate timings, and score function as V2, while deliberately ignoring reported distances. Add evaluator-content SHA-256 to manifests and refuse unsafe resume after an evaluator-content change. Focused unit tests pass. A local CPU subprocess smoke on a tiny synthetic set confirmed that V1-matched scores a candidate with fabricated distances while V2 rejects it at preflight and never enters timed mode. Before B2 freeze, still run a no-model-call Docker rehearsal on saved candidates to check the actual packaged runtime.
2. Add explicit selected-program manifest, evaluator/version/source hashes, selection fallback rule, and holdout-check result artifact.
3. Test fabricated but correctly shaped distances, NaN/Inf, duplicate/out-of-range IDs, valid approximate neighbors with recall below 1, timeout, and no timing after a failed preflight.
4. Test resume for an interrupted pair, duplicate/missing call indices, mismatched model/protocol hash, and completed-manifest recovery.
5. Verify watchdog's terminal GCS state cannot be overwritten by a running heartbeat; test reconciliation with complete, incomplete, malformed, duplicated, and mismatched records.
6. Freeze and hash the paired-seed allocation, holdout generator/seeds, model/runtime config, analysis script, endpoint definitions, sample size, stopping rules, and cost cap.
7. Obtain a final review of the protocol and code before execution. Then run only the frozen cohort; publish the full run inventory and all deviations, including negative or incomplete outcomes.

## 7. Codex and tool allocation

| Work | Suggested model / effort | Tools | Token/cost discipline |
|---|---|---|---|
| Protocol and B2 design | `gpt-6.2-sol` / medium | Read only relevant B1/B2 docs and runner/evaluator modules; Python for power scenarios | No model calls; one concise evidence summary |
| Tests and matched harness | `gpt-6.2-sol` / medium; `high` only for concurrency/recovery design | pytest, Python, Docker, saved candidate JSON, `git diff` | Run focused tests; no LLM generation |
| Power/statistical review | `gpt-6.2-astra` / high for one frozen design review | Python/SciPy or exact binomial enumeration; protocol and assumptions only | One targeted review; do not submit all raw logs |
| PDF/manuscript review after results | `gpt-6.1-terra` / low–medium for rendered pages | Tectonic, Poppler, `pdftotext`, page render/contact sheet | Only after results; no model calls to Gemma/Argon |
| Experiment | Gemma model/digest explicitly frozen in B2 | e2-micro watchdog + Colab L4, GCS artifacts, Ollama, Docker | CU/token caps, 100-CU reserve if still adopted, no cloud-model fallback |

## 8. B2 launch gate

B2 remains **not ready to launch** until the matched timing implementation exists, the independent holdout is committed and inaccessible to the working evaluator, the run-level endpoint and power assumptions are frozen, the budget fits available Colab CU with contingency, and the no-call watchdog rehearsal passes. The e2-micro service is currently disabled and no Colab assignment is active.
