# Evaluator V2 feasibility pilot: preregistration

**Protocol ID:** `evaluator-v2-pilot-2026-11-19`
**Status:** frozen before any new model calls
**Purpose:** technical feasibility only; this pilot is not a confirmatory test and will not support a general efficacy claim.

## Question

Can a correctness gate that checks the reported squared Euclidean distances for every returned kNN neighbor run before timing and fitness, reject fabricated distance outputs, and resume safely after a Colab runtime interruption?

The prior evaluator (V1) checked neighbor ID shape and recall, but did not validate reported distances. V2 first runs an untimed preflight on the development sets. If any ID or distance is invalid, it returns score zero and does not start timing. After preflight passes, V2 times three calls per set and independently validates all three timed outputs before scoring the fastest valid call. The score formula and development sets remain unchanged.

## Conditions

| Condition | Evaluator | Model | Task | Arm | Calls |
|---|---|---|---|---|---:|
| V1 | Existing recall-and-speed evaluator | Ollama `gemma4:12b` | Approximate kNN | `L` (0% cloud) | 20 |
| V2 | Distance correctness gate, then same score | Ollama `gemma4:12b` | Approximate kNN | `L` (0% cloud) | 20 |

Both conditions use seed `20261119`, the same initial program, prompt/config, 16,384-token context, 2,048-token generation cap, temperature 0.7, top-p 0.95, and one Colab L4 session. Conditions run serially in the fixed order V1 then V2. The single-seed design is chosen to cap cost and check implementation feasibility; it does not estimate a population effect. No Argon/Gemini calls are made.

The per-condition budget is exactly 20 logged model calls. A resumed run continues the same folder and seed until that budget is reached; no reset, replacement seed, or unlogged retry is allowed. A backend failure recorded as a call remains part of the 20-call budget.

## Outcomes and decision rules

Primary feasibility checks:

1. A valid output with correct squared distances passes V2 preflight.
2. A candidate that reports fabricated distances receives score zero and produces no timing-phase invocation.
3. Every timed V2 result is independently checked before fitness is accepted.
4. Each condition stops at 20 logged calls and resumes from GCS without duplicate call indices.

Descriptive outcomes include V1/V2 verified completion, best candidate independent re-check score, false-improvement count among V1 candidates above the seed, V2 rejection reasons, model tokens, wall time, and GPU hours. Report raw counts and run artifacts. Do not compute a significance test from one paired seed, claim equivalence, or use the pilot to establish a general performance advantage.

## Runtime and persistence

- Execution: Colab L4, controlled by the dedicated e2-micro watchdog `paper2-colab-watchdog`.
- Source: the exact Git commit recorded in the GCS state and each run manifest.
- State/artifacts: `gs://oe-colab-scale-0903767759/paper2-v2/b1/`.
- Checkpoint cadence: each model response is mirrored to GCS before it is returned to OpenEvolve; other files and state are synchronized every 30 seconds and after each condition.
- Resume: restore the GCS prefix, continue the existing run folder, and preserve logged call indices.
- Model/API scope: local Gemma through Ollama only; no cloud-model fallback.
- Resource ceiling for this feasibility pilot: at most 15 Colab compute units; stop the L4 runtime as soon as both conditions complete or the ceiling is reached.

## Analysis and deviations

This is a predeclared engineering pilot. Summaries are descriptive. Any change in condition order, model tag/digest, call budget, seed, task, prompt, evaluator, hardware, timing policy, or checkpoint procedure is a deviation. Record the reason, timestamp, affected runs, and whether the change happened before or after examining results in `docs/evaluator_v2/DEVIATIONS.md` and the GCS state. Never overwrite a completed run or its raw logs.

The protocol file and its Git commit are the preregistration record. The runner writes the Git SHA, model digest, hardware, protocol ID, per-condition manifests, call logs, candidate logs, and resumable OpenEvolve checkpoints to the GCS prefix above.
