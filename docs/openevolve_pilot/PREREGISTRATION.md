# OpenEvolve pilot — pre-registration

Registered 2026-10-29, before any campaign run. Code state: the commit tagged
`oe-pilot-prereg` (this file's commit). Design:
`docs/superpowers/specs/2026-10-29-openevolve-pilot-design.md`; implementation plan:
`docs/superpowers/plans/2026-10-29-openevolve-pilot.md`; spike measurements (not data):
`docs/openevolve_pilot/spike_notes.md`.

This is a descriptive pilot. No significance test is pre-registered; results are
reported as medians and ranges over seeds and as curves over the LLM-call budget. The
seed count for any follow-up will be chosen from the variance observed here.

## Research questions

- **RQ1.** At an equal budget of 100 LLM calls, does OpenEvolve driven only by
  `gemma4:12b` (arm L) reach a higher *shipped verified score* than best-of-N sampling
  from the same model without evolution (arm B), and than the seed program?
- **RQ2.** What does replacing ~10% of mutations with Gemini 4 Argon (arm LC) change in
  the shipped verified score relative to L, and at what cloud cost (calls, tokens, wall
  time, `agy` weekly quota share)?
- **RQ3.** What fraction of candidates that the working evaluator ranks above the seed
  fail the independent re-check or do not beat the seed's verified score (*false
  improvements*), and in how many runs is the shipped program itself false? Per arm.

Directional expectations, stated in advance (not tested): L > B on both tasks; LC ≥ L;
false improvements exist on both tasks, more often on kNN (more exploitable working
evaluator).

## Design (frozen)

| item | value |
|---|---|
| Tasks | `circle_packing` (n=26, upstream OpenEvolve example), `knn` (ORION approximate kNN) |
| Arms | L: OpenEvolve, Gemma only · LC: OpenEvolve, each call Argon with p=0.1 (seeded per call index) · B: best-of-N from the seed, same prompt template, no history |
| Seeds | 101, 202, 303 |
| Budget | exactly 100 logged LLM calls per run (router-enforced); 18 runs |
| Order | all L and B runs (task × seed), then all LC runs |
| Local model | `gemma4:12b` (Ollama digest `4eb23ef187e2`), Ollama server 0.33.3, `think=false`, `num_ctx=16384`, `num_predict=2048`, temperature 0.7, top_p 0.95 |
| Cloud model | `gemini-4-argon-medium` via Antigravity CLI 1.4.10 (`agy -p`, empty cwd, "answer directly, no tools" prefix) |
| Framework | OpenEvolve 0.4.0; diff-based evolution; `enforce_evolve_blocks: true`; LLM retries 0; evaluator retries 0; 1 evaluation worker; population 60, archive 25, 4 islands |
| Configs | `experiments/openevolve/configs/circle_packing.yaml` (SHA-256 `277e3330…eda7c6`), `knn.yaml` (`20792039…a550`) |
| Sandbox | Docker image `antigravity-orion:local` (`sha256:aae97353…7ae2`), no network, read-only workspace, non-root, 2 CPUs, 4 GB |
| Host | RTX 4060 Laptop 8 GB (driver 610.57.04), 31 GB RAM, Linux 7.0.0-38, Python 3.13.14; no other GPU process (checked by `run_openevolve_pilot.sh check`) |

## Evaluators

**Working evaluators** (drive selection; logged for every candidate):
- circle packing: OpenEvolve's upstream evaluator unchanged (tolerance 1e-6, reported
  sum ignored for the score), candidate executed in the sandbox;
  `combined_score = sum_radii / 2.635` if valid else 0.
- kNN: three fixed Gaussian-blob dev sets (N = 200k/100k/150k, d = 32/16/24); recall@k vs
  exact ids; time = best of 3 calls on identical input vs exact search in the same
  sandbox call; `combined_score = mean_recall * min(1, 0.5 / max_ratio)**2`. Distances
  are not checked.

**Independent re-checks** (never seen by the model, run after the campaign):
- circle packing: 26 circles, finite values, radii > 0, containment and pairwise
  non-overlap with tolerance 1e-12, two sandbox runs with identical output, static check
  (no `os, sys, subprocess, socket, inspect, importlib, builtins, ctypes, pathlib,
  shutil, gc` imports; no `open, eval, exec, compile, __import__, getattr, setattr,
  globals, locals, vars` calls). Verified score = recomputed sum of radii / 2.635.
- kNN: RIE `check_neighbors` on 6 fresh uniform challenges (seeded by the code hash),
  exact-copy queries returned first at distance ≤ 1e-8, the same static check; score =
  the working formula on a holdout of three blob sets with different parameters
  (seeds 9001–9003), exact ids computed on the host, candidate time summed over 3
  disjoint query batches.

## Endpoints

- **Primary:** shipped verified score per run = re-check score of the candidate with the
  highest working score (earliest on ties); if it fails the re-check, the run scores the
  seed's verified score. Compared L vs B (RQ1) and LC vs L (RQ2) by median and range.
- **Secondary:** best verified score among all candidates; false-improvement rate
  (pooled and per run); number of runs whose shipped program is false; curves of best
  working and shipped verified score against LLM calls; Gemma and Argon wall time,
  Argon calls and tokens; failed calls by status.

## Deviation policy

- Argon quota exhaustion pauses the campaign (the call is not logged or spent); it is
  resumed after the `agy` refresh with `--resume`. Every pause and restart is recorded
  in `docs/openevolve_pilot/DEVIATIONS.md` and in the run manifests.
- Crashes are resumed from the logged calls and the last checkpoint; no run is deleted
  or re-run from scratch unless its data is unusable, which is then recorded with the
  reason.
- Any change to code, prompts or settings after this commit is a deviation, recorded
  with its reason before results are analysed.
- Spike data (`temp/oe_spike*`) and probe calls are excluded from all analyses.
