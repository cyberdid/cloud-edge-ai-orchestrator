# Colab scale study — pre-registration

Registered 2026-10-30, before any study run. Code state: the commit tagged
`colab-scale-prereg`. Design: `docs/superpowers/specs/2026-10-30-colab-moe-study-design.md`;
plan: `docs/superpowers/plans/2026-10-30-colab-scale-study.md`; spike (not data):
`docs/colab_scale/spike_notes.md`. Protocol, tasks, evaluators, re-checks and endpoints
are those of the laptop pilot (`docs/openevolve_pilot/PREREGISTRATION.md`, tag
`oe-pilot-prereg`), unchanged except where stated below.

Descriptive study: medians, ranges and budget curves; no significance test is
pre-registered.

## Questions

- **RQ-scale.** Gemma 4 E4B, 12B, 26B A4B (MoE) and 31B (dense) as OpenEvolve mutators
  on one environment: shipped verified score at 100 calls, OpenEvolve (L) vs best-of-N
  (B), false-improvement rate. Is the MoE closer to the 31B or to the E4B, per compute
  unit?
- **RQ-cloud.** MoE: what ~10% Argon mutations (LC) change relative to L.
- **RQ-budget.** MoE: 1,000 iterations (circle packing) / 500 (kNN): best verified score
  vs the record 2.635, and false-improvement rate vs search length.
- **RQ-hardware.** 12B on Colab vs the laptop pilot's 12B.

Expectations stated in advance (not tested): verified score rises with model size;
the MoE lands between 12B and 31B on quality and is the best per compute unit;
L > B for every model; false improvements present for every model.

## Frozen design

| item | value |
|---|---|
| Models | `gemma4:26b` (digest `001e5dafc3c7`), `gemma4:12b`, `gemma4:e4b`, `gemma4:31b` (`17ba34c06c80`); digests of 12b/e4b recorded by the driver at first pull |
| Serving | Ollama 0.40.13 on the VM, `OLLAMA_NUM_PARALLEL=2`, `think=false`, `num_ctx=16384`, `num_predict=2048`, T 0.7, top_p 0.95 |
| Cloud model | `gemini-4-argon-medium` via `agy` 1.4.8 on the VM ("answer directly, no tools" prefix) |
| Hardware | Colab L4 24 GB, 12 vCPU, one VM at a time; 2 runs at once, same model only |
| Run list (156) | per model, tasks `circle_packing` + `knn`, arms L and B, 100 calls: seeds 101…1010 (10) for 26b, 12b, e4b; 101…505 (5) for 31b. MoE long runs: arm L, seeds 101/202/303, circle packing 1,000 and kNN 500 iterations. MoE LC: seeds 101…505, both tasks, 100 calls |
| Order | MoE L/B → 12B → E4B → 31B → MoE long → MoE LC (owner logs in to `agy` once) |
| Framework | OpenEvolve 0.4.0, configs and seed programs identical to the pilot |
| Candidate execution | subprocess executor: uid/gid 65534, no network namespace route, 4 GiB address space, 16 MiB files, same timeouts; Python 3.12 + numpy 2.5.4 (the Docker sandbox's lock). Working-evaluator timings (kNN) are measured on the VM |
| Re-check | after download, on the laptop, in the pilot's Docker sandbox, identical rules |
| Storage | `gs://oe-colab-scale-0903767759` (service account with access to this bucket only) |
| Budget | stop at 330 compute units spent from 399.47; spike used ~2.2 |

## Endpoints

As the pilot: primary = shipped verified score per run (seed's verified score if the
working-best program fails the re-check). Secondary: best verified score, false-
improvement rate, shipped-false runs, budget curves, wall time and compute units per
model, Argon calls/tokens. Scale comparison: medians by model for L and B; units per
verified-score point.

## Deviation policy

- VM ends, kernel restarts, hung `colab exec` and watchdog restarts are expected; they
  are logged in `gs://…/vm_log.jsonl` and summarised in `docs/colab_scale/DEVIATIONS.md`.
  Runs resume from logged calls and checkpoints; no run is deleted or restarted from
  scratch unless unusable (recorded with the reason).
- Argon quota exhaustion or a missing `agy` login pauses the LC phase; Gemma is never
  substituted.
- If the cap is reached before the end, the study stops; completed runs are analysed
  and missing ones are reported as missing, in run-list order.
- Any change to code, prompts or settings after the tag is a deviation, recorded before
  results are analysed.
