# Colab study: OpenEvolve across Gemma 4 sizes, MoE in focus — Design

Date: 2026-10-30. Owner: Oleksandr Pylypenko. Status: design approved by the owner 2026-10-30.
Runs in parallel with, and independently of, the laptop pilot
(`docs/superpowers/specs/2026-10-29-openevolve-pilot-design.md`, pre-registered,
tag `oe-pilot-prereg`), whose code it reuses.

## Intent

The laptop pilot asks whether a 12B Gemma on an 8 GB laptop can drive evolutionary
program search. The owner interrupts the laptop often; Colab can run without it. The
owner has ~400 Colab compute units (Google AI Pro grant, expiring ~90 days after
grant) and wants them used as efficiently as possible, on a larger Gemma — the
Mixture-of-Experts **Gemma 4 26B A4B** — plus Gemini 4 Argon, as in the pilot.

Spike, 2026-10-30, Colab L4 24 GB (1.94 units/h), Ollama 0.40.13, `gemma4:26b`
(digest `001e5dafc3c7`), `think=false`, `num_ctx=16384`: model 100% on GPU, ~90
tokens/s, 15–23 s per pilot mutation prompt (laptop 12B: 73–116 s). A notebook cell
kept running for its full 6 minutes after the laptop client was killed at 30 s.
`agy` 1.4.8 is pre-installed on Colab VMs; it needs the owner's login once per VM.

## Research questions

Tasks, seed programs, configs, working evaluators, re-checks and endpoints are the
pilot's, unchanged:

- **RQ-scale (main).** On one environment (Colab L4, one Ollama version), how do
  Gemma 4 E4B, 12B, 26B A4B (MoE) and 31B (dense) compare as OpenEvolve mutators:
  shipped verified score at 100 calls, OpenEvolve vs best-of-N, and false-improvement
  rate? Is the MoE (≈4B active) closer to the 31B or to the E4B per compute unit?
- **RQ-cloud.** For the MoE: what do ~10% Argon mutations add (arm LC)?
- **RQ-budget.** For the MoE: with 1,000 iterations on circle packing and 500 on kNN,
  how close does it get to the record 2.635, and does the false-improvement rate grow
  with search length?
- **RQ-hardware (by-product).** `gemma4:12b` on Colab vs the laptop pilot: same model,
  same protocol, different hardware/Ollama.

## Design

| item | value |
|---|---|
| Models (Ollama tags, bare, as on the laptop) | `gemma4:e4b`, `gemma4:12b`, `gemma4:26b` (26B A4B MoE), `gemma4:31b` |
| Generation | identical to the pilot: `think=false`, `num_ctx=16384`, `num_predict=2048`, T 0.7, top_p 0.95 |
| Cloud model | `gemini-4-argon-medium` via `agy` on the VM, same "answer directly, no tools" prefix |
| Hardware | Colab L4 24 GB, one session at a time |
| Tasks | `circle_packing`, `knn` |
| Standard runs | each model: 2 tasks × arms L and B × 100 calls; 10 seeds (101…1010), 31B 5 seeds (spike: ~10 tokens/s) |
| Cloud arm | MoE only: arm LC, 2 tasks × 5 seeds × 100 calls (~100 Argon calls) |
| Long runs | MoE, arm L, 3 seeds: circle packing 1,000 iterations, kNN 500 |
| Order | MoE standard L/B → 12B → E4B → 31B → MoE long runs → MoE LC last (owner's choice: one `agy` login on the final VM, fresh weekly Argon quota, no overlap with the laptop pilot's LC runs) |
| Concurrency | 2 runs at a time per VM (`OLLAMA_NUM_PARALLEL=2`), so the GPU generates while the other run evaluates; kept only if the spike shows it does not slow single-run iterations by more than it gains |
| Re-check | after download, on the laptop, in the pilot's Docker sandbox — verified scores of the Colab study and the laptop pilot come from one environment |

Budget estimate (spike rates; 31B dense assumed ~3x slower than the MoE): ~130 L4-hours
≈ 250 compute units; hard cap 330 units, leaving ≥ 70 in reserve. If `gemma4:31b` does
not fit on the L4 with `num_ctx=16384`, it is dropped (no A100: ~3x the price per hour).

## Execution on Colab

- One long-running notebook cell runs a driver over the run list; finished runs are
  skipped, unfinished ones resume (`run_pilot.py --resume`). The laptop can be closed.
- Persistence: a Cloud Storage bucket in project `gen-lang-client-0903767759`, written by a
  dedicated service account whose only permission is object admin on that bucket
  (owner approved 2026-10-30). Its key is placed on each VM under `/root` (mode 600),
  never readable by candidate code. The driver syncs run directories after each
  checkpoint and each run; a lost VM costs at most in-flight work.
- Restarts: Colab ends VMs after at most ~24 h (unguaranteed). A watchdog on the laptop
  (user systemd timer, every 10 min, only while the laptop is awake) checks the bucket
  for an unfinished study and `colab sessions` for a live VM; if none, it creates an L4
  VM, installs Ollama, restores state from the bucket and starts the driver. Only the
  final LC phase needs the owner (one `agy` login); until then the driver waits there.
  Every VM start/stop is logged in the bucket.
- Isolation without Docker: candidates run as an unprivileged user (`nobody`) in a
  temporary directory with time and memory limits, with no read access to the code
  checkout or `/root` (bucket key, agy credentials); network is cut with a
  network namespace if the VM allows it (checked in the spike), otherwise this is
  recorded as a limitation. The holdout and re-check run only on the laptop.
- Spending guard: the driver reads `colab usage` equivalents on the VM where possible,
  and otherwise stops on a wall-clock budget derived from the 330-unit cap; Claude
  stops any idle session.
- Argon quota is shared with the laptop pilot: a quota pause stops the LC phase until
  the weekly refresh; Gemma is never substituted.

## Non-goals and anti-goals

- No change to the running laptop pilot or its pre-registration; separate
  pre-registration and result folder (`benchmarks/colab_scale/`).
- No model outside the Gemma 4 family as a mutator; no fine-tuning.
- Never an idle GPU session; never model-written code with access to the owner's Drive
  or credentials; never substituting Gemma for a failed or quota-blocked Argon call.

## Calls made by Claude

- Bare Ollama tags, as on the laptop; identical generation settings across models, so
  size is the only intended difference (truncation rates are reported, not tuned away).
- 12B rerun on Colab, so the size ladder lives in one environment and the laptop pilot
  becomes a hardware comparison.
- L4, not A100; 31B dropped rather than moved to A100 if it does not fit.
- 10 seeds for standard runs, 5 for LC (Argon quota), 3 for long runs.
- Concurrency 2, subject to the spike measurement.
- GCS bucket + narrowly scoped service account for persistence; laptop watchdog for restarts.
