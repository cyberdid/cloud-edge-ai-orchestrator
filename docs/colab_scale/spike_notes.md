# Colab scale study — spike notes (2026-10-30)

Throwaway measurements before pre-registration; nothing here is study data.

VM: Colab L4 (23 GB), 12 vCPU, 52 GB RAM, Python 3.13, Ubuntu 24.04; 1.94 compute units/h.
Bootstrap (`experiments/openevolve/colab/bootstrap.sh`): 3 min, Ollama 0.40.13,
`gemma4:26b` digest `001e5dafc3c7`, `uv sync --locked`, candidate venv `/opt/oe-cand`
(Python 3.12, numpy 2.5.4 — the Docker sandbox's lock), `unshare -n` available.

## Isolation (candidate as uid 65534, subprocess executor)

`/root/oe/pyproject.toml`, `/root/.oe/sa-key.json`, `/root/oe-runs`: PermissionError;
`http://example.com`: URLError (no network namespace route); numpy 2.5.4.

## Concurrency (MoE, circle packing arm L, 10 iterations)

| mode | wall | per iteration | notes |
|---|---|---|---|
| 1 run | 365 s incl. ~190 s model load | ≈17.5 s | 10/10 replies ok, best 0.73 |
| 2 runs at once | 277 s for 20 iterations | ≈13.9 s | 18 ok, 2 truncated; best 0.566 / 0.688 |

+26% throughput ≥ the plan's 20% threshold → concurrency 2 kept.

## Model fit

- `gemma4:26b` (MoE): 100% GPU, ~90 tokens/s, 15–23 s per mutation.
- `gemma4:31b` (dense, digest `17ba34c06c80`): 100% GPU with `num_ctx=16384`, but
  ~9.7 tokens/s (memory-bandwidth bound) → 60–120 s per mutation, ~6x the MoE.
  Kept (it fits), with 5 seeds instead of 10 to stay under the cap.

## Operational findings

- `colab exec` twice hung client-side before reaching the kernel (kernel IDLE, no
  history entry); a kernel restart cleared it → watchdog restarts the kernel and
  retries once, and verifies the driver reached phase "running".
- A kernel restart kills `ollama serve` (child of the cell) → the driver restarts it.
- The client connection can drop during a long cell; the cell keeps running.

## Budget (revised)

MoE standard ~22 h, 12B ~20 h, E4B ~11 h, 31B (5 seeds) ~50 h, MoE long ~25 h,
MoE LC ~6 h → ~134 L4-hours ≈ 260 units; cap 330.
