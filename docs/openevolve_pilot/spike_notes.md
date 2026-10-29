# OpenEvolve pilot — spike notes

Throwaway measurements taken before pre-registration. None of these calls or
programs are campaign data.

## Argon probe (Task 8), 2026-10-29

Prompt: the circle-packing seed prompt that arms B and L+C send (OpenEvolve 0.4.0
diff-evolution template, upstream system message), through
`CloudArchitectClient(backend="agy", model="gemini-4-argon-medium").complete` in an
empty working directory.

### Probe 1 (no extra instruction)

| call | latency | prompt tok | completion tok | reply chars | applies as diff | error |
|---|---|---|---|---|---|---|
| 0 | 25.8 s | – | – | 0 | no | `agy produced no text; denied tool use: RunCommand` |
| 1 | 45.2 s | 2,055 | 3,045 | 7,732 | yes | |
| 2 | 53.9 s | 15,626 | 4,010 | 6,593 | yes | |

Finding: `agy` is an agent; in headless mode a tool call is denied and the turn ends
without text. With 1 of 3 calls lost this way, the L+C arm would be handicapped by an
artefact of the CLI, not by the model. `agy` has no flag to disable tools, so the
router now prefixes Argon prompts with an explicit "answer directly, no tools"
instruction (`router.AGY_NO_TOOLS`, commit `7bc0356`).

Latency is ~0.5–1 min per call, far below the earlier estimate of several minutes,
so ~60 Argon calls cost about one hour of wall time.

### Probe 2 (with the no-tools instruction)

| call | latency | prompt tok | completion tok | reply chars | applies as diff | error |
|---|---|---|---|---|---|---|
| 0 | 27.8 s | 2,091 | 2,395 | 6,909 | yes | |
| 1 | 34.8 s | 2,092 | 2,777 | 5,466 | yes | |
| 2 | 66.9 s | 2,097 | 5,372 | 7,474 | yes | |

3 of 3 replies applied as diffs; prompt size is stable (~2.1k tokens, the 15.6k of
probe 1 call 2 was agy's own tool context). Median latency 35 s.

### Quota

The owner reads the `agy` weekly quota (Gemini 4 Argon bucket) before and after the
probes; 63.5% remained on 2026-10-29 before any probe.

## Gemma spike (Task 9), 2026-10-29

Arm L, seed 1, 10 iterations per task, `gemma4:12b` via Ollama 0.33.3, `think=false`,
`num_predict=2048`. Data in `temp/oe_spike*` (git-ignored, never analysed).

| run | task | context | calls ok / truncated | replies applied as diff | median s/call | best working score |
|---|---|---|---|---|---|---|
| spike 1 (stopped) | circle_packing | 4096 (Ollama default) | 2 / 8 | ~5/10 | 54 | 0.562 |
| spike 2 | circle_packing | 16384 | 10 / 0 | 10/10 | 73 | 0.619 (seed 0.364) |
| spike 2 | knn | 16384 | 8 / 2 | 4/10 | 116 | 0.394 (seed 0.224) |

Spike 1: prompt 3,147 + reply 949 = 4,096 tokens: Ollama's default context, not
`num_predict`, cut the replies. Fixed by `num_ctx=16384`, the value Paper I used for the
same model (commit `bf2878d`); placement stayed 19%/81% CPU/GPU.

Decision rules (plan Task 9): truncated 2/10 on kNN is not > 2/10, so `num_predict`
stays 2048; applicable-diff share 4/10 is not < 3/10, so the system message is
unchanged. kNN's 6 failed iterations: 3 SEARCH blocks that did not match the parent,
2 truncated replies (unmatched delimiter), 1 other. Frozen settings:
`num_ctx=16384`, `num_predict=2048`, `think=false`, temperature 0.7, top_p 0.95.

Wall time: circle packing 757 s / 10 iterations, kNN 1,069 s / 10 iterations →
campaign estimate ≈ 44 h for the Gemma calls (900 per task) plus ~1 h of Argon.
