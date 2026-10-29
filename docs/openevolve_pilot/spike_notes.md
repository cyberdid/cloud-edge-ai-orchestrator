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
