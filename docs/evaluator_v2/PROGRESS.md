# Paper II revision and evaluator V2 progress

Last updated: 2026-11-20

## Status

| Work package | Status | Evidence / next action |
|---|---|---|
| A1–A3: manuscript corrections | Complete in `main` commit `0e5f8fb1bb0279dffd656618ff2ab3a0a17f17e8` | Preregistration history, claims, and layout revisions are in that commit. |
| B1: evaluator V2 feasibility pilot | Complete as a technical feasibility run; top-level GCS state reconciled after runner response loss | V1 and V2 each have a `complete` manifest and exactly 20 contiguous call records (18 `ok`, 2 `truncated`); evaluator/model/seed/protocol/commit match. Raw artifacts are in `gs://oe-colab-scale-0903767759/paper2-v2/b1/`. No active Colab assignment remains. e2-micro checkout is at `68a1c8a68994395d5f69f4558d71a7966c3bc53c`; the service is inactive and disabled. This is one-seed descriptive evidence, not a confirmatory result. See `DEVIATIONS.md` and `B1_RESULTS.md`. |
| B2: confirmatory V1/V2 comparison | Design memo, matched V1 control, and official L/LC selection-artifact tracking implemented; not ready to launch | See `B2_DESIGN.md`. B1 provides no efficacy estimate. Exact McNemar planning scenarios require 61–287 paired seeds (hypothetical assumptions) and project to ~715–3,367 CU, far above the ~111 CU available above the 100-CU reserve. A post-hoc audit found V1 candidate-log maximum (0.2502050) differs from official selected artifact recorded score (0.2433403); both code hashes are tracked and future primary endpoints use the official program hash. Remaining gates: sealed holdout, empirically justified power assumptions, no-call watchdog rehearsal, and a feasible registered budget. Do not launch a small run as confirmatory. |
| B2 estimation/feasibility cohort | Budget-capped six-pair draft and runner implemented; Colab persistence rehearsal passed; **no model calls made** | `B2_ESTIMATION_DRAFT.md`, `B2_NO_CALL_REHEARSAL.json`. Cap: 240 Gemma calls / 70.4 account-wide CU draw / 100-CU reserve. The isolated Colab runtime upload-stop-recreate-restore check passed for source `34d663e`; measured no-call draw was 0.16 CU, leaving 208.49 CU on 2026-11-20. Remaining gates: freeze checker/holdout and protocol hashes; systemd detached-laptop test and service install; pre-run balance check. This is estimation only, not confirmatory. |
| C: VoiceID external validation | Not started; no runnable clustering implementation located | Read-only audit of `/home/cyberdid/Projects/VoiceID` found no `.py` runtime. Its VoxCeleb notebook defines `WaveFakeModelEval`, `WaveFakeModelAudit`, and `UltimateWaveFakeModel` for spoof detection, not speaker clustering or cluster merge. Do not treat it as the planned external task; first locate a runnable cluster/merge entrypoint, approved labeled data/embeddings, and a speaker-disjoint holdout. No VoiceID code or private data was changed or executed. |

## Frozen B1 feasibility pilot

- Task: the existing approximate kNN benchmark.
- Conditions: evaluator V1 followed by evaluator V2.
- Model: local Gemma 4 12B through Ollama; no Gemini Argon calls.
- Budget: 20 logged model calls per condition, fixed seed `20261119`.
- Compute guard: stop at 15 Colab compute units spent or at 100 units remaining, whichever comes first.
- Interpretation: infrastructure and evaluator feasibility only. The pilot is not a confirmatory statistical test and will not support population-level claims.

## B1 observed outcome (descriptive only)

- Both conditions reached 20 logged Gemma calls, call indices 0–19, using model digest `6114515d63c1`, seed `20261119`, commit `9e45ee6b2cee02e910222c03e84bf680043d43ce`, and the frozen protocol SHA-256.
- V1: 18 `ok`, 2 `truncated`; 17 candidate records; recorded wall time 2002.3 s.
- V2: 18 `ok`, 2 `truncated`; 9 candidate records; recorded wall time 2729.0 s. Three candidates were marked valid; six received zero after timeouts (one preflight, five timed evaluation). No observed candidate specifically tested fabricated distances.
- The top-level GCS state was stale after both manifests completed because the watchdog did not receive/observe the runner's final response. After verifying both manifests and raw call ledgers, the watchdog service was stopped and the GCS state reconciled to complete with an audit note. No raw run files were edited or regenerated.
- A post-hoc local Docker re-check used the registered-source checker on the official V2 selection. V1 and V2 both selected the generation-0 seed (same SHA-256); V2 passed the independent correctness check (valid, recall 1.0, speed ratio 1.1338). This does not establish a V2 benefit. Identical code produced variable wall-clock scores across local re-checks, so timing remains a known source of noise. Provenance: `B1_REGISTERED_RECHECK.json`.
- The frozen feasibility questions about checkpoint continuity and the evaluator implementation are partly supported by these records and offline tests. Because V1 required six runtime starts and the pilot had one seed, no effectiveness, equivalence, or population-level claim follows.
- After reviewing B1, the runner's terminal-state heartbeat race and watchdog recovery path were patched in source. Tests validate reconciliation from complete manifests and refuse to rewrite top-level state when either raw call ledger is incomplete. The e2-micro service stays disabled until a separate protocol and run are ready.
- B2 implementation work adds the `v1_matched` timing control, evaluator code hashes in run manifests, and early refusal of mismatched evaluator content on resume. Focused verification across matched evaluator, V1/V2 evaluator, pilot runner, and watchdog tests: 40 passed, 12 Docker-marked tests skipped; Python compilation and `git diff --check` pass. No model calls were made.
- A separate CPU subprocess smoke used a tiny synthetic kNN set and a candidate that returned valid IDs with fabricated squared distances. V1-matched scored it (`recall=1.0`, positive score); V2 rejected it for a distance mismatch and its execution trace contained only `working_v2_preflight`, with no timed phase. This is an implementation smoke, not B1 data or an efficacy observation. No local GPU was used. The user's active `b12eval` Colab session was left untouched; the short-lived `paper2-smoke` session created during setup was stopped immediately.
- A local Docker smoke then loaded the actual B1 saved best programs from GCS and exercised them on a tiny synthetic set: V1-matched produced a score; V2 passed its correctness checks. A second Docker run with fabricated distances showed V2 preflight rejection/no timed phase. No LLM call or local GPU was used; these are harness checks, not additions to the B1 experiment.
- The focused opt-in Docker regression test is reproducible and passed: `RUN_DOCKER_TESTS=1 uv run pytest -q tests/test_oe_evaluator_v1_matched.py` (2 passed). It uses the local Docker image and a small synthetic dataset only; no GPU or model service is involved.

## Durability gate for any future run

Do not start the service or treat a future study as running independently until all of these checks pass:

1. The e2-micro systemd service is `active` and `enabled`.
2. GCS `paper2-v2/b1/state.json` reports `phase=running` and the pinned `git_sha` and protocol hash.
3. Per-call JSONL records are present under the pilot's GCS `runs/` prefix and continue to advance.
4. The watchdog can reconnect to the Colab session and the persisted account balance is above the reserve.

If the Colab runtime is interrupted, use the GCS checkpoint prefix as the durable source of state. Do not start a second runner against the same prefix while the first runner is alive.

## Reproducibility records

The preregistration is `PREREGISTRATION.md`; any deviations must be appended to `DEVIATIONS.md` with a timestamp, reason, and affected records. Preserve the exact Git SHA, protocol SHA-256, model digest, evaluator version, and raw call logs with the final results.

## B2 budget and durability checkpoint — 2026-11-20

- GitHub `main` contains the separate runner/watchdog and no-call rehearsal implementation. The remaining untracked benchmark, Agensi, and skill directories are pre-existing and untouched.
- e2-micro `paper2-colab-watchdog` is running; its B1 and B-12 systemd services currently report inactive. Colab still reports one pre-existing active L4 assignment. The B2 session is absent/stopped after rehearsal.
- A fresh `paper2-b2-estimation` runtime uploaded a sentinel to `gs://oe-colab-scale-0903767759/paper2-v2/b2-estimation-2026-11-20/no-call-rehearsal/rehearsal-1795163650/`, was stopped, and a newly allocated runtime restored the sentinel and passed a synthetic terminal-state roundtrip. No model calls were made. The balance changed from 208.65 to 208.49 CU.
- This verifies GCS persistence across Colab runtime recreation. It does not yet verify the detached systemd service survives laptop shutdown; that service has not been installed or started. The B2 protocol remains draft and live calls remain gated off.
