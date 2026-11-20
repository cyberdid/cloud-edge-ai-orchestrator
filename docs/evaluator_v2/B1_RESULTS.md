# B1 evaluator V2 feasibility pilot: results and audit

**Protocol:** `evaluator-v2-pilot-2026-11-19`  
**Protocol SHA-256:** `0a67287d4f2e0bd00dcc9032614efbaa4c8c0bdee17403c27beba59107cbc9c5`  
**Code commit:** `9e45ee6b2cee02e910222c03e84bf680043d43ce`  
**Model:** Ollama `gemma4:12b`, digest `6114515d63c1`; no cloud-model calls  
**Task / seed:** approximate kNN / `20261119`  
**Artifact prefix:** `gs://oe-colab-scale-0903767759/paper2-v2/b1/`

## Scope

B1 was preregistered as a single-seed engineering feasibility pilot. It checks the V1/V2 runner, evaluator gate, checkpoint persistence, and fixed per-condition call budget. It is not powered or designed to establish comparative efficacy, equivalence, or generalization. Do not report a p-value for this pilot.

## Verified execution records

| Condition | Manifest | Calls | Call indices | Call status | Candidate records | Recorded wall time |
|---|---|---:|---|---|---:|---:|
| V1 | `runs/knn_v1_s20261119/manifest.json` | 20 | 0–19, contiguous | 18 `ok`, 2 `truncated` | 17 | 2002.3 s |
| V2 | `runs/knn_v2_s20261119/manifest.json` | 20 | 0–19, contiguous | 18 `ok`, 2 `truncated` | 9 | 2729.0 s |

Both manifests say `complete` and agree on task, arm L, seed, commit, model digest, protocol ID, and protocol hash. Raw ledgers contain 20 Gemma calls per condition, with no duplicate or missing call index. The manifest timestamps record six V1 runtime starts (2026-11-19 15:05–15:30 UTC) and one V2 start (16:04 UTC). See the chronological infrastructure deviations in `DEVIATIONS.md`.

## Evaluator observations

- The V2 candidate log marks three of nine records valid with nonzero scores. Six were assigned zero after timeouts: one preflight timeout and five timed-evaluation timeouts.
- No V2 candidate record documents a fabricated-distance mismatch. The pilot therefore does not empirically show an adversarial candidate being rejected for fabricated distances. The gate's fabricated-distance behavior is established by offline unit tests in `tests/test_oe_evaluator_v2.py` (`test_rejects_correct_shaped_ids_with_fabricated_distances` and `test_v2_rejects_preflight_distance_error_before_timing`).
- V1 and V2 candidate counts differ because they are evaluator and execution traces, not independent task-level observations. Their best raw scores must not be interpreted as a comparative treatment effect.
- **Selection artifact audit (post hoc):** comparing the official `openevolve/best/best_program.py` and `best_program_info.json` against the candidate ledgers showed that V1's official selected score was `0.2433403`, while the maximum candidate-log score was `0.2502050`; V2's official score and ledger maximum both equal `0.2482293`. In both conditions the official selected program SHA-256 matches the hash on the candidate-log maximum row. V1 therefore demonstrates that repeated evaluations of the same code can record different working scores, so the candidate-log maximum is not a reliable proxy for the selected artifact's recorded score. The run-level correctness endpoint is program-identity based and must use the official artifact hash. The B1 raw data are unchanged; analysis code now records and honors official selection for future L/LC runs. This does not change B1's feasibility-only status.
- Each condition recorded 18 complete and 2 truncated model responses. These remain in the fixed budget; there were no unlogged retries.

## Compute and interruption

At final audit, the Colab CLI reported no session, zero active assignments, a usage rate of 0.00 CU/hour, and 211.39 CU remaining. The e2-micro supervisor was stopped at 2026-11-20 07:01 UTC. Its repeated restart/wait cycle followed failure to observe a top-level `phase=complete` response even though the two condition manifests and call ledgers were complete. After verifying those underlying files, the top-level `state.json` was reconciled to `phase=complete` with an explicit audit record. This reconciliation changes only summary state; the raw manifests, call logs, candidate logs, and OpenEvolve checkpoints were left untouched.

## Reproducibility limits and next step

The run shows that the two 20-call conditions and their artifacts can be persisted in GCS, but operational interruption affected V1 and the supervisor's completion handshake. It does not support an effectiveness claim. The watchdog's terminal-state and recovery behavior has now been patched and covered by focused tests; the e2-micro service remains disabled. Before B2, deploy the reviewed source when resuming infrastructure and freeze a separate confirmatory protocol (including sample size and stopping rules) before any further model calls. Preserve B1 as a feasibility record; do not pool it into B2 as though it were confirmatory data.
