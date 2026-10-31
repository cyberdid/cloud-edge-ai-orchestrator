# Colab scale study — deviations and interruptions

Recorded as required by `PREREGISTRATION.md` (tag `colab-scale-prereg`). Raw VM events:
`gs://oe-colab-scale-0903767759/vm_log.jsonl`.

| when (EET) | what | effect on data |
|---|---|---|
| 2026-10-30 ~18:40 | First VM (started 11:32) reclaimed by Colab ~1 h after the laptop (watchdog host and `colab exec` client) went to sleep. No watchdog ran until the laptop woke. | Last sync 18:37; `knn_L_s101` (MoE) stopped at 90 logged calls and resumed on the next VM. ~20 h without a VM; no compute units spent. |
| 2026-10-31 14:55 | Laptop watchdog started VM 2; first `colab exec` hung 10 min, kernel restart + retry worked (designed recovery). | None. |
| 2026-10-31 16:00 | Watchdog moved to an always-on e2-micro (`oe-watchdog`, GCE free tier). Its first tick did not recognise VM 2 (created on the laptop; listed as `?`) and created a duplicate VM; stopped by Claude after ~3 min, before any driver ran on it. Rule fixed: a fresh heartbeat in `state.json` (< 30 min) always means "do not start" (`tests/test_oe_colab_watchdog.py::test_fresh_heartbeat_never_starts_a_second_vm`). | ~0.1 compute units; `state.json` briefly showed `phase: starting` until the running driver's next heartbeat. No run data touched. |
