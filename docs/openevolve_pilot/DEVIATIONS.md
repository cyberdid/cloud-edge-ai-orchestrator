# OpenEvolve pilot — deviations and interruptions

Recorded as required by `PREREGISTRATION.md` (tag `oe-pilot-prereg`).

| when | what | effect on data |
|---|---|---|
| 2026-10-29 21:46 | Owner asked to stop the campaign 3 minutes after launch (to resume later). Campaign script and run process were terminated; sandbox containers removed. | `circle_packing_L_s101`: 2 of 100 calls logged, 3 candidates (seed + 2), no checkpoint yet, manifest `incomplete`. On resume the run continues at call index 2 from a fresh OpenEvolve population (no checkpoint existed); the two earlier candidates stay in the logs and are re-checked. No other run had started. |
| 2026-10-30 09:14 | Campaign resumed by the owner (`run_openevolve_pilot.sh run`); code unchanged since the pre-registration except this file (HEAD `80d8229`). | `circle_packing_L_s101` resumed with `--resume` at call index 2. |
