# MoE LC extension: 30 runs in total

Owner decision: 2026-11-07 (Europe/Kyiv), before any extension run.
Plan revision: `lc-30-2026-11-07`. This is an amendment to the ongoing descriptive
study, not a replacement of the original pre-registration or an independent
confirmatory replication. Original registration and scientific tags stay intact.

## Scope and provenance

The owner requested **30 LC runs total**, not 30 additional runs. There are two
tasks (`circle_packing`, `knn`), with 15 seeds each. At the decision, the original
149-run plan had 139 completions; its first LC runs were in progress. Earlier L/B
results and early LC operational logs had already been inspected. The extension
must not be described as the originally pre-registered sample size. Its size is
an owner-selected resource allocation, not a statistical power result.

| Cohort | Seeds, on each task | LC runs | New L controls |
| --- | --- | ---: | ---: |
| Original LC | 101, 202, 303, 404, 505 | 10 | 0; already present |
| Extension with existing controls | 606, 707, 808, 909, 1010 | 10 | 0; already present |
| Extension with new controls | 1111, 1212, 1313, 1414, 1515 | 10 | 10 |
| Total | 15 seeds per task | 30 | 10 |

The current 149-run list is preserved exactly as a prefix. The extension appends
30 runs (20 LC + 10 L), so the full plan contains **179 runs**. B retains its
existing ten seeds per task; added controls address RQ-cloud (LC versus L),
not an expanded three-arm B/L/LC comparison on 15 seeds.
RQ-scale comparisons retain the original seed cohorts; do not pool the new L
controls into MoE's size-ladder baseline while other models keep ten seeds. The
generic laptop-pilot aggregator does not separate Colab's long-run budgets or
these cohorts automatically; filter the intended cohort and budget explicitly.

## Fixed order and settings

1. Finish the existing 149-run list without interrupting active runs.
2. For each seed 606, 707, 808, 909, 1010: LC circle packing, then LC kNN.
3. For each seed 1111, 1212, 1313, 1414, 1515: L circle packing, L kNN,
   LC circle packing, LC kNN. Each new L pair precedes the corresponding LC pair.

Use the existing two-run serving policy; tasks are balanced within each extension
seed if the budget truncates the tail. All added runs use `gemma4:26b`, 100 logged
calls and the existing router, prompts, seed programs, candidate executor,
evaluators and generation settings. LC selects Argon with seeded probability 0.1,
not an exactly ten-call cloud quota. No prompt, temperature, evaluator, retry budget
or endpoint changes. Model digests continue to be recorded by the driver.
The owner's current VM UI reported `agy` 1.5.2; the original registration specified
1.4.8. This version difference predates this extension and must be retained in
environment reporting, not silently treated as identical software.

## Budget and stopping

The existing watchdog cap remains **380 compute units spent from 399.47**
(nominal stopping balance 19.47). At the budget check the balance was 85.66,
the rate was 1.94 units/h, and one assignment was active. Thus about 66.19 units
or 34.1 L4-hours remained before the cap; this is a budget envelope, not evidence
that all 30 LC runs and their controls will fit.

Do not buy compute or raise the cap automatically. Keep existing quota/auth pause,
cloud sync and resume rules. Never substitute Gemma for Argon. If the cap or quota
prevents completion, report scheduled, completed and missing runs and the reason;
do not replace seeds based on observed performance. The fixed order controls which
runs are attempted, not a result-dependent stopping rule.

## Analysis and independent checks

Primary endpoint remains the **shipped verified score**, obtained by the existing
independent laptop Docker re-check; live working scores are provisional. Report
each task separately, with matched L/LC differences, medians, ranges, run-level
outcomes and measured costs. Keep the three cohorts above identifiable and provide
the original five-seed comparison separately from expanded descriptive summaries.
Do not claim a pre-registered significance test, statistical equivalence or
generalization beyond these two tasks. More seeds measure variability on these
tasks; they do not add new task families.

If a control or its LC partner is missing, explicitly list the incomplete pair.
A paired summary uses only fully re-checked pairs and states its denominator;
also retain all available unpaired results so budget truncation is visible.

## Live deployment

The running original driver retains its instantiated 149-run list. The GCE watchdog
is updated while it runs. Only after state reports `phase=done`, `done=149`,
`total=149`, no block and sufficient budget does the watchdog upload the extended
`study.py` and attach a new study client on the **same VM**. The new driver skips
the 149 completed manifests and executes the extension. No running candidate is
interrupted and the VM's owner-managed `agy` login is retained. If the assignment
has ended, normal recovery uses the extended plan, with a new owner login if needed.
Unknown assignments retain duplicate protection and require manual attention when
they cannot safely be managed.

Live state continues to show total 149 until handoff; the extended driver writes
total 179 and `plan_revision=lc-30-2026-11-07`. An upload failure leaves the completed
driver untouched and is retried on a later tick; a failed client start blocks with
`extension_start_failed` for inspection. No automatic kernel restart is used by
the in-place handoff.
