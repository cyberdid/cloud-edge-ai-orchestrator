# Revision notes: *When Fitness Scores Mislead*

**Revision:** 2, 19 November 2026

**Scope:** editorial and reproducibility revision of the English and Ukrainian preprint, using existing repository evidence. No new LLM campaign, benchmark run, or sandbox re-check was performed.

## Changes made

### 1. Registration chronology and deviations

- Corrected the description of the Colab study: its 30 October registration specified 156 runs and descriptive analysis only.
- Recorded the dated plan changes: 145 runs on 5 November; 149 on 6 November; and an amendment on 7 November to 179 runs, adding 20 LC runs and 10 unpaired L controls.
- Disclosed that the 7 November amendment followed review of earlier L/B outcomes and early LC operational logs, with 139/149 original-plan runs complete and LC already underway. The additional sample was resource allocation, not a power calculation.
- Reconciled final planned and completed counts from `docs/colab_scale/FINAL_RUNS_2026-11-11.json`: 179 planned, 167 completed, 10 not started, 2 quota-paused partial. The 18 completed LC runs form nine matched pairs per task; 10 added L runs are unpaired and excluded from those comparisons.
- Clarified the laptop pilot: 18 runs were registered; all 12 L/B runs completed; the separate six-run LC condition was excluded after quota exhaustion, with one partial LC run.
- Added a machine-checked inventory assertion to `scripts/paper_ii/revision_metrics.py` so an unnoticed change to the inventory fails the rebuild.

### 2. Selection regret and false-improvement interpretation

- Added a selection audit comparing published best-valid circle-packing values with the selected verified endpoints for the three 1,000-call L runs. The approximate regrets, limited by the source report's three-decimal precision, are 0.000, 2.204 and 2.258; the median is 2.204. Two of the three runs selected an invalid candidate and fell back to the invalid seed score of zero.
- Kept the B-long results explicitly exploratory and reported them at the precision available in the public summary.
- Corrected the LC kNN false-improvement claim: 60/213 (28.2%) above-working-seed **candidate evaluations** were false improvements; 4 failed validity checks and 56 passed validity but had a combined verified score no greater than the pooled seed. This does not mean that the 56 were simply “not faster”: the score combines recall and a relative-time penalty.
- Stated that evaluation denominators include repeated evaluations of code hashes and therefore do not represent unique programs.
- Added the omitted 31B circle-packing rows to the false-improvement table: L 56/404 (13.9%) and B 90/336 (26.8%); kNN was not run for 31B.
- Did not calculate all-candidate kNN regret. About 1,900 below-seed candidates were not serially re-timed, so the public data cannot establish the best verified result over the complete candidate pool.

### 3. kNN component and ranking audit

- Added the selected LC programs' independent distance-contract validity, holdout-recall range, and per-program median maximum holdout speed ratios. Ratios are explicitly relative measures, not absolute latency.
- Added a limited rank-stability summary across the ten unique selected code hashes (one shared seed and nine selected LC programs): Kendall's tau-b for ordinal repeat indices, top-five overlap, and the seed-threshold sensitivity. Repeat indices are not synchronized between programs, so this is descriptive only.
- Retained the limitation that serial re-timing only covered the seed and valid above-seed candidates; repeated hashes still vary. No claim is made that this closes the timing-noise problem for all generated candidates.

### 4. Claim scope and related work

- Narrowed the contribution to a bounded empirical audit of OpenEvolve's candidate ranking versus independent verification, including selected-output regret and the serial timing correction.
- Added EvoSort as a directly related audit and Google Cloud's AlphaEvolve evaluator guidance. The manuscript now explicitly recognizes prior evaluator-exploitation and audit work and does not claim to introduce the general idea.
- Described 2.635 as the benchmark's normalization reference, not as a mathematically established upper bound.
- Moved significance tests out of the abstract's main emphasis and repeatedly labels all inferential comparisons post hoc/exploratory. A nonsignificant result is not treated as evidence of equivalence.
- Corrected the Ukrainian Figure 3 caption to identify the Wilcoxon comparison as paired.

### 5. Layout, provenance, and reproducibility

- Replaced the floating Evidence Map table with an in-place captioned table and inserted a page break before appendices so it cannot float ahead of Appendix B.
- Added Figure 4 and its source script and metrics JSON to the evidence map.
- Updated `build.sh` to regenerate the selection metrics and figures before building both PDFs. It performs no model calls, network requests, benchmark runs, or sandbox executions.
- Updated the package README and citation metadata to revision 2, distinguishing the immutable experiment-artifact snapshot (`57a55d3d23d6c7c3dd7be3abfc1a0d9c758fc6b5`) from the later manuscript revision on `main`.
- Cross-checked the Holm-adjusted MoE-vs-31B value against both `exploratory_tests_2026-11-19.json` and `docs/COLAB_SCALE_RESULTS.md`; each reports 0.426 for the unadjusted test and 1.000 after Holm correction. The manuscript's separate 0.426 value is the adjusted result from its different stated family of three MoE pairwise contrasts; this distinction is retained as reported by the relevant artifact.

## Evidence map

| Manuscript item | Source |
|---|---|
| Final Colab inventory and status | `docs/colab_scale/FINAL_RUNS_2026-11-11.json` |
| Registration, amendments, and deviations | `docs/colab_scale/PREREGISTRATION.md`, `LC_EXTENSION_2026-11-07.md`, `DEVIATIONS.md` |
| Long-budget best-valid values | `docs/COLAB_SCALE_RESULTS.md` (published to 3 decimals) |
| Selected run outcomes and post hoc statistics | `docs/colab_scale/results/exploratory_tests_2026-11-19.json` |
| LC/L paired summaries and false-improvement candidate evaluations | `lc_paired_outcomes_2026-11-08.csv`, `lc_run_outcomes_2026-11-08.csv`, `lc_above_seed_candidates_2026-11-08.csv` |
| Serial kNN re-checks and definitions | `knn_serial_retiming_2026-11-19.jsonl`, `knn_serial_above_seed_2026-11-19.jsonl`, `experiments/openevolve/tasks/knn/recheck.py` |
| Rebuilt audit metrics and Figure 4 | `scripts/paper_ii/revision_metrics.py`, `docs/paper_ii/results/revision_metrics.json`, `docs/paper_ii/figures/fig4_selection_audit_{en,uk}.pdf` |

## Known scope limitations

- This revision does not create new experimental evidence. The additional analyses are reconstructed from public machine-readable summaries and serial re-check artifacts; the long-run best-valid values are available only to three decimal places.
- There are two bounded tasks, small seed counts, a partially completed LC campaign, and post hoc inference. The findings do not establish a general scaling law, a causal benefit of cloud assistance, or performance on real repository bugs.
- Full raw run directories and complete model-call traces remain in separate archives, outside this Git snapshot.
