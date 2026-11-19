# Colab scale study: MoE approaches 31B circle-packing scores at about 24% of its cost; LC pairs improve kNN in 8/9 runs

**Snapshot:** 2026-11-19 (Europe/Kyiv)
**Study:** OpenEvolve with Gemma 4 E4B, 12B, 26B A4B (MoE) and 31B on Colab L4, pre-registered at tag `colab-scale-prereg` ([PREREGISTRATION.md](colab_scale/PREREGISTRATION.md))
**Status:** closed. 167 of 179 planned runs complete. The 12 missing runs are all MoE LC runs that need Argon: 10 never started and 2 stopped at the quota.

## Summary

- **RQ-scale (circle packing).** Median shipped verified sum of radii in arm L, by model:

  | model | median |
  |---|---|
  | E4B | 1.63 |
  | 12B | 1.63 |
  | MoE | 2.17 |
  | 31B | 2.25 (5 seeds) |

  - The MoE is much closer to the 31B than to the E4B.
  - A 100-call MoE run cost about 0.9 compute units, against about 3.9 for a 31B run. E4B used slightly fewer units per median score point (0.40 vs 0.43), while the MoE had a higher absolute median score.
  - Arm L beat arm B for E4B, 12B and the MoE, but not for 31B: B reached 2.40 against 2.25 for L.
- **RQ-cloud (MoE).** With about 10% of mutations from Argon (LC), the selected verified score was higher in 7 of 9 circle-packing pairs (median 2.63 against 2.19; the record is 2.635). On kNN the 2026-11-08 analysis found LC higher in all 9 pairs; under the serial re-timing it is higher in 8 of 9, since the seed-909 LC program (0.159) is slower than the seed (0.223). The later paired tests are exploratory and were not pre-registered. Full analysis: [LC_RESULTS_2026-11-08.md](colab_scale/LC_RESULTS_2026-11-08.md).
- **RQ-budget (MoE circle packing, 1,000 calls).**
  - Longer search found better valid packings, 2.20–2.43 against a median of 2.17 at 100 calls.
  - But selection got worse. 57% of above-seed candidate evaluations were false improvements, and two of three runs shipped an invalid program.
  - The 1,000-call best-of-N control shipped 1.92–2.07.
- **RQ-hardware (12B).** The Colab and laptop 12B results differ, but the two used different model builds, so hardware alone cannot be compared. On Colab, the 12B best-of-N arm never produced a usable circle-packing candidate because that build misquotes one line of the seed.
- **False improvements appear for every model**, as expected. They are most frequent for E4B (47–52% of above-seed candidates) and in the long runs.
- **kNN.** Without Argon, no model improved kNN reliably. The seed program was shipped in 54 of 63 Gemma-only runs, and only 6 shipped a program that beat it. The 500-call MoE runs never produced a candidate the working evaluator ranked above the seed. In the LC arm, 8 of 9 runs shipped a valid, faster program; the median score was 1.0 against 0.223 for the seed. Still, 28% of the LC candidate evaluations ranked above the seed were not better on serial re-check. kNN scores come from serial re-timing (deviation 7).

## Runs and deviations

| model | task | L | B | LC | long L | long B |
|---|---|---:|---:|---:|---:|---:|
| E4B | circle packing, kNN | 10 + 10 | 10 + 10 | — | — | — |
| 12B | circle packing, kNN | 10 + 10 | 10 + 10 | — | — | — |
| MoE | circle packing | 15 | 10 | 9 of 15 | 3 (1,000 calls) | 3 (1,000 calls, exploratory) |
| MoE | kNN | 15 | 10 | 9 of 15 | 3 (500 calls) | — |
| 31B | circle packing | 5 | 5 | — | — | — |

MoE L has 15 runs per task. Seeds 101–1010 belong to the pre-registered size ladder. Seeds 1111–1515 are the unpaired L controls of the 2026-11-07 LC extension, and they are kept out of the size-ladder figures, as [LC_EXTENSION_2026-11-07.md](colab_scale/LC_EXTENSION_2026-11-07.md) requires. All deviations are in [DEVIATIONS.md](colab_scale/DEVIATIONS.md); the ones that bear on interpretation:

1. **31B** was cut to circle packing only, with 5 seeds per arm (2026-11-05 and 2026-11-06). Two 31B streams on one L4 ran at 3.4 tokens/s each, so from 2026-11-04 31B ran one run at a time, while the other models ran two at once. Its cost per run is therefore measured under different serving.
2. **12B build drift** (2026-11-02, 2026-11-13). Colab pulled a rebuilt `gemma4:12b` (`6114515d63c1`), not the laptop's `4eb23ef187e2`. In 998 of 1,000 circle-packing best-of-N replies this build misquotes the comment line `# are inside the square and don't overlap` in its SEARCH block, so no diff applies and every 12B B run ships the seed.
3. **The MoE best-of-N long control** (1,000 calls) was added on 2026-11-05 and is exploratory, not pre-registered.
4. **Spending cap.** It was raised from 330 to 380 units on 2026-11-02, and 363.21 units were spent in all.
5. **LC.** The extension planned 30 LC runs, and the study stopped at 18 complete ones (2026-11-08). The remaining Gemma-only runs finished on 2026-11-11.
6. **VM interruptions and watchdog restarts** occurred as expected. Runs resumed from logged calls and checkpoints, and none was restarted from scratch.
7. **kNN re-timing of the endpoint** (2026-11-19). The first kNN re-check ran up to 16 sandboxes at once. Under that load the identical seed program scored anywhere from 0.02 to 1.0, because the score includes a wall-clock speed penalty. The serial timing files contain eight observations for the seed code hash: five designated calibration repetitions and three more when that same code was selected as a shipped program. The main analysis uses the median across all eight (0.2235; range 0.161–0.244); using only the five designated calibration repetitions gives 0.238 (range 0.192–0.244). The 18 distinct shipped program hashes each have three serial observations; the seed hash is one of those 18. Validity verdicts, which do not depend on timing, come from the full re-check and agree with the serial runs.
8. **kNN re-timing of above-seed candidates** (2026-11-19). Across the Colab study, 231 above-seed candidate evaluations included 10 invalid programs and 221 valid evaluations representing 214 distinct code hashes; all 214 valid hashes were serially timed. The pilot adds 14 further distinct valid hashes, for 228 distinct valid programs across both studies. Each non-seed hash was timed one to three times: three for programs scoring 0.15–0.30 (the band around the seed's 0.223 median), one otherwise. The seed hash has eight total observations as described above. About 1,900 candidates below the seed's working score were never re-checked, so the all-candidate best-verified kNN endpoint is not reported. The published kNN false-improvement denominator counts candidate evaluations, not distinct hashes.

The independent re-checks ran on the laptop in the pilot's Docker sandbox (`sha256:aae97353…7ae2`) with the re-check code at the pilot's registration tag. As in the pilot, the circle-packing re-check reports the raw sum of radii; the ratio to 2.635 is the sum divided by 2.635.

## RQ-scale: model size (circle packing)

Primary endpoint: the re-check score of the working-best candidate, or the seed's verified score (0 for circle packing) if that candidate fails. Seeds 101–1010 for E4B, 12B and MoE; 101–505 for 31B.

| model | arm | runs | shipped verified, median [range] | runs shipping invalid | best verified, median [range] | false / above-seed |
|---|---|---:|---:|---:|---:|---:|
| E4B | L | 10 | 1.628 [0, 1.887] | 1 | 1.649 [1.433, 1.887] | 239 / 505 (47%) |
| E4B | B | 10 | 0.851 [0, 1.920] | 5 | 1.740 [1.608, 1.920] | 213 / 408 (52%) |
| 12B | L | 10 | 1.633 [0, 2.283] | 3 | 1.905 [1.300, 2.283] | 103 / 423 (24%) |
| 12B | B | 10 | 0 (seed in every run) | — | 0 | 0 / 0 † |
| MoE | L | 10 | 2.170 [0, 2.299] | 2 | 2.170 [1.896, 2.299] | 129 / 421 (31%) |
| MoE | B | 10 | 1.911 [0, 2.179] | 1 | 1.911 [1.769, 2.179] | 117 / 710 (16%) |
| 31B | L | 5 | 2.249 [1.827, 2.369] | 0 | 2.249 [1.827, 2.369] | 56 / 404 (14%) |
| 31B | B | 5 | 2.400 [0, 2.450] | 1 | 2.400 [2.340, 2.450] | 90 / 336 (27%) |

† No 12B B reply applied to the seed (deviation 2), so no candidate was ever produced.

**Cost.** While a Colab session was live, the balance fell by 1.92 units per hour, measured from the watchdog log over 177 VM-hours. The table gives the median wall time per 100-call circle-packing run, and units attributed from that log. Units for MoE L are estimated as 1.92 / 2 per run-hour, because those runs predate the log.

| model | L: hours / units per run | B: hours / units per run | L: units per point of median score |
|---|---:|---:|---:|
| E4B | 0.66 / 0.65 | 0.63 / 0.65 | 0.40 |
| 12B | 1.12 / 1.13 | 0.74 / 0.73 | 0.69 |
| MoE | 0.97 / ~0.93 | 0.65 / 0.61 | 0.43 |
| 31B | 2.05 / 3.88 | 1.31 / 2.58 | 1.73 |

Against the expectations stated in advance:

- **"Verified score rises with model size"** holds broadly. E4B and 12B are at about the same level, and the MoE and 31B are about 0.5–0.6 higher.
- **"The MoE lands between 12B and 31B"** holds descriptively: its median is 0.08 below 31B at about a quarter of the units per run. E4B is slightly cheaper per median score point, but at a lower absolute score.
- **"L > B for every model"** holds for the observed medians in E4B, 12B and the MoE; the 12B B arm produced no candidate because of a build-specific patch-application failure. It does not hold for 31B, where B reached 2.40 against 2.25, with 5 seeds per arm.
- **"False improvements present for every model"** holds.

## RQ-cloud: Argon assistance (MoE)

The paired analysis is in [LC_RESULTS_2026-11-08.md](colab_scale/LC_RESULTS_2026-11-08.md) and is not repeated here. In brief, over 9 complete pairs per task (seeds 101–909):

- **Circle packing:** LC median 2.630 against L 2.192. LC was higher in 7 pairs, tied in 1 and lower in 1.
- **kNN (2026-11-08 timing):** LC median 0.826 against L 0.022, which is the seed's score there. LC was higher in all 9 pairs. The serial re-timing in this report gives LC 1.000 against 0.223 and 8 of 9 pairs (see the kNN section).
- **Cost:** the 18 complete LC runs made 198 Argon calls with 4.64 M tokens.
- **False improvements on circle packing** were about as frequent in the pooled data (29% vs 30%). The paired exploratory comparison is inconclusive: median LC-minus-L run-rate difference −0.05, 95% bootstrap interval [−0.11, 0.22]; this does not establish that the rates are equal or that assistance had no effect.

The circle-packing re-check in this report reproduces the LC false-improvement count of that analysis exactly (140 / 486).

## RQ-budget: search length (MoE)

| run | calls | shipped verified (s101 / s202 / s303) | best valid candidate | false / above-seed | truncated replies |
|---|---:|---|---|---:|---:|
| L, 100 calls (seeds 101–1010) | 100 | median 2.170 | median 2.170 | 31% | 15% |
| L long | 1,000 | 2.432 / 0 / 0 | 2.432 / 2.204 / 2.258 | 607 / 1,064 (57%) | 49% |
| B long (exploratory) | 1,000 | 1.936 / 2.072 / 1.924 | same | 358 / 2,089 (17%) | 3% |
| LC, 100 calls | 100 | median 2.630 | median 2.634 | 29% | 26% |

- **Longer evolution found better valid packings** (up to 2.43), and every seed ended above the long best-of-N control.
- **But the working evaluator selected an invalid program in two of three long runs,** so their primary endpoint is 0. The share of false improvements nearly doubles with ten times the budget, and half of the replies hit the 2,048-token output limit as programs grew.
- **Ten times more Gemma calls stayed well short of what 10% Argon reached in 100 calls:** 2.43 at best, against a median of 2.63.

**kNN, 500 calls (3 runs).** The working evaluator ranked no candidate above the seed in any of the three runs, so all three shipped the seed (0.223). Longer Gemma-only search did not move kNN at all.

## RQ-hardware: 12B on Colab vs the laptop

| arm | Colab (s101 / s202 / s303) | laptop pilot (s101 / s202 / s303) |
|---|---|---|
| L | 0 / 1.655 / 1.847 | 2.120 / 1.596 / 1.773 |
| B | 0 / 0 / 0 | 1.793 / 0 / 0 |

- **Medians are close** (1.655 vs 1.773 in L), but the comparison is confounded. The two used different model builds (deviation 2), different Ollama versions, and different candidate execution (a subprocess on the VM against Docker on the laptop).
- **Colab ran about 2.3 times faster:** a 100-call circle-packing L run took a median of 1.12 h on the L4 against 2.48–2.68 h on the laptop.
- **What the build drift shows** is that the same nominal model tag can behave very differently in a diff-based loop.
- **kNN** (seeds 101–303): Colab L shipped the seed in all three runs, and Colab B shipped a faster valid program for seed 101 (score 1.0) and the seed otherwise. On the laptop every shipped kNN program failed the re-check, so all six runs scored the seed.

## kNN (all research questions)

Primary endpoint from the serial re-timing (deviation 7). The seed code hash has eight serial observations (five designated calibration and three selected-shipped repeats); the pooled median is 0.223 (range 0.161–0.244). The five designated calibration observations alone have median 0.238 (range 0.192–0.244). Seeds 101–1010 for E4B, 12B and MoE L; 101–909 for MoE LC. 31B has no kNN runs.

| model | arm | runs | shipped verified, median [range] | runs shipping the seed | shipped program beats seed | false / above-seed evaluations |
|---|---|---:|---:|---:|---:|---:|
| E4B | L | 10 | 0.223 [0.210, 1.000] | 8 | 1 | 1 / 2 |
| E4B | B | 10 | 0.223 [0.223, 0.241] | 9 | 1 | 0 / 1 |
| 12B | L | 10 | 0.223 [0.223, 0.483] | 9 | 1 | 6 / 9 |
| 12B | B | 10 | 0.223 [0.074, 1.000] | 6 | 3 | 1 / 5 |
| MoE | L | 10 | 0.223 [0.223, 0.223] | 9 | 0 | 1 / 1 |
| MoE | B | 10 | 0.223 (seed in every run) | 10 | 0 | 0 / 0 |
| MoE | LC | 9 | **1.000** [0.159, 1.000] | 0 | 8 | 60 / 213 (28%) |
| MoE | L, 500 calls | 3 | 0.223 (seed in every run) | 3 | 0 | 0 / 0 |

- **Without Argon, kNN barely moved.** Across 63 Gemma-only runs the working evaluator rarely ranked any candidate above the seed (18 such candidate evaluations in total). Six runs shipped a program that beat the seed, mostly for the smaller models, and the MoE never did. The registered descriptive results do not support the expected L-over-B or size trend on kNN; the exploratory tests detect no reliable L-vs-B differences, which is not evidence of equivalence.
- **With Argon, it did.** Every LC run shipped a program other than the seed, and 8 of 9 beat it, 5 of them at the score ceiling of 1.0 (at least twice as fast as exact search at full recall). The LC analysis of 2026-11-08 reports a median of 0.826 for the same runs and LC above L in all nine pairs; that re-check ran at a different time and load. Under the serial re-timing LC is above L in 8 of 9 pairs: the seed-909 LC program is valid but slower than the seed (0.159 against 0.223).
- **False improvements on kNN** use the same serial re-timing for every valid above-seed code hash (214 in Colab; 228 across both studies, deviation 8). For LC, 60 of 213 above-seed candidate evaluations were false: 4 invalid and 56 valid but not faster than the pooled seed score. The 2026-11-08 analysis counted 4/213 using a seed score measured under parallel load (0.022); the serial pass supersedes that timing-sensitive comparison. Gemma-only runs produced too few above-seed candidate evaluations (18 in 63 runs) for a stable rate estimate.
- **Measurement note.** Among 71 code hashes with repeated serial timings, the median within-program score range was 0.018; 16 had a range above 0.05. The largest range was 0.490, for one program whose repeated scores were 0.510, 1.000 and 1.000. This residual timing variation matters near the 1.0 speed ceiling.

## Exploratory statistics (not pre-registered)

Both pre-registrations specify descriptive reporting only. The tests below were added after the results were known and should be read as exploratory. Script: [`scripts/analysis/oe_exploratory_tests.py`](../scripts/analysis/oe_exploratory_tests.py); output: [`colab_scale/results/exploratory_tests_2026-11-19.json`](colab_scale/results/exploratory_tests_2026-11-19.json). Endpoints are the registered shipped verified scores (kNN from the serial re-timing). Bootstrap intervals use 10,000 resamples of the paired differences. Paired p-values are unadjusted and exploratory; the size-ladder pairwise tests and nine L-vs-B tests use Holm adjustments as shown.

| comparison | n | result | test, two-sided |
|---|---:|---|---|
| MoE LC vs L, circle packing | 9 pairs | 7 better / 1 equal / 1 worse; median Δ +0.40, 95% CI [0, 0.57] | sign p = 0.070; Wilcoxon p = 0.15 |
| MoE LC vs L, kNN | 9 pairs | 8 / 0 / 1; median Δ +0.78, 95% CI [0.19, 0.78] | sign p = 0.039; Wilcoxon p = 0.0078 |
| circle packing L across E4B, 12B, MoE, 31B | 10/10/10/5 | medians 1.63 / 1.63 / 2.17 / 2.25 | Kruskal–Wallis p = 0.015 |
| MoE vs 31B, circle L | 10 vs 5 | 2.17 vs 2.25 | Mann–Whitney p = 0.426 (Holm 1.000) |
| MoE vs 12B / vs E4B, circle L | 10 vs 10 | 2.17 vs 1.63 / 1.63 | Mann–Whitney p = 0.087 / 0.021 (Holm 0.173 / 0.063) |
| L vs B, circle, per model | 10 vs 10 (31B 5, pilot 3) | 12B p = 0.002 (Holm 0.020), MoE p = 0.037 (Holm 0.30); E4B, 31B, pilot not significant | Mann–Whitney, Holm over 9 tests |
| L vs B, kNN, per model | 10 vs 10 | identical or near-identical (mostly the seed) | not significant |
| false-improvement rate per run, MoE LC vs L, circle | 9 pairs | median Δ −0.05, 95% CI [−0.11, 0.22] | sign p = 0.508; Wilcoxon p = 0.910 |

- **The largest observed paired LC-vs-L separation is on kNN.** LC scored higher in 8 of 9 pairs under serial re-timing; these post-hoc tests are exploratory and do not establish a general causal effect.
- **Circle-packing outcomes differ across the four model groups overall** (Kruskal–Wallis p = 0.015, unadjusted). The three pairwise MoE comparisons do not pass Holm correction; the MoE-vs-31B test does not establish equivalence.
- **The only L-vs-B difference that survives the Holm correction is 12B on circle packing.** This separation coincides with a known build-specific patch-application failure in B (no candidate was produced; deviation 2), so it does not estimate a general algorithmic L-vs-B effect.
- **The circle-packing false-improvement comparison is inconclusive.** Its paired interval spans both a decrease and an increase in per-run rates, so the data do not establish either a reduction or no effect.

## Data

- Run list and status: [FINAL_RUNS_2026-11-11.json](colab_scale/FINAL_RUNS_2026-11-11.json).
- LC analysis tables: [colab_scale/results/](colab_scale/results/).
- Serial kNN re-checks: [`knn_serial_retiming_2026-11-19.jsonl`](colab_scale/results/knn_serial_retiming_2026-11-19.jsonl) and [`knn_serial_above_seed_2026-11-19.jsonl`](colab_scale/results/knn_serial_above_seed_2026-11-19.jsonl). Run-level exploratory results and the calibration sensitivity are in [`exploratory_tests_2026-11-19.json`](colab_scale/results/exploratory_tests_2026-11-19.json).
- Re-run the analysis when the original run folders are available locally: `PYTHONPATH=/path/to/orchestrator python scripts/analysis/oe_exploratory_tests.py /path/to/colab/runs /path/to/pilot/runs docs/colab_scale/results /tmp/exploratory-results.json`. The serial inputs in `docs/colab_scale/results/` are tracked; the complete raw run folders are archived separately and are not tracked in Git.
- Raw runs: `gs://oe-colab-scale-0903767759/runs/` and the laptop copy `benchmarks/colab_scale/runs/` (not tracked). Per-candidate re-checks are in each run's `rechecked.jsonl`.
