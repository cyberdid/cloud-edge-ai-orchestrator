# Colab scale study: MoE comes close to 31B on circle packing at a quarter of the cost; Argon helps most

**Snapshot:** 2026-11-19 (Europe/Kyiv) — **draft: kNN sections wait for the independent re-check**
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
  - A 100-call MoE run cost about 0.9 compute units, against about 3.9 for a 31B run, so the MoE is the best choice per compute unit.
  - Arm L beat arm B for E4B, 12B and the MoE, but not for 31B: B reached 2.40 against 2.25 for L.
- **RQ-cloud (MoE).** With about 10% of mutations from Argon (LC), the selected verified score was higher in 7 of 9 circle-packing pairs and in all 9 kNN pairs. Median circle packing was 2.63 against 2.19; the record is 2.635. Full analysis: [LC_RESULTS_2026-11-08.md](colab_scale/LC_RESULTS_2026-11-08.md).
- **RQ-budget (MoE circle packing, 1,000 calls).**
  - Longer search found better valid packings, 2.20–2.43 against a median of 2.17 at 100 calls.
  - But selection got worse. 57% of above-seed candidates were false improvements, and two of three runs shipped an invalid program.
  - The 1,000-call best-of-N control shipped 1.92–2.07.
- **RQ-hardware (12B).** The Colab and laptop 12B results differ, but the two used different model builds, so hardware alone cannot be compared. On Colab, the 12B best-of-N arm never produced a usable circle-packing candidate because that build misquotes one line of the seed.
- **False improvements appear for every model**, as expected. They are most frequent for E4B (47–52% of above-seed candidates) and in the long runs.
- **kNN** results for the size ladder, best-of-N and long runs are pending the re-check. They will be added here.

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
- **"The MoE lands between 12B and 31B and is the best per compute unit"** holds. Its median is 0.08 below 31B at about a quarter of the units per run. E4B is equally cheap per point, but at a far lower level.
- **"L > B for every model"** holds for E4B and the MoE, and formally for 12B, though that B arm failed for a build-specific reason. It does not hold for 31B, where B reached 2.40 against 2.25, with 5 seeds per arm.
- **"False improvements present for every model"** holds.

## RQ-cloud: Argon assistance (MoE)

The paired analysis is in [LC_RESULTS_2026-11-08.md](colab_scale/LC_RESULTS_2026-11-08.md) and is not repeated here. In brief, over 9 complete pairs per task (seeds 101–909):

- **Circle packing:** LC median 2.630 against L 2.192. LC was higher in 7 pairs, tied in 1 and lower in 1.
- **kNN:** LC median 0.826 against L 0.022, which is the seed's score. LC was higher in all 9 pairs.
- **Cost:** the 18 complete LC runs made 198 Argon calls with 4.64 M tokens.
- **False improvements on circle packing** were about as frequent as in L (29% vs 30%), so Argon raised scores without making the working evaluator more reliable.

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

kNN long runs (500 calls): *pending re-check.*

## RQ-hardware: 12B on Colab vs the laptop

| arm | Colab (s101 / s202 / s303) | laptop pilot (s101 / s202 / s303) |
|---|---|---|
| L | 0 / 1.655 / 1.847 | 2.120 / 1.596 / 1.773 |
| B | 0 / 0 / 0 | 1.793 / 0 / 0 |

- **Medians are close** (1.655 vs 1.773 in L), but the comparison is confounded. The two used different model builds (deviation 2), different Ollama versions, and different candidate execution (a subprocess on the VM against Docker on the laptop).
- **Colab ran about 2.3 times faster:** a 100-call circle-packing L run took a median of 1.12 h on the L4 against 2.48–2.68 h on the laptop.
- **What the build drift shows** is that the same nominal model tag can behave very differently in a diff-based loop.

## kNN (all research questions)

*Pending:* the independent re-check of the Colab kNN runs is still running. When it finishes, this section will give the size-ladder table, the best-of-N comparison, the 500-call MoE runs and the 12B hardware comparison for kNN. Until then, the only kNN results available are the MoE L and LC ones in the LC analysis above. MoE L never shipped anything better than the seed in 9 runs.

**Measurement caveat.** The kNN holdout score includes a wall-clock speed penalty. The laptop pilot showed that the same seed program scores differently under different re-check load. kNN comparisons are therefore made within runs, and between arms only when re-checked together.

## Data

- Run list and status: [FINAL_RUNS_2026-11-11.json](colab_scale/FINAL_RUNS_2026-11-11.json).
- LC analysis tables: [colab_scale/results/](colab_scale/results/).
- Raw runs: `gs://oe-colab-scale-0903767759/runs/` and the laptop copy `benchmarks/colab_scale/runs/` (not tracked). Per-candidate re-checks are in each run's `rechecked.jsonl`.
