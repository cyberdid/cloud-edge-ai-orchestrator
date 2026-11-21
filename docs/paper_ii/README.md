# When Fitness Scores Mislead

**Revision 3.1 draft preprint package, 21 November 2026.** This is an empirical article, separate from *Beyond Passing Tests*. It follows the repository's bilingual LaTeX format and has not been submitted for peer review or deposited with an archive.

## Files

- English PDF: [`article-en.pdf`](article-en.pdf)
- Ukrainian PDF: [`article-uk.pdf`](article-uk.pdf)
- LaTeX sources: [`tex/en/main.tex`](tex/en/main.tex), [`tex/uk/main.tex`](tex/uk/main.tex)
- References: [`tex/references.bib`](tex/references.bib)
- Citation metadata: [`CITATION.cff`](CITATION.cff)
- Licenses: [`LICENSE-ARTICLE`](LICENSE-ARTICLE) (CC BY 4.0), [`LICENSE-CODE`](LICENSE-CODE) (Apache-2.0)
- Figure generator: [`../../scripts/paper_ii/make_figures.py`](../../scripts/paper_ii/make_figures.py)
- Selection and serial kNN audit: [`../../scripts/paper_ii/revision_metrics.py`](../../scripts/paper_ii/revision_metrics.py), [`results/revision_metrics.json`](results/revision_metrics.json)
- Revision and evidence notes: [`REVISION_NOTES.md`](REVISION_NOTES.md)
- SHA-256 manifest: [`SHA256SUMS`](SHA256SUMS)

## B2 follow-up evidence

- Frozen protocol configuration: [`../../experiments/openevolve/configs/b2_estimation_2026-11-20.json`](../../experiments/openevolve/configs/b2_estimation_2026-11-20.json)
- Execution and final results: [`../evaluator_v2/B2_ESTIMATION_DRAFT.md`](../evaluator_v2/B2_ESTIMATION_DRAFT.md)
- Selected-program independent checks: [`../evaluator_v2/B2_SELECTED_RECHECK_2026-11-21.json`](../evaluator_v2/B2_SELECTED_RECHECK_2026-11-21.json)
- Descriptive paired summary: [`../evaluator_v2/B2_ESTIMATION_SUMMARY_2026-11-21.json`](../evaluator_v2/B2_ESTIMATION_SUMMARY_2026-11-21.json)

## Scope and interpretation

The manuscript reports two original descriptive studies of OpenEvolve 0.4.0: a Colab L4 scale study and a 12B laptop pilot. Revision 3 adds a later, separately frozen six-pair B2 estimation cohort comparing distance-gated V2 with timing-matched V1 on approximate kNN. Revision 3.1 clarifies the paired-score distribution and translates all labels in the English selection-audit figure. It treats circle packing and approximate kNN as bounded test problems. The analysis does not establish that one evolutionary algorithm is generally superior, that model scaling is monotonic, that cloud assistance has a causal benefit, or that evaluator V2 reduces selection failures.

The Colab plan began at 156 runs and was amended during the campaign to 179 after interim L/B outcomes and early LC operational logs had been reviewed; 167 runs completed. The laptop pilot completed its 12 planned L/B runs, while its separate LC condition was stopped after quota exhaustion. The serial kNN retiming and inferential comparisons were performed after the original campaigns and are identified as a correction and exploratory analyses throughout the article. The B2 protocol was frozen in Git before its model calls; all six pairs and 240 calls completed. The selected programs passed independent checks in 6/6 V1-matched runs and 5/6 V2 runs. This is a small estimation cohort with a source-visible holdout: no confirmatory test was planned or computed, and it does not establish efficacy, equivalence or generalization. The 12B Colab best-of-N result is affected by a build-specific patch-application failure. Cross-device 12B results are confounded by different model builds and execution environments. Full raw campaign traces and B2 call logs are retained separately and are not part of this Git snapshot; the repository contains the original preregistrations and amendments, B2's frozen configuration and protocol hash, selected-program re-checks, deviations, analysis and code.

## Rebuild

From the repository root, with Python, Matplotlib, NumPy, SciPy and Tectonic available. The PDFs here were built with Python 3.12, Matplotlib 3.11.1, NumPy 1.26.4 and Tectonic 0.15.0:

```bash
bash scripts/paper_ii/build.sh
```

The command first rebuilds `results/revision_metrics.json` from committed run inventories, CSV summaries and serial re-checks, regenerates both audit and manuscript figures, then compiles the English and Ukrainian articles. It refreshes `SHA256SUMS` from its existing file list and verifies the finished package, including the linked B2 protocol and results artifacts. It makes no model calls, network requests or sandbox calls and starts no benchmark. Tectonic intermediates are written under `tmp/pdfs/paper-ii/`.

The immutable experimental source snapshot used for the original run-level artifacts is commit `57a55d3d23d6c7c3dd7be3abfc1a0d9c758fc6b5`. Revision 3.1 and the B2 follow-up artifacts are distributed on the repository's `main` branch. The source commit, B2 frozen source commit and amendment chronology are identified in the manuscript; do not treat later analysis files as part of the earlier experiment snapshots.

Suggested citation: Pylypenko, O. (2026). *When Fitness Scores Mislead: An Empirical Study of LLM-Guided Program Evolution*. Preprint draft, revision 3.1, 21 November 2026. No DOI or peer-reviewed venue is claimed.

## License and transparency

Article text, figures and research summaries are licensed under CC BY 4.0. The figure-generation script is Apache-2.0. The author reports PhD-student status at the State University of Trade and Economics and cooperation with its Department of Software Engineering and Cybersecurity. Experiments were run using the author's equipment and accounts. AI assistants supported analysis code, drafting and translation; the author remains responsible for the final manuscript and its claims.
