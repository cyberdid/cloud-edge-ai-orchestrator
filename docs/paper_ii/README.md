# When Fitness Scores Mislead

**Revised draft preprint package, 19 November 2026.** This is a new empirical article, separate from *Beyond Passing Tests*. It follows the repository's bilingual LaTeX format. It has not been submitted for peer review or deposited with an archive.

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

## Scope and interpretation

The manuscript reports two pre-registered descriptive studies of OpenEvolve 0.4.0: a Colab L4 scale study and a 12B laptop pilot. It treats circle packing and approximate kNN as bounded test problems. The analysis does not establish that one evolutionary algorithm is generally superior, that model scaling is monotonic, or that cloud assistance has a causal benefit.

The Colab preregistration began at 156 planned runs and was amended during the campaign to 179 after interim L/B outcomes and early LC operational logs had been reviewed; 167 runs completed. The laptop pilot completed its 12 planned L/B runs, while its separate LC condition was stopped after quota exhaustion. The serial kNN retiming and inferential comparisons were performed after the campaigns. They are identified as a correction and exploratory analyses throughout the article. The 12B Colab best-of-N result is affected by a build-specific patch-application failure. Cross-device 12B results are confounded by different model builds and execution environments. Complete raw campaign traces are retained in separate archives and are not part of this Git snapshot; the repository contains preregistrations, dated amendments, deviations, run-level analysis, serial re-check artifacts and analysis code.

## Rebuild

From the repository root, with Python, Matplotlib, NumPy, SciPy and Tectonic available. The PDFs here were built with Python 3.12, Matplotlib 3.11.1, NumPy 1.26.4 and Tectonic 0.15.0:

```bash
bash scripts/paper_ii/build.sh
```

The command first rebuilds `results/revision_metrics.json` from committed run inventories, CSV summaries and serial re-checks, regenerates both audit and manuscript figures, then compiles the English and Ukrainian articles. It makes no model calls, network requests or sandbox calls and starts no benchmark. Tectonic intermediates are written under `tmp/pdfs/paper-ii/`.

The immutable experimental source snapshot used for the underlying run-level artifacts is commit `57a55d3d23d6c7c3dd7be3abfc1a0d9c758fc6b5`. This manuscript revision and its additional selection audit are distributed on the repository's `main` branch. The source commit and amendment chronology are identified in the manuscript; do not treat the later audit files as part of the earlier snapshot.

Suggested citation: Pylypenko, O. (2026). *When Fitness Scores Mislead: An Empirical Study of LLM-Guided Program Evolution*. Preprint draft, 19 November 2026. No DOI or peer-reviewed venue is claimed.

## License and transparency

Article text, figures and research summaries are licensed under CC BY 4.0. The figure-generation script is Apache-2.0. The author reports PhD-student status at the State University of Trade and Economics and cooperation with its Department of Software Engineering and Cybersecurity. Experiments were run using the author's equipment and accounts. AI assistants supported analysis code, drafting and translation; the author remains responsible for the final manuscript and its claims.
