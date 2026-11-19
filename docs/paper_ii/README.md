# When Fitness Scores Mislead

**Draft preprint package, 19 November 2026.** This is a new empirical article, separate from *Beyond Passing Tests*. It follows the repository's bilingual LaTeX format. It has not been submitted for peer review or deposited with an archive.

## Files

- English PDF: [`article-en.pdf`](article-en.pdf)
- Ukrainian PDF: [`article-uk.pdf`](article-uk.pdf)
- LaTeX sources: [`tex/en/main.tex`](tex/en/main.tex), [`tex/uk/main.tex`](tex/uk/main.tex)
- References: [`tex/references.bib`](tex/references.bib)
- Citation metadata: [`CITATION.cff`](CITATION.cff)
- Licenses: [`LICENSE-ARTICLE`](LICENSE-ARTICLE) (CC BY 4.0), [`LICENSE-CODE`](LICENSE-CODE) (Apache-2.0)
- Figure generator: [`../../scripts/paper_ii/make_figures.py`](../../scripts/paper_ii/make_figures.py)
- SHA-256 manifest: [`SHA256SUMS`](SHA256SUMS)

## Scope and interpretation

The manuscript reports two pre-registered descriptive studies of OpenEvolve 0.4.0: a Colab L4 scale study and a 12B laptop pilot. It treats circle packing and approximate kNN as bounded test problems. The analysis does not establish that one evolutionary algorithm is generally superior, that model scaling is monotonic, or that cloud assistance has a causal benefit.

The serial kNN retiming and statistical comparisons were performed after the campaigns. They are identified as methodological corrections and exploratory analyses throughout the article. The study's 12B Colab best-of-N result is affected by a build-specific patch-application failure. Cross-device 12B results are confounded by different model builds and execution environments. Complete raw campaign traces are retained in separate archives and are not part of this Git snapshot; the public repository contains the preregistrations, deviations, run-level analysis output, serial re-check artifacts and analysis code.

## Rebuild

From the repository root, with Python, Matplotlib, NumPy and Tectonic available. The PDFs here were built with Python 3.12, Matplotlib 3.11.1, NumPy 1.26.4 and Tectonic 0.15.0:

```bash
bash scripts/paper_ii/build.sh
```

The command regenerates the figures from `docs/colab_scale/results/exploratory_tests_2026-11-19.json`, then compiles the English and Ukrainian articles. It makes no model calls and starts no benchmark. Tectonic intermediates are written under `tmp/pdfs/paper-ii/`.

The analytical source snapshot used by this draft is commit `57a55d3d23d6c7c3dd7be3abfc1a0d9c758fc6b5` on the public `main` branch. The experiment protocols and all departures from them are linked in the article and the result reports.

Suggested citation: Pylypenko, O. (2026). *When Fitness Scores Mislead: An Empirical Study of LLM-Guided Program Evolution*. Preprint draft, 19 November 2026. No DOI or peer-reviewed venue is claimed.

## License and transparency

Article text, figures and research summaries are licensed under CC BY 4.0. The figure-generation script is Apache-2.0. The author reports PhD-student status at the State University of Trade and Economics and cooperation with its Department of Software Engineering and Cybersecurity. Experiments were run using the author's equipment and accounts. AI assistants supported analysis code, drafting and translation; the author remains responsible for the final manuscript and its claims.
