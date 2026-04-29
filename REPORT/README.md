# Report & Slides

Final written deliverables for the NLP project. The compiled PDF is committed; LaTeX sources and figures are kept alongside it so the report can be regenerated end-to-end from the artefacts in [`../data/`](../data).

## Contents

| File | Description |
| ---- | ----------- |
| [`report.tex`](report.tex) | LaTeX source of the final write-up |
| [`report.pdf`](report.pdf) | Compiled final report |
| [`slide.tex`](slide.tex) | LaTeX Beamer source for the project presentation deck |
| [`make_plots.py`](make_plots.py) | Regenerates every figure in `figures/` from the JSON reports in `../data/` |
| [`figures/`](figures/) | All figures referenced by `report.tex` and `slide.tex` |

## Figures

All figures are produced by `make_plots.py` from the same JSON artefacts the Streamlit dashboard reads — the report and dashboard are guaranteed to agree.

| File | Stage | Source artefact |
| ---- | ----- | --------------- |
| `fig_monthly_corpus.png` | 1.1 | `data/reddit_technology_recent.db` |
| `fig_top_domains.png` | 1.1 | `data/topic_report.json` |
| `fig_topic_shares.png` | 1.2 | `data/topic_report.json` |
| `fig_temporal_labels.png` | 1.3 | `data/temporal_report.json` |
| `fig_temporal_trajectories.png` | 1.3 | `data/temporal_report.json` |
| `fig_stance_distribution.png` | 1.4 | `data/stance_report.json` |
| `fig_stance_overlap.png` | 1.4 | `data/stance_report.json` |
| `fig_stance_generic_targeted_distribution.png` | 1.4 | `data/stance_report*.json` |
| `fig_stance_generic_targeted_scatter.png` | 1.4 | `data/stance_report*.json` |
| `fig_stance_targeted_shift.png` | 1.4 | `data/stance_report_targeted.json` |
| `fig_rag_metrics.png` | 2.1 | `data/rag_report_local.json` |
| `fig_rag_faithfulness.png` | 2.1 | `data/rag_report_local.json` |
| `fig_rag_by_type.png` | 2.1 | `data/rag_report_local.json` |
| `fig_hindi_metrics.png` | 2.2 | `data/hindi_translation_report.json` |
| `fig_hindi_tags.png` | 2.2 | `data/hindi_translation_report.json` |

## Building

### Regenerate figures

```bash
micromamba run -n nlp_final_gpu python REPORT/make_plots.py
```

Reads the JSON artefacts in `../data/`, writes PNGs into `figures/`. Safe to re-run.

### Compile the report

```bash
cd REPORT
pdflatex report.tex
pdflatex report.tex   # second pass for cross-references
```

### Compile the slides

```bash
cd REPORT
pdflatex slide.tex
```

Both LaTeX sources use only `figures/` as the image search path, so they compile cleanly as long as `make_plots.py` has been run.

## Notes

- Auxiliary LaTeX outputs (`*.aux`, `*.log`, `*.toc`, `*.out`) are gitignored — only the PDF, the source `.tex` files, and the `figures/` directory ship with the repo.
- The PDF in this folder reflects the final state of the JSON artefacts in `../data/`. Regenerating any of those (e.g. rerunning the stance pipeline) should be followed by `make_plots.py` and a `pdflatex` pass.
