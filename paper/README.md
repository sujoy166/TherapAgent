# `paper/` — ASI 2026 manuscript

Track II submission for the 4th Workshop on Advances in Systems Immunology
(ASI 2026, ACM-BCB Workshop Companion Proceedings). 8-10 page limit, ACM
Master Article Template (`acmart`, `sigconf` style).

## Files

| File / Directory                | Purpose                                                                 |
|---------------------------------|--------------------------------------------------------------------------|
| `main.tex`                      | The manuscript itself. Uses `\inputiffile{…}` to pull per-phase tables.  |
| `references.bib`                | Bibliography (Hartman, Ma-Wang, Howlader, P-NET, Pathformer, IRnet, …).  |
| `figures/`                      | Colorblind-safe matplotlib generators + rendered PDFs.                   |
| `figures/_style.py`             | Okabe-Ito palette + viridis defaults (Nature Methods-recommended).       |
| `figures/fig1_architectures.py` | Side-by-side architecture diagram (no training data needed).             |
| `figures/fig2_metric_bars.py`   | Test-set AUROC/AUPRC/F1 across BINN, GraphPath, PATH.                    |
| `figures/fig3_confusion.py`     | 3×3 confusion-matrix grid (viridis).                                     |
| `figures/fig4_label_imbalance.py` | TMT/RT/OS positive prevalence + per-head BCE pos_weight.               |
| `figures/make_all.sh`           | Renders all four figures.                                                |
| `tables/`                       | (Empty.) Tables live in each model's `artifacts/tex/`; the paper pulls them in via `\inputiffile{…}`. |
| `literature/SURVEY.md`          | Auto-generated literature survey notes (bioRxiv / Nature / arXiv).       |
| `build.sh`                      | One-shot: render figures + run `latexmk` to build `main.pdf`.            |

## Quickstart

1. **Run all three pipelines** so the per-phase LaTeX tables exist:
   ```bash
   ./binn/scripts/run_all.sh
   ./graphpath/scripts/run_all.sh
   ./path/scripts/run_all.sh
   ```
2. **Build the manuscript:**
   ```bash
   ./paper/build.sh
   ```
   Open `paper/main.pdf`.

The build is resilient to missing tables — if a particular phase has not
been run yet, the paper compiles anyway with a clearly-marked placeholder
box where that table will go. So you can compile the manuscript at any
point during development.

## Colour accessibility

Every figure uses one of:

- **Okabe-Ito** (categorical, 8-colour, CVD-tested) for distinct categories
  such as model identity (BINN = blue, GraphPath = orange, PATH = bluish
  green) and class membership (positive = vermillion, negative = sky blue).
- **viridis** (perceptually-uniform sequential) for continuous heatmap data
  such as confusion-matrix counts.

Both palettes are explicitly recommended by Nature Methods for figures that
must remain legible under deuteranopia / protanopia / tritanopia (see
references in `references.bib`).

## Conference target

| Field                | Value                                                                |
|----------------------|----------------------------------------------------------------------|
| Venue                | ASI 2026 — 4th Workshop on Advances in Systems Immunology            |
| Track                | Track II (full paper, 8-10 pages)                                    |
| Proceedings          | ACM-BCB 2026 Workshop Companion Proceedings                          |
| Template             | ACM Master Article Template (`acmart`), `sigconf` style              |
| Submission deadline  | 18 May 2026                                                          |
| Final version        | 30 May 2026                                                          |
| Submission system    | OpenReview (free account required)                                   |
| Workshop date        | 30 June 2026, University of Calabria, Rende (Cosenza), Italy         |

## When to switch to `\documentclass[sigconf]{acmart}` (no `nonacm`, no `review`)

The header is currently:

```latex
\documentclass[sigconf,nonacm,review]{acmart}
```

`review` shows line numbers; `nonacm` hides ACM copyright boilerplate.
For the **camera-ready** version, switch to:

```latex
\documentclass[sigconf]{acmart}
```

and fill in the proper `\acmConference{…}{…}{…}` and `\copyrightyear`
fields once accepted.
