# TherapAgent — Pathway-driven therapy response prediction for TCGA-BRCA

This repository contains a TCGA-BRCA pathway analysis pipeline plus **three
biologically informed deep-learning models** that all predict the same three
therapy-response phenotypes from the same pathway ssGSEA scores.

| Model       | Paper                                                                                                      | Subdir         |
|-------------|------------------------------------------------------------------------------------------------------------|----------------|
| **BINN**    | Hartman *et al.* 2023, *Nat. Commun.* 14:5359 — sparse Reactome-hierarchical NN                            | `binn/`        |
| **GraphPath** | Ma & Wang 2024, *Bioinformatics* 40:btae165 — multi-head GAT over a pathway-pathway interaction graph    | `graphpath/`   |
| **PATH**    | Howlader, Islam, Le 2026, arXiv — edge-aware Graph Transformer with Laplacian PE over Jaccard adjacency    | `path/`        |

`gene_to_pathway.py` is the upstream data-generation script (TCGA HiSeqV2 →
ssGSEA → `pathway_scores.csv` and `pathway_phenotype_mapping.csv`).

---

## Data files

| File | Role | Shape |
|------|------|------:|
| `pathway_scores.csv`            | **Feature matrix** — Reactome pathway ssGSEA scores per TCGA sample, normalised to [0,1] | 1,706 pathways × 1,218 samples |
| `pathway_phenotype_mapping.csv` | **Label table** — `stage` column per sample (pathway-score columns are redundant copies; the models ignore them) | 618 samples × (1 stage + 1,706 pathway columns) |

All three models train on the **618-sample intersection**: features come from
`pathway_scores.csv` (transposed to samples × pathways), labels come from the
`stage` column of `pathway_phenotype_mapping.csv`.

### How `stage` decomposes into three heads

The 0..7 stage integer is a bitfield. Every model predicts these three binary
heads independently (any combination may be present or absent):

| Bit | Value | Phenotype                            |
|-----|------:|--------------------------------------|
| 2   |     4 | **TMT** — targeted molecular therapy |
| 1   |     2 | **RT**  — radiation therapy          |
| 0   |     1 | **OS**  — overall survival ≥ 180 days|

---

## TL;DR — run a model end-to-end

Each subdirectory ships an identical `scripts/run_all.sh`:

```bash
./binn/scripts/run_all.sh        # Reactome-hierarchical BINN
./graphpath/scripts/run_all.sh   # Multi-head GAT
./path/scripts/run_all.sh        # Edge-aware Graph Transformer
```

That chains the five phases below. Each phase has its own `.sh` if you want
to step through, and every phase writes (a) a binary checkpoint for the next
phase plus (b) a **LaTeX (booktabs) table** you can `\input{}` straight into
a paper.

---

## Phase pipeline (identical structure across all three models)

For every model `<m>` in `{binn, graphpath, path}`:

### Phase 1 — Environment setup

```bash
./<m>/scripts/01_setup.sh
```

Installs `numpy`, `pandas`, `scipy`, `scikit-learn`, `torch`,
`requests` (plus `joblib` for `binn`) and records the resolved versions.

| Output | Contents |
|--------|----------|
| `<m>/artifacts/tex/01_environment.tex` | LaTeX table of Python + package versions. |

### Phase 2 — Reactome ingest

```bash
./<m>/scripts/02_reactome.sh
```

Each model derives a *different* biological structure from Reactome:

| Model       | What's built                                                  | Source files                                                   |
|-------------|---------------------------------------------------------------|-----------------------------------------------------------------|
| `binn`      | Layered parent-child hierarchy (≥4 levels), one mask per layer | `ReactomePathways.txt` + `ReactomePathwaysRelation.txt`         |
| `graphpath` | Symmetric `{0,1}` adjacency = parent/child ∪ siblings          | `ReactomePathways.txt` + `ReactomePathwaysRelation.txt`         |
| `path`      | Weighted Jaccard adjacency from pathway gene memberships       | `ReactomePathways.gmt.zip` (matches PATH paper Eq. 2)           |

| Output | Contents |
|--------|----------|
| `<m>/cache/Reactome*.{txt,gmt.zip}`     | Raw Reactome downloads (cached).                                  |
| `<m>/artifacts/reactome.pkl`            | Model-specific Reactome representation for downstream phases.     |
| `<m>/artifacts/tex/02_*.tex`            | LaTeX summary of layer sizes / adjacency density / etc.           |

### Phase 3 — Data preparation

```bash
./<m>/scripts/03_data.sh
```

Identical across models: loads pathway scores, restricts to Reactome-matched
columns, decodes `stage` into 3 binary heads, builds a stratified
**80/10/10** (graphpath, path) or **70/15/15** (binn) train/val/test split,
fits a standardizer on the train fold only, computes per-head
`pos_weight = #neg/#pos` (clipped to `[0.1, 20]`).

| Output | Contents |
|--------|----------|
| `<m>/artifacts/splits.npz`                       | Standardized `X`, labels `Y`, split indices, scaler, `pos_weight`. |
| `<m>/artifacts/tex/03_data_alignment.tex`        | Source CSV cardinalities (scores ∩ mapping).                       |
| `<m>/artifacts/tex/03_data_splits.tex`           | Train/Val/Test sizes + per-head positive prevalence.               |
| `<m>/artifacts/tex/03_head_distribution.tex`     | Per-head positives/negatives + resulting `pos_weight`.             |

### Phase 4 — Training

```bash
./<m>/scripts/04_train.sh                 # full run
./<m>/scripts/04_train.sh --smoke         # 5-epoch sanity
./<m>/scripts/04_train.sh --epochs 80     # custom cap
```

| Model       | Architecture                                                                                                    | Optimizer       | Default LR  |
|-------------|------------------------------------------------------------------------------------------------------------------|-----------------|------------:|
| `binn`      | Sparse `MaskedLinear` per layer → tanh + BN + dropout, **per-layer auxiliary head**, final P = mean(σ(layer headᵢ)) | Adam + L2 1e-3  | 1e-3        |
| `graphpath` | Per-pathway projection → **multi-head GAT (K=3, ELU)** over Reactome adjacency → tanh readout → FC                | SGD + momentum  | 5e-2        |
| `path`      | Per-pathway projection + **Laplacian PE** → L=2 **edge-aware Graph Transformer blocks (H=4)** + soft mask → attention readout → MLP head | AdamW + grad-clip 2.0 | 1e-4 |

All three use class-weighted BCE on the 3 multi-label heads, plateau LR
scheduler, and early stopping on validation loss.

| Output | Contents |
|--------|----------|
| `<m>/artifacts/<m>.pt`                          | Full state dict + scaler + Reactome structure + loss history.   |
| `<m>/artifacts/tex/04_training_summary.tex`     | Hyperparams + final losses + epochs run + parameter count.      |

### Phase 5 — Evaluation

```bash
./<m>/scripts/05_evaluate.sh
```

Per-head metrics on val + test: AUROC, AUPRC (threshold-free), F1 + accuracy
+ TN/FP/FN/TP at the 0.5 threshold.

| Output | Contents |
|--------|----------|
| `<m>/artifacts/results.json`              | Per-head metrics + run metadata.                                 |
| `<m>/artifacts/tex/05_metrics.tex`        | LaTeX: one row per (head × split) with all six numbers + CM.     |

### Phase 6 (optional) — Offline smoke test

```bash
./<m>/scripts/06_smoke.sh
```

No internet required — uses a deterministic synthetic Reactome substitute,
trains for a few epochs, and asserts shapes / finite losses / metrics in
[0, 1] / well-formed booktabs LaTeX output. Ideal for first-time setup on a
new machine.

---

## Cross-model comparison (after running all three)

Because every model emits a `05_metrics.tex` with identical column structure
(`Head | Split | AUROC | AUPRC | F1 | Acc | TN | FP | FN | TP`), you can
build a side-by-side table by `\input{}`-ing all three into a single LaTeX
section, or `cat`-ing the three together at the shell:

```bash
cat binn/artifacts/tex/05_metrics.tex \
    graphpath/artifacts/tex/05_metrics.tex \
    path/artifacts/tex/05_metrics.tex
```

All training/data tables (`03_*`, `04_*`) follow the same per-model
convention.

---

## File layout

```
TherapAgent/
├── README.md                           ← this file (single source of truth)
├── gene_to_pathway.py                  TCGA-BRCA → ssGSEA → CSV generator
├── pathway_scores.csv                  features  (1,706 pathways × 1,218 samples)
├── pathway_phenotype_mapping.csv       labels    (618 samples × stage)
├── RefeerencePaper/                    BINN.pdf, GraphPath.pdf, PATH.pdf
│
├── binn/                               ← Hartman et al. 2023 (Reactome-hierarchical BINN)
│   ├── config.py · reactome.py · data.py · model.py · train.py · evaluate.py · main.py · reporting.py
│   ├── scripts/{01_setup,02_reactome,03_data,04_train,05_evaluate,06_smoke,run_all}.sh
│   ├── tests/test_smoke.py
│   ├── cache/                          Reactome downloads (Phase 2)
│   └── artifacts/                      reactome.pkl, splits.npz, binn.pt, results.json, tex/
│
├── graphpath/                          ← Ma & Wang 2024 (multi-head GAT)
│   ├── config.py · reactome.py · model.py · train.py · main.py
│   ├── scripts/{01..06,run_all}.sh
│   ├── tests/test_smoke.py
│   ├── cache/
│   └── artifacts/                      reactome.pkl, splits.npz, graphpath.pt, results.json, tex/
│
└── path/                               ← Howlader et al. 2026 (edge-aware Graph Transformer)
    ├── config.py · reactome.py · model.py · train.py · main.py
    ├── scripts/{01..06,run_all}.sh
    ├── tests/test_smoke.py
    ├── cache/
    └── artifacts/                      reactome.pkl, splits.npz, path.pt, results.json, tex/
```

`graphpath/` and `path/` import the model-agnostic helpers (`data.py`,
`evaluate.py`, `reporting.py`) from `binn/` — so all three models share an
identical data flow and metric definition, while differing only in
Reactome-derived structure and the model architecture proper.

---

## Including the tables in a LaTeX paper

Every table uses `booktabs`. Preamble:

```latex
\usepackage{booktabs}
```

Then include any of them — each table already carries its own
`\caption{…}` and `\label{tab:…}`:

```latex
% BINN
\input{binn/artifacts/tex/02_reactome_layers.tex}
\input{binn/artifacts/tex/03_data_alignment.tex}
\input{binn/artifacts/tex/03_data_splits.tex}
\input{binn/artifacts/tex/03_head_distribution.tex}
\input{binn/artifacts/tex/04_training_summary.tex}
\input{binn/artifacts/tex/05_metrics.tex}

% GraphPath
\input{graphpath/artifacts/tex/02_pathway_graph.tex}
\input{graphpath/artifacts/tex/04_training_summary.tex}
\input{graphpath/artifacts/tex/05_metrics.tex}

% PATH
\input{path/artifacts/tex/02_pathway_graph.tex}
\input{path/artifacts/tex/04_training_summary.tex}
\input{path/artifacts/tex/05_metrics.tex}
```

Reference any of them in prose with `\ref{tab:binn-metrics}`,
`\ref{tab:gp-metrics}`, `\ref{tab:path-metrics}`, etc. Label prefixes:

| Subdir       | Prefix    |
|--------------|-----------|
| `binn/`      | `binn-`   |
| `graphpath/` | `gp-`     |
| `path/`      | `path-`   |

---

## Design notes

1. **Two CSVs serve distinct roles.** `pathway_scores.csv` is the canonical
   feature matrix; `pathway_phenotype_mapping.csv` only contributes the
   `stage` label. All models ignore the pathway-score copies inside the
   mapping CSV to avoid any drift between "features for labeled samples"
   and "features for all samples."
2. **All three models share the data and metrics, differ in architecture.**
   `data.py`, `evaluate.py`, and `reporting.py` live in `binn/` and are
   imported by `graphpath/` and `path/`. The three subdirs differ in
   `reactome.py` (different Reactome representation), `model.py`
   (architecture), `train.py` (optimizer), and `config.py` (hyperparams).
3. **Pathway-level adaptation.** GraphPath and PATH originally consume
   *gene-level* CNV + mutation profiles. Our dataset only provides
   *pathway-level* ssGSEA scores, so for both we replace the gene-level
   stages with a learnable per-pathway projection (1-D score → embedding).
   Everything downstream (GAT for GraphPath, edge-aware Transformer for PATH)
   is faithful to its paper. This adaptation is documented in each model's
   `model.py` docstring.
4. **Multi-label heads.** TMT, RT, and OS≥180d are independent clinical
   events; three sigmoid heads let a sample be positive for any
   combination, and per-head `pos_weight` counters the strong imbalance
   (TMT is ≈12:1 positive-skewed in this cohort).
5. **Phased pipeline.** Every step writes a checkpoint *and* a LaTeX
   table, so partial reruns are cheap and every published number traces
   back to a versioned artifact.
