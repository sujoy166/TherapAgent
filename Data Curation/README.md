# Datasets and SLURM run book

This document describes every cohort the TherapAgent pipeline accepts, the
TCGA source it was curated from, the resulting sample counts and per-head
label distributions, and the **exact SLURM submission commands** to train
every model on it.

> If you only want to launch the full sweep and pick up the PDF at the end,
> jump straight to **[§Full sweep on Nova](#full-sweep-on-nova)**.

---

## 1. What lives where

```
TherapAgent/
├── Data Curation/                            ← this directory; per-cohort
│   ├── README.md                                ssGSEA generator scripts
│   ├── gene_to_pathway_Breast_Cancer.py
│   ├── gene_to_pathway_Lung_Cancer.py
│   ├── gene_to_pathway_prostate_cancer.py
│   ├── gene_to_pathway_head_neck_cancer.py
│   └── gene_to_pathway_thyroid_cancer.py
├── Intermediate Dataset/<Cohort>.csv         ← 1,706 pathways × N samples
├── Final DataSet/<Cohort>_final.csv          ← N labeled samples ×
│                                                (sample, stage, 1706 paths)
├── binn/        graphpath/        path/      ← 3 models, each with main.py,
│   ├── config.py     (COHORT_FILES registry shared by all three)
│   ├── main.py       (env / reactome / data / train / evaluate phases)
│   ├── checker.py    (paper-conformance check vs the reference paper)
│   └── artifacts/<cohort>/                   ← per-cohort outputs
├── paper/                                    ← ACM-/IEEE-formatted manuscript
└── slurm/                                    ← submission scripts (see §3)
```

The single point of truth for which cohorts are enabled is
`binn/config.py:COHORT_FILES` — graphpath/ and path/ import that same
registry, so adding a new cohort is one edit.

---

## 2. The five enabled cohorts

All five are curated through an identical ssGSEA-against-Reactome workflow:

1. Download TCGA HiSeqV2 expression matrix from UCSC Xena.
2. ssGSEA against Reactome gene sets of size 10–1000, normalised to [0, 1]
   row-wise, written to `Intermediate Dataset/<Cohort>.csv` as
   *(1,706 pathways × N samples)*.
3. Download the matching clinical matrix.
4. Bit-encode three clinical fields into a single `stage` integer:

   ```
   stage = 4·𝟙[TMT given]  +  2·𝟙[RT given]  +  𝟙[OS ≥ 180 days]
   ```

   …and inner-join with the score matrix to produce
   `Final DataSet/<Cohort>_final.csv` as *(N labeled samples ×
   sample + stage + 1,706 pathway columns)*.

### Headline counts

| Cohort        | TCGA project              | Intermediate `n` | Labeled `n` | Stage codes present | TMT+ | RT+ | OS+ |
|---------------|---------------------------|-----------------:|------------:|--------------------|-----:|----:|----:|
| `breast`      | TCGA-BRCA                 | 1,218            |     **618** | 0–7 (all 8)        | 92 % | 56 % | 75 % |
| `lung`        | TCGA-LUNG (LUAD+LUSC)     | 1,129            |     **970** | 0–7 (all 8)        | 32 % | 13 % | 89 % |
| `prostate`    | TCGA-PRAD                 |   550            |     **496** | {0,1,3,4,5,7}      | 11 % | 12 % | 97 % |
| `head_neck`   | TCGA-HNSC                 |   566            |     **429** | {0,1,2,3,5,6,7}    | 35 % | 64 % | 92 % |
| `thyroid`     | TCGA-THCA                 |   572            |     **109** | {1,2,3,7}          |  6 % | 56 % | 99 % |

Bladder (TCGA-BLCA, n=21) was present in an earlier release but has been
dropped: the stratified 70/15/15 split is unsafe with so few patients.

### Per-cohort cards

#### `breast` — TCGA-BRCA

- **Source.** `https://tcga-xena-hub.s3.us-east-1.amazonaws.com/download/TCGA.BRCA.sampleMap%2FHiSeqV2.gz`
  + `TCGA.BRCA.sampleMap%2FBRCA_clinicalMatrix`
- **Labeled cohort.** 618 patients with `targeted_molecular_therapy`,
  `radiation_therapy`, and `OS_Time_nature2012` all present.
- **Notes.** All 8 stage codes are present; this is the most balanced
  multi-label setting in the registry. **This is the manuscript's
  headline cohort** — the trained-model numbers in §V of the paper
  come from here.
- **Regenerate from scratch (≈3 min, needs internet):**
  ```bash
  python3 "Data Curation/gene_to_pathway_Breast_Cancer.py"
  mv tcga_pathway_scores_ssgsea_reactome.csv "Intermediate Dataset/Breast_cancer.csv"
  mv tcga_stage_reactome_scores.csv          "Final DataSet/Breast_Cancer_final.csv"
  ```
- **Train all three models on breast (local, no SLURM):**
  ```bash
  python3 -m binn.main      all --cohort breast
  python3 -m graphpath.main all --cohort breast
  python3 -m path.main      all --cohort breast
  ```
- **Train only one model:** replace the model name above. Pass `--smoke`
  to cap epochs at 5 for a sanity run.

#### `lung` — TCGA-LUNG (LUAD + LUSC)

- **Source.** `…/TCGA.LUNG.sampleMap%2FHiSeqV2.gz` + `LUNG_clinicalMatrix`
- **Labeled cohort.** 970 patients — the **largest** in the registry.
- **Notes.** Re-generated 2026-05-14 after the original
  `Lung_cancer_final.csv` was identified as a stale copy of breast
  (same TCGA-A2-… barcodes). The current file has TCGA-05-… and other
  lung barcodes, zero overlap with breast. Stage codes 0–7 all present.
- **Regenerate (≈4 min):**
  ```bash
  python3 "Data Curation/gene_to_pathway_Lung_Cancer.py"
  mv tcga_pathway_scores_ssgsea_reactome.csv "Intermediate Dataset/Lung_cancer.csv"
  mv tcga_stage_reactome_scores.csv          "Final DataSet/Lung_cancer_final.csv"
  ```
- **Train (local):**
  ```bash
  python3 -m binn.main all --cohort lung
  python3 -m graphpath.main all --cohort lung
  python3 -m path.main all --cohort lung
  ```

#### `prostate` — TCGA-PRAD

- **Source.** `…/TCGA.PRAD.sampleMap%2FHiSeqV2.gz` + `PRAD_clinicalMatrix`
- **Labeled cohort.** 496 patients.
- **Notes.** Heavily skewed toward `OS = 1` (97 % of patients survive
  the 180-day threshold) and `TMT = 0` / `RT = 0` (89 % and 88 %
  negative respectively). Stage codes 2 and 6 absent.
- **Train (local):**
  ```bash
  for m in binn graphpath path; do
    python3 -m "$m".main all --cohort prostate
  done
  ```

#### `head_neck` — TCGA-HNSC

- **Source.** `…/TCGA.HNSC.sampleMap%2FHiSeqV2.gz` + `HNSC_clinicalMatrix`
- **Labeled cohort.** 429 patients.
- **Notes.** RT-leaning (64 % positive) and OS-high (92 %). Stage 4
  absent.
- **Train (local):**
  ```bash
  for m in binn graphpath path; do
    python3 -m "$m".main all --cohort head_neck
  done
  ```

#### `thyroid` — TCGA-THCA

- **Source.** `…/TCGA.THCA.sampleMap%2FHiSeqV2.gz` + `THCA_clinicalMatrix`
- **Labeled cohort.** 109 patients — smallest enabled cohort.
- **Notes.** Only stage codes {1, 2, 3, 7} are present; the
  `stratified_split` routine in `binn/data.py` falls back to a
  non-stratified partition. OS saturates at 99 %; TMT is only 6 %
  positive.
- **Train (local):**
  ```bash
  for m in binn graphpath path; do
    python3 -m "$m".main all --cohort thyroid
  done
  ```

---

## 3. SLURM run book {#full-sweep-on-nova}

The whole sweep runs on **one A100 sequentially** (cluster gives at most
one GPU). Two-job dependency chain:

```
  00_install.sbatch   →   10_sweep.sbatch
  (venv build)            (5 cohorts × 3 models + paper build)
```

**Canonical repo path on Nova:** `/work/mech-ai-scratch/tirtho/TherapAgent`
— hard-coded as the default `THERAP_REPO` in all three slurm scripts and
in `submit_all.sh`. Submit from anywhere; the scripts `cd` into that
path themselves.

`#SBATCH --mail-user=tirtho@iastate.edu` is baked into both sbatch
files. Sweep wall-clock ≈ 50 min on an A100; allocation
`--time=08:00:00`.

### Full sweep (recommended)

```bash
# From anywhere on Nova:
bash /work/mech-ai-scratch/tirtho/TherapAgent/slurm/submit_all.sh
```

That's it. Walks all five cohorts × three models, regenerates the
LaTeX tables, renders all 9 figures, and rebuilds `paper/main.pdf`.

If you want SLURM mail to go somewhere else:

```bash
THERAP_SLURM_EMAIL=other@example.com bash slurm/submit_all.sh
```

### Skip the install (venv already exists)

```bash
NO_INSTALL=1 bash slurm/submit_all.sh
```

### Resume after a timeout / partial sweep

`THERAP_RESUME=1` makes the sweep skip any `(model, cohort)` whose
`results.json` already exists:

```bash
THERAP_RESUME=1 bash slurm/submit_all.sh
```

### Train a single cohort with SLURM

Submit `10_sweep.sbatch` directly with the cohort list overridden:

```bash
# Trick: limit the sweep to one cohort by setting THERAP_RESUME=1
# on a half-done run, or by hand-editing slurm/10_sweep.sbatch's
# COHORTS=(...) array. Easiest no-edit form:
sbatch --export=ALL,COHORTS=lung \
       --mail-user=tirtho@iastate.edu \
       slurm/10_sweep.sbatch
```

(Edit the `COHORTS=(...)` array inside `10_sweep.sbatch` if you want
something more durable.)

### Manual sbatch (without the wrapper)

```bash
cd /work/mech-ai-scratch/tirtho/TherapAgent

# 1. One-time install
sbatch slurm/00_install.sbatch
# → returns <JID_INSTALL>

# 2. Sweep (depends on the install job's success)
sbatch --dependency=afterok:<JID_INSTALL> slurm/10_sweep.sbatch
```

Both files already declare `--mail-user=tirtho@iastate.edu` and the
canonical `$THERAP_REPO`, so the bare `sbatch` invocations above are
sufficient.

### Watch progress

```bash
squeue --me
tail -f slurm/logs/therap_sweep-<JID>.out
```

### Re-build paper only (no retraining)

```bash
source .venv-therap/bin/activate
for c in breast lung prostate head_neck thyroid; do
    python3 paper/scripts/regen_tex.py --cohort "$c"
done
( cd paper/figures && ./make_all.sh )
( cd paper && tectonic main.tex )
```

---

## 4. What artifacts you get

After a successful sweep:

```
<model>/artifacts/<cohort>/
├── reactome.pkl       Reactome layer/adjacency spec (Phase 2)
├── splits.npz         standardised X, Y, train/val/test indices, scaler, pos_weight  (Phase 3)
├── <model>.pt         trained checkpoint + loss history                              (Phase 4)
├── results.json       per-head AUROC / AUPRC / F1 / accuracy / CM / threshold        (Phase 5)
├── importance.npz     per-head gradient×input pathway importance                     (Phase 5)
└── tex/01..05_*.tex   booktabs LaTeX tables per phase

paper/
├── figures/fig*.pdf       9 colorblind-safe figures (Okabe-Ito + viridis)
├── artifacts/tex/         06_top_pathways_<cohort>.tex
└── main.pdf               8-page manuscript ready to submit
```

The 15 `(model, cohort)` combinations don't clobber each other because
each writes to a per-cohort subdirectory.

---

## 5. Environment overrides cheat sheet

| Variable | Default | Purpose |
|----------|---------|---------|
| `THERAP_REPO`           | `/work/mech-ai-scratch/tirtho/TherapAgent` | Absolute path to the cloned repo on Nova. |
| `THERAP_VENV`           | `$THERAP_REPO/.venv-therap`     | Python venv path. |
| `THERAP_PYTHON_MODULE`  | `python/3.11`                   | First module name tried; the loader also walks a fallback list (`python/3.11.7`, `python/3.11.5`, `python-3.11`, `Python/3.11`, `py-python/3.11`, `python3/3.11`). |
| `THERAP_CUDA_MODULE`    | `cuda/12.1`                     | Optional. Set `THERAP_LOAD_CUDA=0` to skip. |
| `THERAP_TECTONIC`       | `tectonic`                      | LaTeX engine binary. |
| `THERAP_SLURM_EMAIL`    | `tirtho@iastate.edu`            | `--mail-user` target. |
| `THERAP_RESUME`         | `0`                             | If `1`, the sweep skips cells that already have a `results.json`. |
| `NO_INSTALL`            | `0`                             | If `1`, `submit_all.sh` skips the venv job. |

---

## 6. Pre-flight checks (run locally before submitting)

```bash
# 1. All five cohorts resolve to existing CSVs:
python3 -c "
from binn.config import COHORT_FILES
from pathlib import Path
for k, (i, f) in COHORT_FILES.items():
    p1 = Path(f'Intermediate Dataset/{i}.csv'); p2 = Path(f'Final DataSet/{f}.csv')
    assert p1.exists() and p2.exists(), f'{k}: missing'
print('all 5 cohorts resolve')
"

# 2. The paper-conformance check for every model passes:
for m in binn graphpath path; do
    python3 -m "$m".checker || exit 1
done

# 3. The offline smoke tests pass (no internet needed):
for m in binn graphpath path; do
    python3 -m "$m".tests.test_smoke
done
```

If all three checks are green, the SLURM submission will succeed for
every (model, cohort) cell.
