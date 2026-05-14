# TherapAgent

Pathway-informed deep learning for **multi-phenotype therapy-response
stratification** across **five TCGA solid-tumor cohorts** (breast, lung,
prostate, head & neck, thyroid).

Three architectures benchmarked on the identical data flow:

| Subdir       | Model     | Reference                              |
|--------------|-----------|----------------------------------------|
| `binn/`      | BINN      | Hartman *et al.* 2023 (Nat. Commun.)   |
| `graphpath/` | GraphPath | Ma & Wang 2024 (Bioinformatics)        |
| `path/`      | PATH      | Howlader *et al.* 2026 (arXiv)         |

Each model exposes the same five-phase CLI
(`env → reactome → data → train → evaluate`) accepting `--cohort
{breast,lung,prostate,head_neck,thyroid}`, and ships a
`<model>.checker` that aborts before training if the implementation
has drifted from the reference paper.

## Quick start

### Local (one cohort, one model)

```bash
python3 -m binn.main      all --cohort breast
python3 -m graphpath.main all --cohort lung
python3 -m path.main      all --cohort thyroid
```

### Full sweep on Nova (3 models × 5 cohorts on one GPU)

Canonical repo path is `/work/mech-ai-scratch/tirtho/TherapAgent`,
hard-coded as the default `THERAP_REPO`. Submit from anywhere:

```bash
bash /work/mech-ai-scratch/tirtho/TherapAgent/slurm/submit_all.sh
```

The sweep asks SLURM for `--gres=gpu:1` (any available GPU type — A100,
A40, V100, RTX, etc.). SLURM mail goes to `tirtho@iastate.edu` by
default (override with `THERAP_SLURM_EMAIL=other@example.com bash …`).
End-of-sweep wall-clock on an A100 is ≈ 50 min; budget more on slower
cards.

## Where to look next

| File / dir | What's in it |
|---|---|
| **[`Data Curation/README.md`](Data%20Curation/README.md)** | **Detailed per-cohort cards** (TCGA project, sample counts, label distributions, regen + train commands per cohort) and the **full SLURM run book** (submission flags, resume mode, manual sbatch, pre-flight checks). |
| [`slurm/README.md`](slurm/README.md) | Short SLURM cheat sheet (env overrides, wall-clock estimates, manual sbatch). |
| `paper/main.pdf` | 8-page IEEE-conference manuscript for **ASI 2026** (ACM-BCB Workshop Companion Proceedings). |
| `paper/main.tex` | Source. Cross-cohort figures auto-regenerate from each cohort's `results.json`. |
| `paper/scripts/regen_tex.py` | Regenerate every per-phase LaTeX table from saved checkpoints. |
| `binn/checker.py`, `graphpath/checker.py`, `path/checker.py` | Run any of them with `python3 -m <model>.checker` to verify the code still matches the reference paper. |

## What lands in the working tree after a successful sweep

```
<model>/artifacts/<cohort>/
├── reactome.pkl          ← Reactome layer/adjacency spec (Phase 2)
├── splits.npz            ← standardised X, Y, indices, scaler        (Phase 3)
├── <model>.pt            ← trained checkpoint + loss history          (Phase 4)
├── results.json          ← per-head AUROC/AUPRC/F1/accuracy/CM/threshold  (Phase 5)
├── importance.npz        ← per-head pathway importance (SHAP-analogue)
└── tex/01..05_*.tex      ← booktabs LaTeX tables per phase

paper/
├── figures/fig{1..9}_*.pdf   ← 9 colorblind-safe (Okabe-Ito + viridis) figures
├── artifacts/tex/06_top_pathways_<cohort>.tex
└── main.pdf                  ← 8-page manuscript
```
