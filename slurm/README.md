# SLURM scripts for the TherapAgent training sweep on Nova

This directory submits the full **3 models × 5 cohorts = 15 training
runs** plus the paper rebuild as a single SLURM dependency chain. Logs
land in `slurm/logs/`.

```
slurm/
├── submit_all.sh          one-shot driver (orchestrates the three sbatch
│                          jobs below with the right --dependency flags)
├── 00_install.sbatch      one-time: builds .venv-therap from
│                          requirements.txt
├── 10_train.sbatch        15-task #SBATCH --array=0-14 ; each task runs
│                          `<model>.main all --cohort <c>`
└── 20_paper.sbatch        regenerates LaTeX tables + figures + builds
                           paper/main.pdf via tectonic. Depends on the
                           training array.
```

## TL;DR

```bash
cd /path/to/TherapAgent
THERAP_SLURM_EMAIL=you@example.com bash slurm/submit_all.sh
```

That submits three jobs. SLURM will run them in order; you'll get
`BEGIN`/`END`/`FAIL` mail for each, and the final manuscript will be at
`paper/main.pdf` when the dependency chain completes.

## Environment overrides

All scripts honour the same env vars (with defaults shown):

| Variable | Default | Purpose |
|----------|---------|---------|
| `THERAP_REPO`           | `$PWD`                          | Absolute path to the cloned repository. |
| `THERAP_VENV`           | `$THERAP_REPO/.venv-therap`     | Python virtual-env path. |
| `THERAP_PYTHON_MODULE`  | `python/3.11`                   | `module load` name on Nova. |
| `THERAP_CUDA_MODULE`    | `cuda/12.1`                     | Optional CUDA module loaded by the train array. Set `THERAP_LOAD_CUDA=0` to skip. |
| `THERAP_TECTONIC`       | `tectonic`                      | LaTeX engine binary used by `20_paper.sbatch`. |
| `THERAP_SLURM_EMAIL`    | _(required)_                    | Address used by `--mail-user`. |

Override at the shell or per-sbatch on the command line:

```bash
THERAP_PYTHON_MODULE=python/3.12 \
THERAP_CUDA_MODULE=cuda/12.4 \
THERAP_SLURM_EMAIL=you@example.com \
  bash slurm/submit_all.sh
```

## Resource budgets

The array task allocation is sized for the heaviest single run
(PATH × lung):

| #SBATCH | Value |
|---------|-------|
| `--nodes`         | 1 |
| `--cpus-per-task` | 8 |
| `--mem`           | 64G |
| `--gres`          | `gpu:a100:1` |
| `--time`          | 4 h (PATH × lung finishes in ≈10 min; this is generous) |
| `--partition`     | `nova` |

BINN and GraphPath finish in a small fraction of that allocation but
share the same template for scheduling simplicity. To squeeze the
smaller models into a different QoS, override per-task at submit:

```bash
# Train only BINN cohorts with a tighter resource ask
sbatch --array=0-4 --time=00:30:00 --mem=16G --gres=gpu:a100:0 \
       --mail-user="$THERAP_SLURM_EMAIL" slurm/10_train.sbatch
```

## Job-array task index → (model, cohort)

```
       cohort       0=breast 1=lung 2=prostate 3=head_neck 4=thyroid
model
  0=BINN              0       1      2          3           4
  1=GraphPath         5       6      7          8           9
  2=PATH             10      11     12         13          14
```

Resume a single failed cell by passing its index:

```bash
# Re-run PATH × thyroid only:
sbatch --array=14 --mail-user="$THERAP_SLURM_EMAIL" slurm/10_train.sbatch
```

The training script always re-runs paper-conformance check
(`python -m <model>.checker`) as its first step, so the run aborts
before training if the implementation has drifted from the reference
paper.

## What each task writes

```
<model>/artifacts/<cohort>/
├── reactome.pkl           Phase 2 — Reactome layer/adjacency spec
├── splits.npz             Phase 3 — X, Y, train/val/test indices, scaler, pos_weight
├── <model>.pt             Phase 4 — model state + loss history
├── results.json           Phase 5 — per-head AUROC/AUPRC/F1/accuracy/CM
└── tex/
    ├── 01_environment.tex
    ├── 02_*.tex (Reactome graph)
    ├── 03_*.tex (data split + head distribution)
    ├── 04_training_summary.tex
    └── 05_metrics.tex
```

The paper build job picks these up automatically.

## Just the paper, after manual runs

If you trained from the login node and only need the PDF rebuilt:

```bash
sbatch --mail-user="$THERAP_SLURM_EMAIL" slurm/20_paper.sbatch
```

That script tolerates missing cohorts — it iterates over the five
enabled cohorts and only refreshes the tex tables for cohorts that
have a `results.json`.

## Local dry-run (no SLURM)

```bash
# Single (model, cohort), no SLURM scheduling:
python3 -m binn.main all --cohort thyroid

# All 15 combinations sequentially:
for c in breast lung prostate head_neck thyroid; do
  for m in binn graphpath path; do
    python3 -m "$m".main all --cohort "$c"
  done
done
python3 paper/scripts/regen_tex.py --cohort breast   # repeat per cohort
( cd paper && ./build.sh )
```
