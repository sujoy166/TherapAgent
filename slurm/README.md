# SLURM scripts for the TherapAgent training sweep on Nova

Two-job dependency chain that runs the full pipeline on a **single
allocated GPU**. The training loop is sequential — 5 cohorts × 3 models
= 15 runs that share one GPU — so the cluster only needs to schedule
one GPU at a time. The sweep asks for `--gres=gpu:1` (type-agnostic),
so it lands on whichever card is available (A100, A40, V100, RTX, …).

```
slurm/
├── submit_all.sh         one-shot driver (00_install + 10_sweep with
│                         --dependency=afterok)
├── 00_install.sbatch     one-time venv build from requirements.txt
└── 10_sweep.sbatch       single-GPU sequential sweep:
                              for cohort in 5 cohorts:
                                for model in (binn, graphpath, path):
                                    paper-conformance check
                                    full phase pipeline (env→reactome→
                                                         data→train→
                                                         evaluate)
                                regen LaTeX tables for cohort
                            render figures, compile paper PDF
```

## TL;DR

```bash
bash /work/mech-ai-scratch/tirtho/TherapAgent/slurm/submit_all.sh
```

That submits two jobs; SLURM runs them in order; the final manuscript
ends up at `paper/main.pdf`.

## Wall-clock budget

Sequential sweep on a single A100 (no oversubscription; other cards are
proportionally slower — A40 ≈ 1.3×, V100 ≈ 1.5×, consumer RTX ≈ 1.5–2×):

| Model    | per-cohort time | × 5 cohorts |
|----------|-----------------|-------------|
| BINN     | ≈ 30 s          | ≈ 3 min |
| GraphPath| ≈ 2 min         | ≈ 10 min |
| PATH     | ≈ 7 min         | ≈ 35 min |
| **Total** |                | **≈ 50 min** |

Plus the paper build (≈ 1 min) and the per-cohort Reactome download
(cached, ≈ 5 s for first cohort, 0 thereafter). The `--time=08:00:00`
allocation is generous; tighten it if your QoS prefers shorter walls.

## Environment overrides

| Variable | Default | Purpose |
|----------|---------|---------|
| `THERAP_REPO`           | `/work/mech-ai-scratch/tirtho/TherapAgent` | Canonical repo path on Nova; hard-coded in all three slurm scripts. |
| `THERAP_VENV`           | `$THERAP_REPO/.venv-therap`     | Python virtual-env path. |
| `THERAP_PYTHON_MODULE`  | `python/3.11`                   | `module load` name on Nova. |
| `THERAP_CUDA_MODULE`    | `cuda/12.1`                     | Optional CUDA module. Set `THERAP_LOAD_CUDA=0` to skip. |
| `THERAP_TECTONIC`       | `tectonic`                      | LaTeX engine binary. |
| `THERAP_SLURM_EMAIL`    | _(required)_                    | `--mail-user` target. |
| `THERAP_RESUME`         | `0`                             | If `1`, skip any (model, cohort) for which `results.json` already exists. Useful to continue after a timeout. |
| `NO_INSTALL`            | `0`                             | If `1`, skip the install job. |

## What lands where after a successful sweep

```
<model>/artifacts/<cohort>/
├── reactome.pkl       Reactome layer/adjacency spec for this cohort
├── splits.npz         standardised X, Y, indices, pos_weight
├── <model>.pt         trained checkpoint + loss history
├── results.json       per-head AUROC/AUPRC/F1/accuracy/CM/threshold
├── importance.npz     per-head pathway importance scores
└── tex/01..05_*.tex   per-phase LaTeX (booktabs) tables

paper/
├── figures/fig*.pdf       9 colorblind-safe figures
├── artifacts/tex/         06_top_pathways_<cohort>.tex
└── main.pdf               final 8-page manuscript
```

The 15 (model × cohort) runs do not clobber each other because each
writes to a per-cohort subdirectory.

## Re-running a single cell

If only one cell failed (e.g. PATH × thyroid), set `THERAP_RESUME=1`
to skip the cells whose `results.json` is already on disk:

```bash
THERAP_RESUME=1 \
THERAP_SLURM_EMAIL=you@example.com \
  bash slurm/submit_all.sh
```

If you want to manually rerun one cell from the login node:

```bash
source .venv-therap/bin/activate
python3 -m path.main all --cohort thyroid
python3 paper/scripts/regen_tex.py --cohort thyroid
( cd paper/figures && ./make_all.sh )
( cd paper && tectonic main.tex )
```

## Local dry-run (no SLURM)

```bash
# Single (model, cohort), no SLURM scheduling:
python3 -m binn.main all --cohort thyroid

# Full sweep sequentially:
for c in breast lung prostate head_neck thyroid; do
  for m in binn graphpath path; do
    python3 -m "$m".main all --cohort "$c"
  done
  python3 paper/scripts/regen_tex.py --cohort "$c"
done
( cd paper/figures && ./make_all.sh )
( cd paper && ./build.sh )
```

## Conformance gating

The sweep aborts before any training if any model's paper-conformance
checker (`python3 -m <model>.checker`) reports a BUG. This catches
silent drifts between the code and the reference papers (BINN /
GraphPath / PATH) before a single GPU-second is spent.
