#!/usr/bin/env bash
# Two-step SLURM driver for TherapAgent on Nova:
#   00_install.sbatch  → one-time venv build (skip with NO_INSTALL=1)
#   10_sweep.sbatch    → sequential sweep over all 15 (model × cohort)
#                        runs on a single GPU, followed by paper rebuild
#
# Usage (from the repo root):
#   THERAP_SLURM_EMAIL=you@example.com bash slurm/submit_all.sh
#
# Env overrides (with defaults shown):
#   THERAP_REPO=$PWD             # absolute path to the cloned repo
#   THERAP_VENV=$THERAP_REPO/.venv-therap
#   THERAP_PYTHON_MODULE=python/3.11
#   THERAP_CUDA_MODULE=cuda/12.1
#   THERAP_SLURM_EMAIL=...       # REQUIRED: --mail-user target
#   NO_INSTALL=1                 # skip the install step (venv already there)
#   THERAP_RESUME=1              # only run (model, cohort) without results.json
#
# After submission:
#   squeue --me
#   tail -f slurm/logs/therap_sweep-*.out
# Final PDF: paper/main.pdf

set -euo pipefail

THERAP_REPO="${THERAP_REPO:-$PWD}"
THERAP_VENV="${THERAP_VENV:-$THERAP_REPO/.venv-therap}"
THERAP_SLURM_EMAIL="${THERAP_SLURM_EMAIL:-}"

if [[ -z "$THERAP_SLURM_EMAIL" ]]; then
    echo "Set THERAP_SLURM_EMAIL=you@example.com so SLURM can send you BEGIN/END/FAIL mail." >&2
    exit 2
fi

export THERAP_REPO THERAP_VENV THERAP_SLURM_EMAIL

mkdir -p "$THERAP_REPO/slurm/logs"
cd "$THERAP_REPO"

echo ">> submitting from: $THERAP_REPO"
echo ">> notify         : $THERAP_SLURM_EMAIL"

# ── 1. Optional one-time install ───────────────────────────────────────
PREDECESSOR=""
if [[ "${NO_INSTALL:-0}" != "1" ]]; then
    JID_INSTALL=$(sbatch --parsable \
        --mail-user="$THERAP_SLURM_EMAIL" \
        slurm/00_install.sbatch)
    echo ">> install job   : $JID_INSTALL"
    PREDECESSOR="--dependency=afterok:$JID_INSTALL"
fi

# ── 2. Sequential single-GPU sweep + paper build ──────────────────────
JID_SWEEP=$(sbatch --parsable \
    $PREDECESSOR \
    --mail-user="$THERAP_SLURM_EMAIL" \
    slurm/10_sweep.sbatch)
echo ">> sweep job     : $JID_SWEEP  (5 cohorts × 3 models, sequential)"

echo
echo ">> Watch progress:"
echo "      squeue --me"
echo "      tail -f slurm/logs/therap_sweep-${JID_SWEEP}.out"
echo
echo ">> When it finishes, the manuscript will be at:"
echo "      $THERAP_REPO/paper/main.pdf"
