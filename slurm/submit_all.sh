#!/usr/bin/env bash
# End-to-end SLURM driver for TherapAgent:
#   00_install.sbatch  → builds the venv (skip with NO_INSTALL=1)
#   10_train.sbatch    → 15-task array (3 models × 5 cohorts)
#   20_paper.sbatch    → depends on 10's success; rebuilds main.pdf
#
# Usage (from the repo root):
#   THERAP_SLURM_EMAIL=you@example.com bash slurm/submit_all.sh
#
# Env overrides (with defaults shown):
#   THERAP_REPO=$PWD             # absolute path to the cloned repo
#   THERAP_VENV=$THERAP_REPO/.venv-therap
#   THERAP_PYTHON_MODULE=python/3.11
#   THERAP_CUDA_MODULE=cuda/12.1
#   THERAP_SLURM_EMAIL=you@example.com
#   NO_INSTALL=1                 # skip the install step (venv already there)
#
# After submission, see queue status with:  squeue --me
# Per-task logs land in slurm/logs/

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
    echo ">> install job  : $JID_INSTALL"
    PREDECESSOR="--dependency=afterok:$JID_INSTALL"
fi

# ── 2. Training array (15 tasks) ──────────────────────────────────────
JID_TRAIN=$(sbatch --parsable \
    $PREDECESSOR \
    --mail-user="$THERAP_SLURM_EMAIL" \
    slurm/10_train.sbatch)
echo ">> train array  : $JID_TRAIN  (3 models × 5 cohorts = 15 tasks)"

# ── 3. Paper build (depends on training success) ──────────────────────
JID_PAPER=$(sbatch --parsable \
    --dependency=afterok:$JID_TRAIN \
    --mail-user="$THERAP_SLURM_EMAIL" \
    slurm/20_paper.sbatch)
echo ">> paper build  : $JID_PAPER  (depends on $JID_TRAIN)"

echo
echo ">> Watch progress:"
echo "      squeue --me"
echo "      tail -f slurm/logs/therap_train-${JID_TRAIN}_0.out"
echo
echo ">> When all green, the final PDF will be at:"
echo "      $THERAP_REPO/paper/main.pdf"
