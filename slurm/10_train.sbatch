#!/bin/bash
#SBATCH --job-name=therap_train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=04:00:00
#SBATCH --partition=nova
#SBATCH --array=0-14
#SBATCH --output=logs/therap_train-%A_%a.out
#SBATCH --error=logs/therap_train-%A_%a.err
#SBATCH --mail-type=BEGIN,END,FAIL

# 15-task array: every (model, cohort) combination on the TherapAgent
# benchmark. Each task runs that model's full phase pipeline (env →
# reactome → data → train → evaluate) on one cohort and writes its
# artifacts into <model>/artifacts/<cohort>/.
#
# Submit:
#   THERAP_REPO=$PWD \
#   THERAP_SLURM_EMAIL=you@example.com \
#     sbatch --mail-user="$THERAP_SLURM_EMAIL" slurm/10_train.sbatch
#
# Resume a single failed cell, e.g. PATH×thyroid (index 14):
#     sbatch --array=14 --mail-user="$THERAP_SLURM_EMAIL" slurm/10_train.sbatch
#
# Resource budget per task is sized for PATH×lung (the largest run);
# BINN and GraphPath finish in a fraction of the time but use the same
# allocation for scheduling simplicity. Override at submit with e.g.
# `--time=01:00:00` for the smaller models.

set -euo pipefail

THERAP_REPO="${THERAP_REPO:-$PWD}"
THERAP_VENV="${THERAP_VENV:-$THERAP_REPO/.venv-therap}"
THERAP_PYTHON_MODULE="${THERAP_PYTHON_MODULE:-python/3.11}"

# ── Resolve which (model, cohort) this array task handles ──────────────
MODELS=(binn graphpath path)
COHORTS=(breast lung prostate head_neck thyroid)
N_MODELS=${#MODELS[@]}
N_COHORTS=${#COHORTS[@]}

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    echo "[train] SLURM_ARRAY_TASK_ID not set; abort." >&2
    exit 2
fi
idx=$SLURM_ARRAY_TASK_ID
model_idx=$(( idx / N_COHORTS ))
cohort_idx=$(( idx % N_COHORTS ))
MODEL="${MODELS[$model_idx]}"
COHORT="${COHORTS[$cohort_idx]}"

echo "[train] repo    : $THERAP_REPO"
echo "[train] venv    : $THERAP_VENV"
echo "[train] task    : $idx → model=$MODEL cohort=$COHORT"
echo "[train] node    : $(hostname)"
echo "[train] gpu     : ${CUDA_VISIBLE_DEVICES:-(none)}"

# ── Load Python + activate venv ────────────────────────────────────────
if command -v module >/dev/null 2>&1; then
    module purge
    module load "$THERAP_PYTHON_MODULE"
    if [[ "${THERAP_LOAD_CUDA:-1}" == "1" ]]; then
        module load "${THERAP_CUDA_MODULE:-cuda/12.1}" || true
    fi
fi
# shellcheck disable=SC1091
source "$THERAP_VENV/bin/activate"

cd "$THERAP_REPO"

# Pre-flight: paper-conformance check for this model. Fails fast if the
# implementation has drifted from the reference paper.
echo "[train] paper-conformance check"
python3 -m "${MODEL}.checker" >/dev/null

# Run the full phase pipeline. main.py handles --cohort and writes into
# <model>/artifacts/<cohort>/.
echo "[train] python3 -m ${MODEL}.main all --cohort ${COHORT}"
python3 -m "${MODEL}.main" all --cohort "$COHORT"

# Regenerate this cohort's per-phase LaTeX tables from the saved
# checkpoints — this is what the paper build later \input{}s.
echo "[train] regenerating LaTeX tables for cohort=$COHORT"
python3 paper/scripts/regen_tex.py --cohort "$COHORT" || true

echo "[train] done: $MODEL / $COHORT"
