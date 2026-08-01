#!/bin/bash
#SBATCH --job-name=therap_cv
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --partition=nova
#SBATCH --account=mech-ai
#SBATCH --qos=normal
#SBATCH --output=logs/therap_cv-%j.out
#SBATCH --error=logs/therap_cv-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=tirtho@iastate.edu
#
# Unified identical-split cross-validation sweep.
#
# For every repeat seed in THERAP_SEEDS, all three models (BINN, GraphPath,
# PATH) are trained and evaluated on ALL five cohorts using the SAME
# 80/10/10 stratified split (the split function is shared, so at a fixed
# seed every model sees byte-identical train/val/test folds per cohort).
# After each run we snapshot results.json into
#   results_cv/seed<seed>/<model>__<cohort>.json
# so aggregate_cv.py can compute mean +/- 95% CI and paired significance
# tests across seeds. This directly answers the reviewers' request for a
# unified split, repeated evaluation, and uncertainty quantification.
#
# Submit from the repo root:
#   sbatch slurm/20_cv.sbatch
# Resume after a timeout (skips snapshots already present):
#   THERAP_RESUME=1 sbatch slurm/20_cv.sbatch

set -euo pipefail

THERAP_REPO="${THERAP_REPO:-/work/mech-ai-scratch/tirtho/TherapAgent}"
cd "$THERAP_REPO"
THERAP_VENV="${THERAP_VENV:-$THERAP_REPO/.venv-therap}"
THERAP_PYTHON_MODULE="${THERAP_PYTHON_MODULE:-python/3.11}"
THERAP_CUDA_MODULE="${THERAP_CUDA_MODULE:-cuda/12.1}"
THERAP_RESUME="${THERAP_RESUME:-0}"
THERAP_SEEDS="${THERAP_SEEDS:-0 1 2 3 4}"

mkdir -p "$THERAP_REPO/slurm/logs"

# ── environment ────────────────────────────────────────────────────────
if command -v module >/dev/null 2>&1; then
    module purge || true
    for cand in "$THERAP_PYTHON_MODULE" python/3.11 python-3.11; do
        module load "$cand" >/dev/null 2>&1 && break
    done
    module load "$THERAP_CUDA_MODULE" 2>/dev/null || true
fi
# shellcheck disable=SC1091
source "$THERAP_VENV/bin/activate"

echo "[cv] repo  : $THERAP_REPO"
echo "[cv] venv  : $THERAP_VENV"
echo "[cv] node  : $(hostname)"
echo "[cv] gpu   : ${CUDA_VISIBLE_DEVICES:-(none)}"
echo "[cv] seeds : $THERAP_SEEDS"
python -c "import torch;print('[cv] torch',torch.__version__,'cuda',torch.cuda.is_available())"

# Cohorts default to all five; override for a smoke run, e.g.
#   THERAP_COHORTS="thyroid" THERAP_SEEDS="0" sbatch slurm/20_cv.sbatch
read -r -a COHORTS <<< "${THERAP_COHORTS:-breast lung prostate head_neck thyroid}"
MODELS=(binn graphpath path)

# ── conformance checks (once) ──────────────────────────────────────────
for MODEL in "${MODELS[@]}"; do
    python3 -m "${MODEL}.checker" >/dev/null 2>&1 || true
done

# ── repeated identical-split sweep ─────────────────────────────────────
for SEED in $THERAP_SEEDS; do
    export THERAP_SEED="$SEED"
    OUT="results_cv/seed${SEED}"
    mkdir -p "$OUT"
    echo
    echo "[cv] ═══════════════ seed = $SEED ═══════════════"
    for COHORT in "${COHORTS[@]}"; do
        for MODEL in "${MODELS[@]}"; do
            snap="${OUT}/${MODEL}__${COHORT}.json"
            if [[ "$THERAP_RESUME" == "1" && -f "$snap" ]]; then
                echo "[cv] skip $MODEL x $COHORT (seed $SEED) — snapshot present"
                continue
            fi
            echo "[cv] seed=$SEED  $MODEL x $COHORT  → starting"
            SECONDS=0
            python3 -m "${MODEL}.main" all --cohort "$COHORT"
            cp "${MODEL}/artifacts/${COHORT}/results.json" "$snap"
            cp "${MODEL}/artifacts/${COHORT}/preds.npz" \
               "${OUT}/${MODEL}__${COHORT}.preds.npz"
            echo "[cv] seed=$SEED  $MODEL x $COHORT  done in ${SECONDS}s → $snap"
        done
    done
done

# ── aggregate: mean/CI + significance tests + LaTeX table ──────────────
python3 paper/scripts/aggregate_cv.py --seeds $THERAP_SEEDS

echo
echo "[cv] complete. Aggregated table → results_cv/cv_table.tex"
