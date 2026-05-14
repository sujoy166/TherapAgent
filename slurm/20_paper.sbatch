#!/bin/bash
#SBATCH --job-name=therap_paper
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --partition=nova
#SBATCH --output=logs/therap_paper-%j.out
#SBATCH --error=logs/therap_paper-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL

# Build the ASI 2026 manuscript end-to-end after every cohort has
# finished training. Renders all colorblind-safe figures, including the
# cross-cohort summary fig8 which aggregates results.json from each of
# the five cohorts; then compiles main.tex to main.pdf with tectonic
# (or latexmk as a fallback).
#
# Submit:
#   THERAP_REPO=$PWD \
#   THERAP_SLURM_EMAIL=you@example.com \
#     sbatch --mail-user="$THERAP_SLURM_EMAIL" slurm/20_paper.sbatch
#
# Submit with a dependency on the training array (so it only runs after
# the array succeeds — done automatically by slurm/submit_all.sh):
#   sbatch --dependency=afterok:<TRAIN_JOB_ID> ...

set -euo pipefail

THERAP_REPO="${THERAP_REPO:-$PWD}"
THERAP_VENV="${THERAP_VENV:-$THERAP_REPO/.venv-therap}"
THERAP_PYTHON_MODULE="${THERAP_PYTHON_MODULE:-python/3.11}"
THERAP_TECTONIC="${THERAP_TECTONIC:-tectonic}"

echo "[paper] repo : $THERAP_REPO"

if command -v module >/dev/null 2>&1; then
    module purge
    module load "$THERAP_PYTHON_MODULE"
fi
# shellcheck disable=SC1091
source "$THERAP_VENV/bin/activate"
cd "$THERAP_REPO"

# Regenerate per-cohort LaTeX tables. Skip cohorts that did not run.
for COHORT in breast lung prostate head_neck thyroid; do
    if [[ -f "binn/artifacts/${COHORT}/results.json" \
        && -f "graphpath/artifacts/${COHORT}/results.json" \
        && -f "path/artifacts/${COHORT}/results.json" ]]; then
        echo "[paper] regen tex tables for cohort=$COHORT"
        python3 paper/scripts/regen_tex.py --cohort "$COHORT" || true
    else
        echo "[paper] skip cohort=$COHORT (results.json missing)"
    fi
done

echo "[paper] rendering figures"
( cd paper/figures && ./make_all.sh )

echo "[paper] compiling main.tex"
( cd paper && "$THERAP_TECTONIC" main.tex )

echo "[paper] done. PDF at $(realpath paper/main.pdf)"
