#!/bin/bash
#SBATCH --job-name=therap_install
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --partition=nova
#SBATCH --output=logs/therap_install-%j.out
#SBATCH --error=logs/therap_install-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL

# One-time environment setup for the TherapAgent training sweep.
# Submit from the repo root:
#   THERAP_REPO=$PWD \
#   THERAP_SLURM_EMAIL=you@example.com \
#     sbatch --mail-user="$THERAP_SLURM_EMAIL" slurm/00_install.sbatch
#
# Creates $THERAP_VENV (default $THERAP_REPO/.venv-therap) and installs
# requirements.txt into it. Idempotent — re-running just refreshes pip.

set -euo pipefail

THERAP_REPO="${THERAP_REPO:-$PWD}"
THERAP_VENV="${THERAP_VENV:-$THERAP_REPO/.venv-therap}"
THERAP_PYTHON_MODULE="${THERAP_PYTHON_MODULE:-python/3.11}"

echo "[install] repo  : $THERAP_REPO"
echo "[install] venv  : $THERAP_VENV"

# Nova-style module load. Falls back to whichever python3 is on PATH if
# `module` is not available (e.g. running this script locally).
if command -v module >/dev/null 2>&1; then
    module purge
    module load "$THERAP_PYTHON_MODULE"
fi

cd "$THERAP_REPO"
python3 -m venv "$THERAP_VENV"
source "$THERAP_VENV/bin/activate"
python -m pip install --quiet --upgrade pip wheel
python -m pip install --quiet -r requirements.txt

# Verify the imports the pipeline actually uses.
python -c "import torch, numpy, pandas, sklearn, scipy, matplotlib; \
print('[install] torch', torch.__version__, '| cuda', torch.cuda.is_available())"

echo "[install] done."
