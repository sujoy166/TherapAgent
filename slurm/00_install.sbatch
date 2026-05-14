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
#SBATCH --mail-user=tirtho@iastate.edu

# One-time environment setup for the TherapAgent training sweep.
# Submit from the repo root:
#   THERAP_REPO=$PWD \
#   THERAP_SLURM_EMAIL=you@example.com \
#     sbatch --mail-user="$THERAP_SLURM_EMAIL" slurm/00_install.sbatch
#
# Creates $THERAP_VENV (default $THERAP_REPO/.venv-therap) and installs
# requirements.txt into it. Idempotent — re-running just refreshes pip.

set -euo pipefail

# Hard-coded canonical repo path on Nova. Override only if the
# repository has been moved.
THERAP_REPO="${THERAP_REPO:-/work/mech-ai-scratch/tirtho/TherapAgent}"
cd "$THERAP_REPO"
THERAP_VENV="${THERAP_VENV:-$THERAP_REPO/.venv-therap}"
THERAP_PYTHON_MODULE="${THERAP_PYTHON_MODULE:-python/3.11}"

echo "[install] repo  : $THERAP_REPO"
echo "[install] venv  : $THERAP_VENV"

# Load a Python 3.11 module. Tries the user-specified module first, then
# the most common Nova / HPC naming variants, and finally falls back to
# whatever `python3` is on PATH. Always picks 3.11 if available because
# our torch/pandas wheels are pinned to that minor.
load_python_module() {
    if ! command -v module >/dev/null 2>&1; then
        echo "[install] no 'module' command — using system python3 ($(python3 --version 2>&1))"
        return 0
    fi
    module purge
    local cand
    for cand in \
        "$THERAP_PYTHON_MODULE" \
        python/3.11 python/3.11.7 python/3.11.5 python/3.11.4 \
        python-3.11 Python/3.11 py-python/3.11 python3/3.11; do
        if module load "$cand" >/dev/null 2>&1; then
            echo "[install] module load $cand  →  $(python3 --version 2>&1)"
            return 0
        fi
    done
    echo "[install] WARNING: no python/3.11 module found; falling back to system python3" >&2
    return 1
}
load_python_module

python3 -m venv "$THERAP_VENV"
source "$THERAP_VENV/bin/activate"
python -m pip install --quiet --upgrade pip wheel
python -m pip install --quiet -r requirements.txt

# Verify the imports the pipeline actually uses.
python -c "import torch, numpy, pandas, sklearn, scipy, matplotlib; \
print('[install] torch', torch.__version__, '| cuda', torch.cuda.is_available())"

echo "[install] done."
