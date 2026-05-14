#!/bin/bash
#SBATCH --job-name=therap_sweep
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=08:00:00
#SBATCH --partition=nova
#SBATCH --output=logs/therap_sweep-%j.out
#SBATCH --error=logs/therap_sweep-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=tirtho@iasatete.edu

# Single-GPU sequential sweep:
#   for each cohort:
#     for each model:
#       (paper-conformance check) -> (full pipeline) -> (regen tex)
#   regenerate figures and compile main.pdf
#
# All 15 (model × cohort) runs share the one allocated GPU. Wall-clock
# budget on an A100, including the Reactome download, ssGSEA-derived
# data preparation, training, evaluation, and pathway-importance
# computation, is ~80 minutes for the full sweep; --time=08:00:00 gives
# generous headroom for the largest PATH × lung case and re-downloads.
#
# Submit from the repo root:
#   THERAP_REPO=$PWD \
#   THERAP_SLURM_EMAIL=you@example.com \
#     sbatch --mail-user="$THERAP_SLURM_EMAIL" slurm/10_sweep.sbatch
#
# To resume after a partial run (skip cohorts already finished), set
#   THERAP_RESUME=1
# and the script will skip any (model, cohort) for which
# <model>/artifacts/<cohort>/results.json already exists.

set -euo pipefail

THERAP_REPO="${THERAP_REPO:-$PWD}"
THERAP_VENV="${THERAP_VENV:-$THERAP_REPO/.venv-therap}"
THERAP_PYTHON_MODULE="${THERAP_PYTHON_MODULE:-python/3.11}"
THERAP_CUDA_MODULE="${THERAP_CUDA_MODULE:-cuda/12.1}"
THERAP_TECTONIC="${THERAP_TECTONIC:-tectonic}"
THERAP_RESUME="${THERAP_RESUME:-0}"

# ── Load Python + CUDA + activate venv ─────────────────────────────────
# Robust python/3.11 loader: try the configured module name, then the
# common HPC variants, then fall back to system python3. The venv built
# by 00_install.sbatch is already pinned, so we only need the right
# python3 binary on PATH here.
load_python_module() {
    if ! command -v module >/dev/null 2>&1; then
        echo "[sweep] no 'module' command — using system python3"
        return 0
    fi
    module purge
    local cand
    for cand in \
        "$THERAP_PYTHON_MODULE" \
        python/3.11 python/3.11.7 python/3.11.5 python/3.11.4 \
        python-3.11 Python/3.11 py-python/3.11 python3/3.11; do
        if module load "$cand" >/dev/null 2>&1; then
            echo "[sweep] module load $cand  →  $(python3 --version 2>&1)"
            return 0
        fi
    done
    echo "[sweep] WARNING: no python/3.11 module found; using system python3" >&2
}
load_python_module
if [[ "${THERAP_LOAD_CUDA:-1}" == "1" ]] && command -v module >/dev/null 2>&1; then
    module load "$THERAP_CUDA_MODULE" 2>/dev/null \
        && echo "[sweep] module load $THERAP_CUDA_MODULE" \
        || echo "[sweep] CUDA module $THERAP_CUDA_MODULE not available"
fi
# shellcheck disable=SC1091
source "$THERAP_VENV/bin/activate"

cd "$THERAP_REPO"

echo "[sweep] repo : $THERAP_REPO"
echo "[sweep] venv : $THERAP_VENV"
echo "[sweep] node : $(hostname)"
echo "[sweep] gpu  : ${CUDA_VISIBLE_DEVICES:-(none)}"
echo "[sweep] resume mode : $THERAP_RESUME"

COHORTS=(breast lung prostate head_neck thyroid)
MODELS=(binn graphpath path)

# ── 1. One-time paper-conformance check for every model ────────────────
for MODEL in "${MODELS[@]}"; do
    echo "[sweep] paper-conformance check: $MODEL"
    python3 -m "${MODEL}.checker" >/dev/null
done

# ── 2. Sequential per-(cohort, model) training ─────────────────────────
for COHORT in "${COHORTS[@]}"; do
    echo
    echo "[sweep] ════════════ cohort = $COHORT ════════════"
    for MODEL in "${MODELS[@]}"; do
        marker="${MODEL}/artifacts/${COHORT}/results.json"
        if [[ "$THERAP_RESUME" == "1" && -f "$marker" ]]; then
            echo "[sweep] skip $MODEL × $COHORT (results.json present)"
            continue
        fi
        echo "[sweep] $MODEL × $COHORT  → starting"
        SECONDS=0
        python3 -m "${MODEL}.main" all --cohort "$COHORT"
        echo "[sweep] $MODEL × $COHORT  done in ${SECONDS}s"
    done
    # Per-cohort LaTeX regeneration
    python3 paper/scripts/regen_tex.py --cohort "$COHORT" || true
done

# ── 3. Figures + paper build ───────────────────────────────────────────
echo
echo "[sweep] rendering figures"
( cd paper/figures && ./make_all.sh )

echo "[sweep] compiling main.tex"
( cd paper && "$THERAP_TECTONIC" main.tex )

echo
echo "[sweep] complete."
echo "[sweep] PDF: $(realpath paper/main.pdf)"
echo "[sweep] tables/figures/results all auto-organised in:"
echo "        binn/artifacts/<cohort>/, graphpath/artifacts/<cohort>/,"
echo "        path/artifacts/<cohort>/, paper/figures/, paper/artifacts/tex/"
