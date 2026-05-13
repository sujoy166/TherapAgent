#!/usr/bin/env bash
# Phase 1 — install Python dependencies and emit the environment LaTeX table.
#
# Usage:  ./binn/scripts/01_setup.sh
# Env:    PYTHON=python3.11 ./binn/scripts/01_setup.sh   (to pin interpreter)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$HERE/../.." && pwd)"
cd "$PROJECT_ROOT"

PY="${PYTHON:-python3}"
echo ">> Using interpreter: $($PY --version)"

# Resolved versions match what BINN's code paths actually need.
"$PY" -m pip install --quiet --upgrade pip
"$PY" -m pip install --quiet \
    numpy pandas scipy scikit-learn torch joblib requests

"$PY" -c "import torch, numpy, pandas, sklearn, scipy; print('>> dependency import OK')"

# Emit binn/artifacts/tex/01_environment.tex
"$PY" -m binn.main env
