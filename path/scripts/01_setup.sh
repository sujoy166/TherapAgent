#!/usr/bin/env bash
# PATH Phase 1 — install Python dependencies and emit env LaTeX table.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$HERE/../.." && pwd)"
cd "$PROJECT_ROOT"
PY="${PYTHON:-python3}"
echo ">> Using interpreter: $($PY --version)"
"$PY" -m pip install --quiet --upgrade pip
"$PY" -m pip install --quiet numpy pandas scipy scikit-learn torch requests
"$PY" -c "import torch, numpy, pandas, sklearn; print('>> dependency import OK')"
"$PY" -m path.main env
