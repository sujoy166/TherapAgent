#!/usr/bin/env bash
# Phase 3 — decode the stage label into TMT/RT/OS heads, build stratified
# 70/15/15 splits, fit the standardizer on train, compute pos_weights.
# Writes:
#   binn/artifacts/splits.npz                       (X, Y, indices, scaler, pos_w)
#   binn/artifacts/tex/03_data_splits.tex           (per-split counts)
#   binn/artifacts/tex/03_head_distribution.tex     (head weights)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$HERE/../.." && pwd)"
cd "$PROJECT_ROOT"

PY="${PYTHON:-python3}"
"$PY" -m binn.main data "$@"
