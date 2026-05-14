#!/usr/bin/env bash
# Phase 4 — train the multi-head BINN on Reactome-masked layers.
# Writes:
#   binn/artifacts/binn.pt                       (state dict + scaler + names)
#   binn/artifacts/tex/04_training_summary.tex   (hyperparams + losses)
#
# Pass-through flags:
#   --smoke               cap epochs at 5 (sanity)
#   --epochs N            override Config.max_epochs
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$HERE/../.." && pwd)"
cd "$PROJECT_ROOT"

PY="${PYTHON:-python3}"
"$PY" -m binn.main train "$@"
