#!/usr/bin/env bash
# Phase 5 — evaluate the trained BINN on val + test splits.
# Writes:
#   binn/artifacts/results.json              (per-head AUROC/AUPRC/F1/acc/CM)
#   binn/artifacts/tex/05_metrics.tex        (LaTeX table, 3 heads × 2 splits)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$HERE/../.." && pwd)"
cd "$PROJECT_ROOT"

PY="${PYTHON:-python3}"
"$PY" -m binn.main evaluate "$@"
