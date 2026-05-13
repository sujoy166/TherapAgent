#!/usr/bin/env bash
# Phase 2 — download Reactome (Homo sapiens) and build the layer/mask spec.
# Writes:
#   binn/cache/ReactomePathways.txt            (raw)
#   binn/cache/ReactomePathwaysRelation.txt    (raw)
#   binn/artifacts/reactome.pkl                (layers + masks)
#   binn/artifacts/tex/02_reactome_layers.tex  (LaTeX table)
#
# Re-running with the cache present is a no-op for the download.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$HERE/../.." && pwd)"
cd "$PROJECT_ROOT"

PY="${PYTHON:-python3}"
"$PY" -m binn.main reactome "$@"
