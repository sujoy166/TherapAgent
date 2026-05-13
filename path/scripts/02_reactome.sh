#!/usr/bin/env bash
# PATH Phase 2 — download Reactome GMT and build the Jaccard adjacency.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$HERE/../.." && pwd)"
cd "$PROJECT_ROOT"
PY="${PYTHON:-python3}"
"$PY" -m path.main reactome "$@"
