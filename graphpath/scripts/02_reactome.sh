#!/usr/bin/env bash
# GraphPath Phase 2 — download Reactome and build pathway-pathway adjacency.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$HERE/../.." && pwd)"
cd "$PROJECT_ROOT"
PY="${PYTHON:-python3}"
"$PY" -m graphpath.main reactome "$@"
