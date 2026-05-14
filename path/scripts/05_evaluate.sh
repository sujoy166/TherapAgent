#!/usr/bin/env bash
# PATH Phase 5 — per-head metrics on val + test.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$HERE/../.." && pwd)"
cd "$PROJECT_ROOT"
PY="${PYTHON:-python3}"
"$PY" -m path.main evaluate "$@"
