#!/usr/bin/env bash
# GraphPath Phase 6 — offline architecture sanity test (no internet needed).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$HERE/../.." && pwd)"
cd "$PROJECT_ROOT"
PY="${PYTHON:-python3}"
"$PY" -m graphpath.tests.test_smoke
