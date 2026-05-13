#!/usr/bin/env bash
# GraphPath Phase 3 — decode labels, build 80/10/10 splits, fit scaler.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$HERE/../.." && pwd)"
cd "$PROJECT_ROOT"
PY="${PYTHON:-python3}"
"$PY" -m graphpath.main data "$@"
