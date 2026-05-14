#!/usr/bin/env bash
# Phase 6 (optional) — offline end-to-end smoke test using a synthetic
# Reactome hierarchy. Use this when you have no internet access or want
# a fast verification that the architecture, training loop, and metrics
# all wire up correctly.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$HERE/../.." && pwd)"
cd "$PROJECT_ROOT"

PY="${PYTHON:-python3}"
"$PY" -m binn.tests.test_smoke
