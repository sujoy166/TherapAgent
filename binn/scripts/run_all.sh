#!/usr/bin/env bash
# Run every phase end-to-end: env → reactome → data → train → evaluate.
# Forwarded flags (e.g. --smoke) are passed only to the training phase.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

"$HERE/01_setup.sh"
"$HERE/02_reactome.sh"
"$HERE/03_data.sh"
"$HERE/04_train.sh" "$@"
"$HERE/05_evaluate.sh"

echo
echo ">> Pipeline complete."
echo "   Checkpoints  → binn/artifacts/"
echo "   LaTeX tables → binn/artifacts/tex/"
