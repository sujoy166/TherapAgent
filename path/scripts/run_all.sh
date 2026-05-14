#!/usr/bin/env bash
# Run all PATH phases end-to-end.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"$HERE/01_setup.sh"
"$HERE/02_reactome.sh"
"$HERE/03_data.sh"
"$HERE/04_train.sh" "$@"
"$HERE/05_evaluate.sh"
echo
echo ">> PATH pipeline complete."
echo "   Checkpoints  → path/artifacts/"
echo "   LaTeX tables → path/artifacts/tex/"
