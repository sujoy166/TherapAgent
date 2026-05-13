#!/usr/bin/env bash
# Render every paper/figures/*.py figure to PDF (Okabe-Ito + viridis,
# colorblind-safe, ready for ACM acmart).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
PY="${PYTHON:-python3}"

for f in fig1_architectures.py fig2_metric_bars.py fig3_confusion.py fig4_label_imbalance.py; do
    echo ">> $f"
    "$PY" "$f"
done
echo ">> all figures rendered into $HERE"
