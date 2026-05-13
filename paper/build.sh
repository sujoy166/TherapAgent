#!/usr/bin/env bash
# Build the ASI 2026 manuscript end-to-end.
#
# Order:
#   1. Render colorblind-safe figures (matplotlib).
#   2. Run latexmk twice with bibtex to resolve references.
#
# Requires:
#   * pdflatex (TeX Live with acmart) on $PATH
#   * latexmk on $PATH
#   * python3 with matplotlib + numpy + pandas
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

PY="${PYTHON:-python3}"
LATEXMK="${LATEXMK:-latexmk}"

echo ">> [1/2] rendering figures"
( cd figures && ./make_all.sh )

if ! command -v "$LATEXMK" >/dev/null 2>&1; then
    echo "!! latexmk not found in PATH; skipping LaTeX build."
    echo "   Install via TeX Live (https://www.tug.org/texlive/) and re-run."
    exit 0
fi

echo ">> [2/2] compiling main.tex with latexmk"
"$LATEXMK" -pdf -interaction=nonstopmode -halt-on-error main.tex

echo
echo ">> Build complete: $HERE/main.pdf"
