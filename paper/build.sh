#!/usr/bin/env bash
# Build the ASI 2026 manuscript end-to-end.
#
# Order:
#   1. Render colorblind-safe figures (matplotlib).
#   2. Compile main.tex → main.pdf.
#
# Prefers `tectonic` (self-contained, auto-downloads packages) and falls
# back to `latexmk` if tectonic is not on PATH.
#
# Install hints:
#   brew install tectonic
#   # or: curl --proto '=https' --tlsv1.2 -fsSL https://drop-sh.fullyjustified.net | sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

PY="${PYTHON:-python3}"

echo ">> [1/2] rendering figures"
( cd figures && ./make_all.sh )

# Pick the available engine.
if command -v tectonic >/dev/null 2>&1; then
    ENGINE=tectonic
elif command -v "$HOME/.local/bin/tectonic" >/dev/null 2>&1; then
    ENGINE="$HOME/.local/bin/tectonic"
elif command -v latexmk >/dev/null 2>&1; then
    ENGINE=latexmk
else
    echo "!! No LaTeX engine on PATH (tried: tectonic, latexmk)."
    echo "   Install tectonic: https://tectonic-typesetting.github.io/"
    echo "   or a TeX Live distribution: https://www.tug.org/texlive/"
    exit 1
fi

echo ">> [2/2] compiling main.tex with $ENGINE"
case "$(basename "$ENGINE")" in
    tectonic) "$ENGINE" main.tex ;;
    latexmk)  "$ENGINE" -pdf -interaction=nonstopmode -halt-on-error main.tex ;;
esac

echo
echo ">> Build complete: $HERE/main.pdf"
