"""Colorblind-safe matplotlib style for the ASI 2026 paper.

Uses the Okabe-Ito categorical palette (Nature Methods recommendation; Okabe
& Ito 2008) for distinct categories and the viridis colormap for continuous
gradients.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib import rcParams

# Okabe-Ito 8-colour palette (hex codes verified against Color Universal Design)
OKABE_ITO = {
    "orange":         "#E69F00",
    "sky_blue":       "#56B4E9",
    "bluish_green":   "#009E73",
    "yellow":         "#F0E442",
    "blue":           "#0072B2",
    "vermillion":     "#D55E00",
    "reddish_purple": "#CC79A7",
    "black":          "#000000",
}

# Model assignments — chosen to be discriminable for both deuteranopes and
# protanopes (avoid red/green confusion pairs).
MODEL_COLORS = {
    "BINN":      OKABE_ITO["blue"],
    "GraphPath": OKABE_ITO["orange"],
    "PATH":      OKABE_ITO["bluish_green"],
}

HEAD_HATCHES = {"TMT": "", "RT": "//", "OS": "xx"}


def apply_style() -> None:
    """Set rcParams so every figure looks consistent and is print-safe."""
    rcParams.update({
        "figure.dpi":         150,
        "savefig.dpi":        300,
        "savefig.bbox":       "tight",
        "savefig.format":     "pdf",
        "pdf.fonttype":       42,    # embed Type-42 fonts so editors can edit text
        "ps.fonttype":        42,
        "font.family":        "serif",
        "font.serif":         ["Times New Roman", "DejaVu Serif", "Times"],
        "font.size":          9,
        "axes.titlesize":     10,
        "axes.labelsize":     9,
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "axes.linewidth":     0.6,
        "xtick.labelsize":    8,
        "ytick.labelsize":    8,
        "legend.fontsize":    8,
        "legend.frameon":     False,
        "lines.linewidth":    1.4,
        "lines.markersize":   4,
        "grid.linewidth":     0.4,
        "grid.alpha":         0.3,
    })


def color_for(model_name: str) -> str:
    return MODEL_COLORS.get(model_name, OKABE_ITO["black"])
