"""Figure 1 — schematic comparison of the three pathway-informed architectures.

Generates `paper/figures/fig1_architectures.pdf`. Requires no training
artifacts; only matplotlib.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

from _style import OKABE_ITO, apply_style, color_for


def _box(ax, x, y, w, h, label, fc, ec=OKABE_ITO["black"], fontsize=8):
    rect = mpatches.FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.04",
        linewidth=0.8, edgecolor=ec, facecolor=fc,
    )
    ax.add_patch(rect)
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
            fontsize=fontsize, wrap=True)


def _arrow(ax, x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", linewidth=0.7,
                                color=OKABE_ITO["black"]))


def draw_binn(ax):
    ax.set_title("BINN (Hartman et al.\\ 2023)", fontsize=10)
    fc = color_for("BINN") + "30"
    # Input
    _box(ax, 0.05, 0.78, 0.9, 0.10, "ssGSEA scores\n(N pathways)", fc)
    # Layers
    labels = [
        "Sparse Linear (Reactome mask) → tanh → BN → Dropout",
        "Sparse Linear (parent layer 2) → tanh → BN → Dropout",
        "Sparse Linear (parent layer 3) → tanh → BN → Dropout",
    ]
    for i, lab in enumerate(labels):
        _box(ax, 0.05, 0.58 - i * 0.18, 0.6, 0.10, lab, fc, fontsize=7)
    # Heads — one per layer
    _box(ax, 0.72, 0.78, 0.23, 0.10, "head\\,$_0$ (3-d)", color_for("BINN") + "60", fontsize=7)
    for i in range(3):
        _box(ax, 0.72, 0.58 - i * 0.18, 0.23, 0.10,
             f"head\\,$_{i+1}$ (3-d)", color_for("BINN") + "60", fontsize=7)
    # Final average
    _box(ax, 0.30, 0.05, 0.40, 0.10,
         r"$p_h = \mathrm{mean}_\ell\, \sigma(\mathrm{head}_\ell[h])$",
         OKABE_ITO["yellow"] + "60", fontsize=7)
    # Arrows
    for i in range(4):
        _arrow(ax, 0.83, 0.78 - i * 0.18, 0.55, 0.10 + 0.04)
    for i in range(3):
        _arrow(ax, 0.35, 0.78 - i * 0.18, 0.35, 0.58 - i * 0.18 + 0.10)


def draw_graphpath(ax):
    ax.set_title("GraphPath (Ma \\& Wang 2024)", fontsize=10)
    fc = color_for("GraphPath") + "30"
    _box(ax, 0.05, 0.80, 0.9, 0.10, "ssGSEA scores (N pathways)", fc)
    _box(ax, 0.05, 0.65, 0.9, 0.10,
         "Per-pathway proj.\\ → tanh \\quad (1 → $F'$)", fc, fontsize=7)
    _box(ax, 0.05, 0.45, 0.9, 0.13,
         "Multi-head GAT ($K=3$, ELU)\nattention on Reactome adjacency", fc,
         fontsize=7)
    _box(ax, 0.05, 0.28, 0.9, 0.10,
         "Per-node readout (shared Linear, tanh)\\quad → vector $\\in\\mathbb{R}^N$",
         fc, fontsize=7)
    _box(ax, 0.05, 0.12, 0.9, 0.10,
         "FC → sigmoid \\quad (3-d multi-label)", color_for("GraphPath") + "60",
         fontsize=7)
    for y_from, y_to in [(0.80, 0.75), (0.65, 0.58), (0.45, 0.38),
                          (0.28, 0.22)]:
        _arrow(ax, 0.50, y_from, 0.50, y_to)


def draw_path(ax):
    ax.set_title("PATH (Howlader et al.\\ 2026)", fontsize=10)
    fc = color_for("PATH") + "30"
    _box(ax, 0.05, 0.82, 0.9, 0.10, "ssGSEA scores (N pathways)", fc)
    _box(ax, 0.05, 0.69, 0.55, 0.10,
         "Per-pathway proj.\\ → tanh \\quad (1 → $d$)", fc, fontsize=7)
    _box(ax, 0.62, 0.69, 0.33, 0.10,
         "Laplacian PE\n(top-$k$ eigvecs)", color_for("PATH") + "60", fontsize=7)
    _box(ax, 0.05, 0.50, 0.9, 0.14,
         "L = 2 × edge-aware Graph Transformer blocks\n(soft mask, edge bias, $H=4$ heads, FFN×4, BN)",
         fc, fontsize=7)
    _box(ax, 0.05, 0.32, 0.9, 0.10,
         "Attention-weighted readout \\quad ($g = \\sum w_p\\,x_p$)",
         fc, fontsize=7)
    _box(ax, 0.05, 0.16, 0.9, 0.10,
         "BN → GELU → Dropout → FC → sigmoid (3-d multi-label)",
         color_for("PATH") + "60", fontsize=7)
    for y_from, y_to in [(0.82, 0.79), (0.69, 0.64), (0.50, 0.42),
                          (0.32, 0.26)]:
        _arrow(ax, 0.50, y_from, 0.50, y_to)


def main() -> None:
    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 4.0))
    for ax in axes:
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.axis("off")
    draw_binn(axes[0])
    draw_graphpath(axes[1])
    draw_path(axes[2])

    fig.suptitle(
        "Three pathway-informed architectures for predicting therapy-response phenotypes "
        "(TMT / RT / OS$\\geq$180 d) from Reactome ssGSEA scores",
        fontsize=9, y=1.02,
    )
    out = Path(__file__).parent / "fig1_architectures.pdf"
    fig.savefig(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
