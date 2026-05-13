"""Figure 1 — clean three-column architecture schematic.

One vertical pipeline per model. No crossing arrows. Math text uses
matplotlib mathtext (raw strings) so super/subscripts and Greek letters
render properly. Colours follow Okabe-Ito (blue / orange / green for
BINN / GraphPath / PATH).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

from _style import OKABE_ITO, apply_style, color_for


# --------------------------------------------------------------------------
# Drawing helpers
# --------------------------------------------------------------------------
def _box(ax, y_top, height, text, *, face, edge=None, fontsize=7.5,
         bold=False, italic=False):
    """Draw a rounded box centred horizontally between x=0.04 and x=0.96."""
    x = 0.04
    w = 0.92
    edge = edge or OKABE_ITO["black"]
    rect = mpatches.FancyBboxPatch(
        (x, y_top - height), w, height,
        boxstyle="round,pad=0.005,rounding_size=0.025",
        linewidth=0.7, edgecolor=edge, facecolor=face,
    )
    ax.add_patch(rect)
    weight = "bold" if bold else "normal"
    style = "italic" if italic else "normal"
    ax.text(x + w / 2, y_top - height / 2, text,
            ha="center", va="center", fontsize=fontsize,
            weight=weight, style=style, wrap=True)
    return y_top - height


def _arrow(ax, y_from, y_to, x=0.5, color=None):
    color = color or OKABE_ITO["black"]
    arrow = FancyArrowPatch(
        (x, y_from), (x, y_to),
        arrowstyle="-|>", mutation_scale=8,
        linewidth=0.8, color=color,
    )
    ax.add_patch(arrow)


def _side_tap(ax, y, label, color):
    """Annotate an auxiliary-head tap on the right side."""
    x_box = 0.96
    x_lbl = 1.02
    # short horizontal line + dot
    ax.plot([x_box, x_lbl - 0.005], [y, y], color=color, linewidth=0.6,
            clip_on=False)
    ax.plot([x_lbl - 0.005], [y], "o", markersize=2.5, color=color,
            clip_on=False)
    ax.text(x_lbl, y, label, ha="left", va="center", fontsize=6.5,
            color=color, clip_on=False)


# --------------------------------------------------------------------------
# Per-model panels
# --------------------------------------------------------------------------
def draw_binn(ax):
    ax.set_title("BINN  (Hartman et al., 2023)", fontsize=9, pad=4)
    fc_input  = color_for("BINN") + "30"
    fc_hidden = color_for("BINN") + "55"
    fc_out    = OKABE_ITO["yellow"] + "60"

    y = 0.97
    # Input row
    y = _box(ax, y, 0.10,
             r"ssGSEA scores  (N pathways)",
             face=fc_input, bold=True)
    _side_tap(ax, y + 0.05, r"head$_0$", color_for("BINN"))

    # Three hidden sparse-mask blocks
    block_labels = [
        r"Sparse Linear  (Reactome mask, $L_1$) $\to$ tanh $\to$ BN $\to$ Drop",
        r"Sparse Linear  (Reactome parents, $L_2$) $\to$ tanh $\to$ BN $\to$ Drop",
        r"Sparse Linear  (Reactome parents, $L_3$) $\to$ tanh $\to$ BN $\to$ Drop",
    ]
    for i, label in enumerate(block_labels, start=1):
        _arrow(ax, y, y - 0.04)
        y -= 0.04
        y = _box(ax, y, 0.10, label, face=fc_hidden, fontsize=6.8)
        _side_tap(ax, y + 0.05, rf"head$_{i}$", color_for("BINN"))

    # Final output (mean of per-layer sigmoids)
    _arrow(ax, y, y - 0.05)
    y -= 0.05
    _box(ax, y, 0.09,
         r"$\hat{p}_h = \mathrm{mean}_{\ell}\,\sigma(\mathrm{head}_\ell[h])$",
         face=fc_out, bold=True, fontsize=8)


def draw_graphpath(ax):
    ax.set_title("GraphPath  (Ma & Wang, 2024)", fontsize=9, pad=4)
    fc_input  = color_for("GraphPath") + "30"
    fc_hidden = color_for("GraphPath") + "55"
    fc_out    = OKABE_ITO["yellow"] + "60"

    y = 0.97
    y = _box(ax, y, 0.10,
             r"ssGSEA scores  (N pathways)",
             face=fc_input, bold=True)
    _arrow(ax, y, y - 0.04); y -= 0.04

    y = _box(ax, y, 0.10,
             r"Per-pathway projection  $(1 \to F')$",
             face=fc_input, fontsize=7)
    _arrow(ax, y, y - 0.04); y -= 0.04

    y = _box(ax, y, 0.13,
             "Multi-head GAT  ($K{=}3$ heads, ELU)\n"
             "attention over Reactome adjacency",
             face=fc_hidden, fontsize=7)
    _arrow(ax, y, y - 0.04); y -= 0.04

    y = _box(ax, y, 0.10,
             r"Per-node readout (shared Linear, tanh)",
             face=fc_hidden, fontsize=7)
    _arrow(ax, y, y - 0.04); y -= 0.04

    _box(ax, y, 0.09,
         r"FC $\to$ sigmoid  (3-d multi-label)",
         face=fc_out, bold=True, fontsize=7.5)


def draw_path(ax):
    ax.set_title("PATH  (Howlader et al., 2026)", fontsize=9, pad=4)
    fc_input  = color_for("PATH") + "30"
    fc_hidden = color_for("PATH") + "55"
    fc_out    = OKABE_ITO["yellow"] + "60"

    y = 0.97
    y = _box(ax, y, 0.09,
             r"ssGSEA scores  (N pathways)",
             face=fc_input, bold=True)
    _arrow(ax, y, y - 0.03); y -= 0.03

    y = _box(ax, y, 0.10,
             r"Per-pathway projection  $(1 \to d)$" "\n"
             r"+  Laplacian PE  (top-$k$ eigvecs)",
             face=fc_input, fontsize=6.8)
    _arrow(ax, y, y - 0.03); y -= 0.03

    y = _box(ax, y, 0.18,
             r"$L{=}2 \times$ edge-aware Graph Transformer blocks" "\n"
             r"(soft mask, edge bias, $H{=}4$ heads," "\n"
             r"FFN$\times$4, BatchNorm)",
             face=fc_hidden, fontsize=6.8)
    _arrow(ax, y, y - 0.03); y -= 0.03

    y = _box(ax, y, 0.09,
             r"Attention-weighted readout  $g{=}\sum_p w_p\, x_p$",
             face=fc_hidden, fontsize=7)
    _arrow(ax, y, y - 0.03); y -= 0.03

    _box(ax, y, 0.10,
         r"BN $\to$ GELU $\to$ Drop $\to$ FC" "\n"
         r"(3-d multi-label)",
         face=fc_out, bold=True, fontsize=7)


# --------------------------------------------------------------------------
def main() -> None:
    apply_style()
    # Stack the panels into a tight ~3.3-inch-tall figure so the page does
    # not show dead space below the lowest BINN box.
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 3.3))
    for ax in axes:
        ax.set_xlim(0, 1.0)
        ax.set_ylim(0.30, 1.0)   # crop unused bottom half
        ax.axis("off")
    draw_binn(axes[0])
    draw_graphpath(axes[1])
    draw_path(axes[2])

    fig.suptitle(
        "Three pathway-informed architectures predicting therapy-response "
        r"phenotypes  (TMT / RT / OS $\geq$ 180 d)  from Reactome ssGSEA scores",
        fontsize=8.5, y=1.01,
    )
    fig.subplots_adjust(left=0.005, right=0.995, top=0.92, bottom=0.01,
                        wspace=0.18)
    out = Path(__file__).parent / "fig1_architectures.pdf"
    fig.savefig(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
