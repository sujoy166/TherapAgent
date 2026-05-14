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
         bold=False, italic=False, subline=None, subline_color=None,
         subline_fontsize=6.2):
    """Draw a rounded box. If `subline` is given, the box renders two
    stacked text lines: the main `text` (centred in the upper portion)
    and the smaller italic `subline` (centred in the lower portion).
    This is how each BINN layer announces its auxiliary classifier head."""
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
    if subline is None:
        ax.text(x + w / 2, y_top - height / 2, text,
                ha="center", va="center", fontsize=fontsize,
                weight=weight, style=style, wrap=True)
    else:
        # Reserve upper 60% for main text, lower 40% for subline.
        ax.text(x + w / 2, y_top - height * 0.32, text,
                ha="center", va="center", fontsize=fontsize,
                weight=weight, style=style, wrap=True)
        ax.text(x + w / 2, y_top - height * 0.78, subline,
                ha="center", va="center", fontsize=subline_fontsize,
                style="italic",
                color=subline_color or OKABE_ITO["black"])
    return y_top - height


def _arrow(ax, y_from, y_to, x=0.5, color=None):
    color = color or OKABE_ITO["black"]
    arrow = FancyArrowPatch(
        (x, y_from), (x, y_to),
        arrowstyle="-|>", mutation_scale=8,
        linewidth=0.8, color=color,
    )
    ax.add_patch(arrow)


# --------------------------------------------------------------------------
# Per-model panels
# --------------------------------------------------------------------------
def draw_binn(ax):
    ax.set_title("BINN  (Hartman et al., 2023)", fontsize=9, pad=4)
    color = color_for("BINN")
    fc_input  = color + "30"
    fc_hidden = color + "55"
    fc_out    = OKABE_ITO["yellow"] + "60"

    y = 0.97
    box_h = 0.12   # taller boxes so the subline fits without overlap
    gap = 0.03
    # Input row
    y = _box(ax, y, box_h,
             r"ssGSEA scores  (N pathways)",
             face=fc_input, bold=True,
             subline=r"$\to$ classifier head$_0$",
             subline_color=color)

    block_labels = [
        r"Sparse Linear  (Reactome mask, $L_1$) $\to$ tanh $\to$ BN $\to$ Drop",
        r"Sparse Linear  (Reactome parents, $L_2$) $\to$ tanh $\to$ BN $\to$ Drop",
        r"Sparse Linear  (Reactome parents, $L_3$) $\to$ tanh $\to$ BN $\to$ Drop",
    ]
    for i, label in enumerate(block_labels, start=1):
        _arrow(ax, y, y - gap)
        y -= gap
        y = _box(ax, y, box_h, label, face=fc_hidden, fontsize=6.6,
                 subline=rf"$\to$ classifier head$_{i}$",
                 subline_color=color)

    _arrow(ax, y, y - gap - 0.005)
    y -= gap + 0.005
    _box(ax, y, 0.10,
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
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 3.6))
    for ax in axes:
        ax.set_xlim(0, 1.0)
        ax.set_ylim(0.18, 1.0)   # extra vertical room for BINN's taller boxes
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
                        wspace=0.28)
    out = Path(__file__).parent / "fig1_architectures.pdf"
    fig.savefig(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
