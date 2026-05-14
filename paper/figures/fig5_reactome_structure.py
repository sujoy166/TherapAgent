"""Figure 5 — Reactome structure consumed by each model.

Three panels (BINN / GraphPath / PATH) showing:
  BINN      : layer node counts (1,686 → 656 → 306 → 163) with edges-to-next
              layer overlaid as count annotations.
  GraphPath : log-binned adjacency-degree histogram (parent ∪ sibling edges).
  PATH      : log-binned adjacency-degree histogram + edge-weight CDF
              (Jaccard adjacency).

Reads `*/artifacts/reactome.pkl` produced by Phase 2. Falls back to a
clearly marked placeholder if any checkpoint is missing.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from _style import OKABE_ITO, apply_style, color_for

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PATHS = {
    "BINN":      PROJECT_ROOT / "binn"      / "artifacts" / "breast" / "reactome.pkl",
    "GraphPath": PROJECT_ROOT / "graphpath" / "artifacts" / "breast" / "reactome.pkl",
    "PATH":      PROJECT_ROOT / "path"      / "artifacts" / "breast" / "reactome.pkl",
}


def _load(path: Path):
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def panel_binn(ax, r):
    color = color_for("BINN")
    if r is None:
        ax.text(0.5, 0.5, "binn/artifacts/breast/reactome.pkl missing",
                ha="center", va="center", transform=ax.transAxes,
                color=OKABE_ITO["vermillion"])
        ax.set_axis_off(); return

    sizes = [len(layer) for layer in r["layers"]]
    edges = [int(m.sum()) for m in r["masks"]]
    xs = np.arange(len(sizes))
    bars = ax.bar(xs, sizes, color=color, edgecolor=OKABE_ITO["black"],
                  linewidth=0.6)
    for b, s in zip(bars, sizes):
        ax.text(b.get_x() + b.get_width() / 2, s + max(sizes) * 0.02,
                f"{s:,}", ha="center", va="bottom", fontsize=8,
                weight="bold")
    # annotate downstream edge counts as arrow labels between bars
    for i, e in enumerate(edges):
        ax.annotate(f"{e:,} edges", xy=(i + 0.5, max(sizes) * 0.55),
                    ha="center", fontsize=7, color=OKABE_ITO["black"],
                    rotation=0)
        ax.annotate("", xy=(i + 0.85, max(sizes) * 0.50),
                    xytext=(i + 0.15, max(sizes) * 0.50),
                    arrowprops=dict(arrowstyle="->", linewidth=0.8,
                                    color=OKABE_ITO["black"]))
    ax.set_xticks(xs)
    ax.set_xticklabels([f"L{i}\n(layer {i})" for i in range(len(sizes))],
                       fontsize=8)
    ax.set_ylabel("Pathway nodes per layer")
    ax.set_title("BINN — Reactome parent/child hierarchy", fontsize=9)
    ax.set_ylim(0, max(sizes) * 1.20)
    ax.set_axisbelow(True); ax.grid(axis="y")


def panel_graphpath(ax, r):
    color = color_for("GraphPath")
    if r is None:
        ax.text(0.5, 0.5, "graphpath/artifacts/breast/reactome.pkl missing",
                ha="center", va="center", transform=ax.transAxes,
                color=OKABE_ITO["vermillion"])
        ax.set_axis_off(); return

    A = r["adjacency"]
    deg = (A > 0).sum(axis=1)
    nodes = len(deg)
    edges = int((A > 0).sum() / 2)
    ax.hist(deg, bins=np.arange(0, deg.max() + 2) - 0.5, color=color,
            edgecolor=OKABE_ITO["black"], linewidth=0.5)
    ax.axvline(deg.mean(), color=OKABE_ITO["vermillion"], linestyle="--",
               linewidth=1.0, label=f"mean = {deg.mean():.2f}")
    ax.set_xlabel("Node degree")
    ax.set_ylabel("Number of pathways")
    ax.set_title("GraphPath — parent / sibling adjacency", fontsize=9)
    ax.set_axisbelow(True); ax.grid(axis="y")
    ax.legend(loc="upper right")
    ax.text(0.98, 0.78, f"{nodes:,} nodes\n{edges:,} edges",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=8, bbox=dict(boxstyle="round,pad=0.25",
                                   facecolor="white", edgecolor=color,
                                   linewidth=0.6))


def panel_path(ax, r):
    color = color_for("PATH")
    if r is None:
        ax.text(0.5, 0.5, "path/artifacts/breast/reactome.pkl missing",
                ha="center", va="center", transform=ax.transAxes,
                color=OKABE_ITO["vermillion"])
        ax.set_axis_off(); return

    A = r["adjacency"]
    deg = (A > 0).sum(axis=1)
    nodes = len(deg)
    edges = int((A > 0).sum() / 2)

    ax.hist(deg, bins=40, color=color, edgecolor=OKABE_ITO["black"],
            linewidth=0.5)
    ax.axvline(deg.mean(), color=OKABE_ITO["vermillion"], linestyle="--",
               linewidth=1.0, label=f"mean = {deg.mean():.1f}")
    ax.set_xlabel("Node degree (Jaccard $>$ 0)")
    ax.set_ylabel("Number of pathways")
    ax.set_title("PATH — Jaccard gene-set adjacency", fontsize=9)
    ax.set_axisbelow(True); ax.grid(axis="y")
    ax.legend(loc="upper right")
    ax.text(0.98, 0.65, f"{nodes:,} nodes\n{edges:,} edges",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=8, bbox=dict(boxstyle="round,pad=0.25",
                                   facecolor="white", edgecolor=color,
                                   linewidth=0.6))


def main() -> None:
    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.5))
    panel_binn(axes[0],      _load(PATHS["BINN"]))
    panel_graphpath(axes[1], _load(PATHS["GraphPath"]))
    panel_path(axes[2],      _load(PATHS["PATH"]))
    fig.suptitle(
        "Reactome structure ingested by each model "
        "(Phase 2 output, 1,686 of 1,706 input pathways matched)",
        fontsize=8.5, y=1.02,
    )
    fig.tight_layout()
    out = Path(__file__).parent / "fig5_reactome_structure.pdf"
    fig.savefig(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
