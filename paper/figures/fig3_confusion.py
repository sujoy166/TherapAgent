"""Figure 3 — confusion-matrix counts at threshold 0.5 across models × heads.

Uses a viridis colormap (perceptually uniform, colorblind-safe) and annotates
each cell with the integer count. One row per model, one column per head.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from _style import apply_style, OKABE_ITO

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS = {
    "BINN":      PROJECT_ROOT / "binn"      / "artifacts" / "results.json",
    "GraphPath": PROJECT_ROOT / "graphpath" / "artifacts" / "results.json",
    "PATH":      PROJECT_ROOT / "path"      / "artifacts" / "results.json",
}
HEADS = ("TMT", "RT", "OS")


def _cm(d, head):
    cm = d["test"][head]["cm"]
    return np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])


def _load(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text())


def main() -> None:
    apply_style()
    data = {name: _load(p) for name, p in MODELS.items()}
    fig, axes = plt.subplots(3, 3, figsize=(6.9, 6.5))
    missing = []
    for r, name in enumerate(MODELS):
        for c, head in enumerate(HEADS):
            ax = axes[r, c]
            d = data[name]
            if d is None:
                missing.append(name)
                ax.text(0.5, 0.5, "no results.json", ha="center", va="center",
                        color=OKABE_ITO["vermillion"], fontsize=8,
                        transform=ax.transAxes)
                ax.axis("off")
                continue
            cm = _cm(d, head)
            im = ax.imshow(cm, cmap="viridis", vmin=0, vmax=cm.max())
            for i in range(2):
                for j in range(2):
                    val = cm[i, j]
                    txt_color = "white" if val < cm.max() / 2 else "black"
                    ax.text(j, i, str(val), ha="center", va="center",
                            color=txt_color, fontsize=10, fontweight="bold")
            ax.set_xticks([0, 1]); ax.set_xticklabels(["Pred 0", "Pred 1"])
            ax.set_yticks([0, 1]); ax.set_yticklabels(["True 0", "True 1"])
            ax.set_title(f"{name} — {head}", fontsize=9)
            for spine in ax.spines.values():
                spine.set_visible(True); spine.set_linewidth(0.4)

    fig.suptitle(
        "Confusion matrices at the 0.5 probability threshold (test split)",
        fontsize=10, y=1.02,
    )
    if missing:
        fig.text(0.5, -0.01,
                 f"Run Phase 5 for: {', '.join(sorted(set(missing)))}",
                 ha="center", fontsize=7, color=OKABE_ITO["vermillion"])
    out = Path(__file__).parent / "fig3_confusion.pdf"
    fig.savefig(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
