"""Figure 2 — per-head metric comparison across the three models.

Reads `{binn,graphpath,path}/artifacts/results.json` (Phase 5 outputs) and
plots grouped AUROC / AUPRC / F1 bars on the held-out test split.
Falls back to a clearly marked placeholder if any results.json is missing.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from _style import MODEL_COLORS, apply_style, OKABE_ITO

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS = {
    "BINN":      PROJECT_ROOT / "binn"      / "artifacts" / "results.json",
    "GraphPath": PROJECT_ROOT / "graphpath" / "artifacts" / "results.json",
    "PATH":      PROJECT_ROOT / "path"      / "artifacts" / "results.json",
}
HEADS = ("TMT", "RT", "OS")
METRICS = ("auroc", "auprc", "f1")
METRIC_LABELS = {"auroc": "AUROC", "auprc": "AUPRC", "f1": "F1"}


def _load(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text())


def main() -> None:
    apply_style()
    data = {name: _load(p) for name, p in MODELS.items()}
    missing = [n for n, d in data.items() if d is None]

    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.6), sharey=True)
    width = 0.25
    x = np.arange(len(HEADS))

    for ax, metric in zip(axes, METRICS):
        for i, (name, d) in enumerate(data.items()):
            if d is None:
                vals = np.array([np.nan] * 3)
            else:
                vals = np.array([d["test"][h][metric] for h in HEADS])
            ax.bar(x + (i - 1) * width, vals, width,
                   color=MODEL_COLORS[name], edgecolor=OKABE_ITO["black"],
                   linewidth=0.6, label=name)
        ax.set_xticks(x); ax.set_xticklabels(HEADS)
        ax.set_title(METRIC_LABELS[metric])
        ax.set_ylim(0, 1)
        ax.set_axisbelow(True); ax.grid(axis="y")
    axes[0].set_ylabel("Score (test split)")

    if missing:
        fig.text(0.5, -0.02,
                 f"Note: placeholder values for {', '.join(missing)} — "
                 "run the corresponding Phase 5 to populate.",
                 ha="center", fontsize=7, color=OKABE_ITO["vermillion"])

    axes[-1].legend(loc="upper right", bbox_to_anchor=(1.0, 1.02), ncol=1)
    fig.suptitle("Held-out test-set performance per therapy-response head",
                 fontsize=9, y=1.05)
    out = Path(__file__).parent / "fig2_metric_bars.pdf"
    fig.savefig(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
