"""Figure 6 — training/validation loss curves per model.

Three panels (BINN / GraphPath / PATH) showing the train and validation
loss recorded by the training phase (stored inside each model's checkpoint).
A dashed vertical line marks the best-validation-loss epoch (the early-stop
restore point).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from _style import OKABE_ITO, apply_style, color_for

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PATHS = {
    "BINN":      PROJECT_ROOT / "binn"      / "artifacts" / "breast" / "binn.pt",
    "GraphPath": PROJECT_ROOT / "graphpath" / "artifacts" / "breast" / "graphpath.pt",
    "PATH":      PROJECT_ROOT / "path"      / "artifacts" / "breast" / "path.pt",
}


def _load(path: Path):
    if not path.exists():
        return None
    return torch.load(path, map_location="cpu", weights_only=False)


def _panel(ax, name, ckpt):
    color = color_for(name)
    if ckpt is None:
        ax.text(0.5, 0.5, f"{name} checkpoint missing", ha="center",
                va="center", transform=ax.transAxes,
                color=OKABE_ITO["vermillion"])
        ax.set_axis_off(); return

    h = ckpt["history"]
    train = np.asarray(h["train_loss"])
    val = np.asarray(h["val_loss"])
    epochs = np.arange(1, len(train) + 1)
    best_ep = int(np.argmin(val)) + 1
    best_val = float(np.min(val))

    ax.plot(epochs, train, color=color, linewidth=1.4, label="train")
    ax.plot(epochs, val, color=OKABE_ITO["vermillion"], linewidth=1.4,
            linestyle="--", label="val")
    ax.axvline(best_ep, color=OKABE_ITO["black"], linestyle=":", linewidth=0.8)
    ax.annotate(
        f"best epoch = {best_ep}\nval loss = {best_val:.4f}",
        xy=(best_ep, best_val),
        xytext=(0.55, 0.85), textcoords="axes fraction",
        fontsize=7,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                  edgecolor=color, linewidth=0.6),
        arrowprops=dict(arrowstyle="->", linewidth=0.6,
                        color=OKABE_ITO["black"]),
    )
    ax.set_title(f"{name}  ({len(train)} epochs)", fontsize=9)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("BCE loss")
    ax.set_axisbelow(True); ax.grid()
    ax.legend(loc="lower left", fontsize=7)


def main() -> None:
    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.4), sharey=False)
    for name, ax in zip(("BINN", "GraphPath", "PATH"), axes):
        _panel(ax, name, _load(PATHS[name]))
    fig.suptitle("Training and validation BCE loss per model "
                 "(dotted line = best-val epoch restored by early stopping)",
                 fontsize=8.5, y=1.02)
    fig.tight_layout()
    out = Path(__file__).parent / "fig6_training_curves.pdf"
    fig.savefig(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
