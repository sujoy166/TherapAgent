"""Figure 4 — therapy-response label distribution and per-head positive class weight.

Reads `binn/artifacts/splits.npz` if available; otherwise reproduces the
counts directly from `pathway_phenotype_mapping.csv` using stage bit-decoding.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _style import OKABE_ITO, apply_style

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SPLITS = PROJECT_ROOT / "binn" / "artifacts" / "splits.npz"
MAPPING = PROJECT_ROOT / "pathway_phenotype_mapping.csv"


def _load_counts() -> dict:
    if SPLITS.exists():
        blob = np.load(SPLITS, allow_pickle=True)
        Y, train_idx = blob["Y"], blob["train_idx"]
        pos_train = Y[train_idx].sum(axis=0)
        neg_train = len(train_idx) - pos_train
        pos_all = Y.sum(axis=0)
        neg_all = len(Y) - pos_all
    else:
        df = pd.read_csv(MAPPING, usecols=["sample", "stage"], index_col=0)
        s = df["stage"].astype(int)
        tmt, rt, osv = (s // 4) % 2, (s // 2) % 2, s % 2
        pos_all = np.array([tmt.sum(), rt.sum(), osv.sum()])
        neg_all = len(s) - pos_all
        pos_train = neg_train = None
    return {"pos_all": pos_all, "neg_all": neg_all,
            "pos_train": pos_train, "neg_train": neg_train}


def main() -> None:
    apply_style()
    counts = _load_counts()
    heads = ["TMT", "RT", r"OS $\geq$ 180 d"]
    pos = counts["pos_all"]; neg = counts["neg_all"]
    total = pos + neg

    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.6))

    # Left: stacked counts
    ax = axes[0]
    x = np.arange(len(heads))
    ax.bar(x, neg, color=OKABE_ITO["sky_blue"], edgecolor=OKABE_ITO["black"],
           linewidth=0.6, label="Negative")
    ax.bar(x, pos, bottom=neg, color=OKABE_ITO["vermillion"],
           edgecolor=OKABE_ITO["black"], linewidth=0.6, label="Positive")
    for i, (p, n, t) in enumerate(zip(pos, neg, total)):
        ax.text(i, n / 2, f"{n}\n({n/t:.0%})", ha="center", va="center",
                color="white", fontsize=8)
        ax.text(i, n + p / 2, f"{p}\n({p/t:.0%})", ha="center", va="center",
                color="white", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(heads)
    ax.set_ylabel("Number of samples")
    ax.set_title("Therapy-response label balance (full labeled cohort, $n{=}618$)")
    ax.legend(loc="lower right")
    ax.set_axisbelow(True); ax.grid(axis="y")

    # Right: positive-class weight (#neg/#pos)
    ax = axes[1]
    w = neg / np.maximum(pos, 1)
    bars = ax.bar(x, w, color=OKABE_ITO["bluish_green"],
                  edgecolor=OKABE_ITO["black"], linewidth=0.6)
    for b, v in zip(bars, w):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.1,
                f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(heads)
    ax.set_ylabel(r"$\mathrm{pos\_weight} = \#\mathrm{neg}/\#\mathrm{pos}$")
    ax.set_title("Per-head positive class weighting for BCE loss")
    ax.set_axisbelow(True); ax.grid(axis="y")

    fig.suptitle("Class balance and loss weighting for the multi-label setting",
                 fontsize=10, y=1.03)
    out = Path(__file__).parent / "fig4_label_imbalance.pdf"
    fig.savefig(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
