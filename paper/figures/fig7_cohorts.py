"""Figure 7 — cross-cohort label distribution.

For each of the four supported TCGA cohorts (breast, prostate, head_neck,
thyroid), the figure reports:

  • sample count, and
  • TMT / RT / OS≥180d positive prevalence

decoded from the `stage` bitfield of the per-cohort
`Final DataSet/<Cohort>_final.csv`. The two cohorts disabled in the
COHORT_FILES registry (lung, bladder) are explicitly *not* plotted; an
annotation in the lower-right explains why.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _style import OKABE_ITO, apply_style

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FINAL_DIR = PROJECT_ROOT / "Final DataSet"
COHORTS = [
    ("breast",    "Breast_Cancer_final.csv",    "Breast"),
    ("prostate",  "Prostate_cancer_final.csv",  "Prostate"),
    ("head_neck", "Head_Neck_Cancer_Final.csv", "Head & Neck"),
    ("thyroid",   "Thyroid_Cancer_Final.csv",   "Thyroid"),
    ("lung",      "Lung_cancer_final.csv",      "Lung"),
]

HEADS = ("TMT", "RT", r"OS $\geq$ 180 d")
HEAD_COLORS = (OKABE_ITO["blue"], OKABE_ITO["orange"],
               OKABE_ITO["bluish_green"])


def _decode_heads(stage: pd.Series) -> dict:
    s = stage.astype(int)
    return {
        "TMT": ((s // 4) % 2).values,
        "RT":  ((s // 2) % 2).values,
        "OS":  (s % 2).values,
    }


def _panel(ax, cohort_key, csv_name, pretty_name):
    csv = FINAL_DIR / csv_name
    if not csv.exists():
        ax.text(0.5, 0.5, f"{csv.name}\nmissing", ha="center", va="center",
                transform=ax.transAxes, color=OKABE_ITO["vermillion"])
        ax.set_axis_off(); return

    df = pd.read_csv(csv, usecols=["sample", "stage"])
    df = df.dropna(subset=["stage"])
    n = len(df)
    heads = _decode_heads(df["stage"])
    pos = np.array([heads["TMT"].sum(), heads["RT"].sum(), heads["OS"].sum()])
    prev = pos / max(n, 1)

    xs = np.arange(3)
    bars = ax.bar(xs, prev, color=HEAD_COLORS,
                  edgecolor=OKABE_ITO["black"], linewidth=0.6)
    for b, p, count in zip(bars, prev, pos):
        ax.text(b.get_x() + b.get_width() / 2, p + 0.025,
                f"{p:.0%}\n({count}/{n})", ha="center", va="bottom",
                fontsize=7)
    ax.set_xticks(xs); ax.set_xticklabels(HEADS, fontsize=8)
    ax.set_ylim(0, 1.15)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=7)
    ax.set_title(rf"{pretty_name}  ($n{{=}}${n})", fontsize=9)
    ax.set_axisbelow(True); ax.grid(axis="y")


def main() -> None:
    apply_style()
    fig, axes = plt.subplots(1, 5, figsize=(7.1, 2.3), sharey=True)
    for ax, (k, csv, pretty) in zip(axes, COHORTS):
        _panel(ax, k, csv, pretty)
    axes[0].set_ylabel("Positive prevalence")

    fig.suptitle(
        "Per-cohort positive prevalence of the three therapy-response heads. "
        "Bladder ($n{=}21$) is present in the data but excluded from training "
        "because its sample count is too low for a stratified split.",
        fontsize=8.5, y=1.04,
    )
    fig.tight_layout()
    out = Path(__file__).parent / "fig7_cohorts.pdf"
    fig.savefig(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
