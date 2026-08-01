"""Regenerate the two metric figures (fig1 = grouped AUROC bars with 95% CI,
fig2 = cross-cohort AUROC heatmap) from the unified cross-validation summary
(results_cv/summary.json). Keeps them consistent with Tables 2-3.

Outputs paper/fig1.png and paper/fig2.png.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
try:
    from _style import MODEL_COLORS, apply_style
    apply_style()
except Exception:
    MODEL_COLORS = {"BINN": "#0072B2", "GraphPath": "#E69F00", "PATH": "#009E73"}

PAPER = os.path.dirname(HERE)
REPO = os.path.dirname(PAPER)
SUMMARY = os.path.join(REPO, "results_cv", "summary.json")

MODELS = ["binn", "graphpath", "path"]
MODEL_LABEL = {"binn": "BINN", "graphpath": "GraphPath", "path": "PATH"}
COHORTS = ["breast", "lung", "prostate", "head_neck", "thyroid"]
COHORT_LABEL = {"breast": "Breast", "lung": "Lung", "prostate": "Prostate",
                "head_neck": "Head & Neck", "thyroid": "Thyroid"}
HEADS = ["TMT", "RT", "OS"]

S = json.load(open(SUMMARY))


def cell(model, cohort, head, met="auroc"):
    d = S.get(f"{model}/{cohort}/{head}/{met}", {})
    m = d.get("mean", np.nan)
    lo = d.get("ci_lo", np.nan)
    hi = d.get("ci_hi", np.nan)
    return m, lo, hi


# ---------------------------------------------------------------- fig1: bars
def make_bars():
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), sharey=True)
    x = np.arange(len(COHORTS))
    w = 0.26
    for ax, head in zip(axes, HEADS):
        for i, model in enumerate(MODELS):
            means, los, his = [], [], []
            for c in COHORTS:
                m, lo, hi = cell(model, c, head)
                means.append(m)
                los.append(0 if np.isnan(lo) else max(0, m - lo))
                his.append(0 if np.isnan(hi) else max(0, hi - m))
            ax.bar(x + (i - 1) * w, means, w,
                   yerr=[los, his], capsize=2,
                   color=MODEL_COLORS[MODEL_LABEL[model]],
                   label=MODEL_LABEL[model], error_kw={"lw": 0.8})
        ax.axhline(0.5, ls=":", c="0.4", lw=1)
        ax.set_title(head)
        ax.set_xticks(x)
        ax.set_xticklabels([COHORT_LABEL[c] for c in COHORTS],
                           rotation=35, ha="right", fontsize=7)
        ax.set_ylim(0, 1.0)
    axes[0].set_ylabel("Test AUROC (mean $\\pm$ 95% CI)")
    axes[-1].legend(fontsize=7, frameon=False, loc="upper right")
    fig.tight_layout()
    out = os.path.join(PAPER, "fig1.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print("wrote", out)


# ------------------------------------------------------------ fig2: heatmap
def make_heat():
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.0))
    fig.subplots_adjust(wspace=0.35)
    for j, (ax, head) in enumerate(zip(axes, HEADS)):
        M = np.array([[cell(m, c, head)[0] for m in MODELS] for c in COHORTS])
        im = ax.imshow(M, cmap="viridis", vmin=0.3, vmax=0.9, aspect="auto")
        ax.set_xticks(range(len(MODELS)))
        ax.set_xticklabels([MODEL_LABEL[m] for m in MODELS],
                           rotation=30, ha="right", fontsize=7)
        ax.set_yticks(range(len(COHORTS)))
        if j == 0:
            ax.set_yticklabels([COHORT_LABEL[c] for c in COHORTS], fontsize=7)
        else:
            ax.set_yticklabels([])
        ax.set_title(head)
        for r in range(len(COHORTS)):
            for cc in range(len(MODELS)):
                v = M[r, cc]
                if not np.isnan(v):
                    ax.text(cc, r, f"{v:.2f}", ha="center", va="center",
                            color="w" if v < 0.62 else "k", fontsize=6.5)
    fig.colorbar(im, ax=axes, shrink=0.8, label="Mean test AUROC")
    out = os.path.join(PAPER, "fig2.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    make_bars()
    make_heat()
