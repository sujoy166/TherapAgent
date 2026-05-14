"""Figure 8 — cross-cohort summary.

Reads `{binn,graphpath,path}/artifacts/<cohort>/results.json` for every
cohort and draws three grouped bar panels (one per head: TMT / RT / OS)
showing test-set AUROC across the 5 cohorts × 3 models. This is the
single figure that summarises the full pan-cancer sweep.

Falls back to a clearly-marked placeholder if any cohort × model is
missing — the SLURM pipeline writes them in parallel, so the figure
will populate progressively as runs finish.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from _style import MODEL_COLORS, OKABE_ITO, apply_style

PROJECT_ROOT = Path(__file__).resolve().parents[2]
COHORTS = ("breast", "lung", "prostate", "head_neck", "thyroid")
COHORT_LABELS = {
    "breast": "Breast", "lung": "Lung", "prostate": "Prostate",
    "head_neck": "Head & Neck", "thyroid": "Thyroid",
}
MODELS = ("BINN", "GraphPath", "PATH")
MODEL_DIRS = {"BINN": "binn", "GraphPath": "graphpath", "PATH": "path"}
HEADS = ("TMT", "RT", "OS")


def _load(model: str, cohort: str):
    p = (PROJECT_ROOT / MODEL_DIRS[model] / "artifacts" / cohort
         / "results.json")
    if not p.exists():
        return None
    return json.loads(p.read_text())


def main() -> None:
    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.6), sharey=True)

    # Pull AUROC per (head, cohort, model). NaN where missing.
    auroc = np.full((len(HEADS), len(COHORTS), len(MODELS)), np.nan)
    for ci, c in enumerate(COHORTS):
        for mi, m in enumerate(MODELS):
            d = _load(m, c)
            if d is None:
                continue
            for hi, h in enumerate(HEADS):
                auroc[hi, ci, mi] = d["test"][h]["auroc"]

    missing = []
    for ci, c in enumerate(COHORTS):
        for mi, m in enumerate(MODELS):
            if np.isnan(auroc[:, ci, mi]).all():
                missing.append(f"{m}×{c}")

    width = 0.26
    x = np.arange(len(COHORTS))
    for hi, head in enumerate(HEADS):
        ax = axes[hi]
        for mi, model in enumerate(MODELS):
            ys = auroc[hi, :, mi]
            ax.bar(x + (mi - 1) * width, ys, width,
                   color=MODEL_COLORS[model],
                   edgecolor=OKABE_ITO["black"], linewidth=0.5,
                   label=model)
        ax.axhline(0.5, color=OKABE_ITO["black"], linestyle=":",
                   linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels([COHORT_LABELS[c] for c in COHORTS],
                           rotation=30, ha="right", fontsize=7)
        ax.set_ylim(0, 1)
        ax.set_title(head, fontsize=9)
        ax.set_axisbelow(True); ax.grid(axis="y")
    axes[0].set_ylabel("Test AUROC")
    axes[-1].legend(loc="upper right", bbox_to_anchor=(1.0, 1.02), ncol=1,
                    fontsize=7)

    title = ("Cross-cohort test-set AUROC across the three pathway-informed "
             "architectures. Dotted line = chance.")
    if missing:
        title += f"\nPending: {', '.join(missing[:6])}{'…' if len(missing) > 6 else ''}"
    fig.suptitle(title, fontsize=8.5, y=1.05)
    fig.tight_layout()
    out = Path(__file__).parent / "fig8_cross_cohort.pdf"
    fig.savefig(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
