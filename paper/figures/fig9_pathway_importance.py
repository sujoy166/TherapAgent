"""Figure 9 — top pathways per head per model.

Reads ``{binn, graphpath, path}/artifacts/<cohort>/importance.npz`` and
draws one panel per head. Each panel shows the union top-10 pathways
(ranked by max importance across models) with three horizontal bars per
pathway — one per model. This visualises *agreement* and *disagreement*
between the three architectures about which Reactome pathways drive
each therapy-response head.

Defaults to the breast cohort because that is the manuscript's headline
cohort. Pass a different `COHORT` env var to retarget.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np

from _style import MODEL_COLORS, OKABE_ITO, apply_style

PROJECT_ROOT = Path(__file__).resolve().parents[2]
COHORT = os.environ.get("COHORT", "breast")
MODELS = ("BINN", "GraphPath", "PATH")
MODEL_DIRS = {"BINN": "binn", "GraphPath": "graphpath", "PATH": "path"}
HEADS = ("TMT", "RT", "OS")
TOP_K = 10


def _short(name: str, n: int = 42) -> str:
    return name if len(name) <= n else name[: n - 1] + "…"


def _load_one(model: str):
    p = PROJECT_ROOT / MODEL_DIRS[model] / "artifacts" / COHORT / "importance.npz"
    if not p.exists():
        return None, None
    d = np.load(p, allow_pickle=True)
    names = [str(x) for x in d["pathway_names"]]
    per_head = {h: d[h] for h in HEADS if h in d.files}
    return names, per_head


def _normalise(v: np.ndarray) -> np.ndarray:
    """Per-model min-max [0, 1] so bars are comparable on a single axis."""
    if v.max() == 0:
        return v
    return v / v.max()


def main() -> None:
    apply_style()
    loaded = {m: _load_one(m) for m in MODELS}
    # If no model has been trained on this cohort yet, emit a placeholder.
    if all(v[0] is None for v in loaded.values()):
        fig, ax = plt.subplots(figsize=(7.1, 2.4))
        ax.text(0.5, 0.5,
                f"importance.npz missing for cohort '{COHORT}'.\n"
                "Run the train + evaluate phases first.",
                ha="center", va="center", transform=ax.transAxes,
                color=OKABE_ITO["vermillion"])
        ax.axis("off")
        fig.savefig(Path(__file__).parent / "fig9_pathway_importance.pdf")
        return

    # Pull a single shared pathway-name vector. All three models train on
    # the same Reactome-matched input pathway list, so the names align by
    # index (we verify by length match).
    names_ref = next(v[0] for v in loaded.values() if v[0] is not None)
    P = len(names_ref)

    fig, axes = plt.subplots(3, 1, figsize=(3.45, 7.0), sharex=True)
    for ax, head in zip(axes, HEADS):
        # Build a (P, n_models) matrix of normalised importance.
        mat = np.full((P, len(MODELS)), np.nan)
        for mi, m in enumerate(MODELS):
            names, per_head = loaded[m]
            if names is None or len(names) != P or head not in per_head:
                continue
            mat[:, mi] = _normalise(per_head[head])

        # Union top-TOP_K by max across available models
        max_per_pw = np.nanmax(mat, axis=1)
        order = np.argsort(np.where(np.isfinite(max_per_pw),
                                     max_per_pw, -np.inf))[::-1][:TOP_K]

        y = np.arange(len(order))[::-1]   # top one at the top of the axis
        h = 0.26                           # bar height per model
        for mi, m in enumerate(MODELS):
            offset = (mi - 1) * h
            ax.barh(y + offset, np.nan_to_num(mat[order, mi]), height=h,
                    color=MODEL_COLORS[m],
                    edgecolor=OKABE_ITO["black"], linewidth=0.4,
                    label=m if head == HEADS[0] else None)
        ax.set_yticks(y)
        ax.set_yticklabels([_short(names_ref[i]) for i in order], fontsize=6.5)
        ax.set_xlim(0, 1.05)
        ax.set_xlabel("Normalised importance (per model)")
        ax.set_title(head, fontsize=9)
        ax.set_axisbelow(True); ax.grid(axis="x")

    axes[0].legend(loc="lower right", fontsize=7)
    fig.suptitle(
        f"Top-{TOP_K} pathways per head, ranked by union max importance "
        f"across the three models  (cohort: {COHORT}).",
        fontsize=8.5, y=1.01,
    )
    fig.tight_layout()
    out = Path(__file__).parent / "fig9_pathway_importance.pdf"
    fig.savefig(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
