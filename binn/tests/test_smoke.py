"""End-to-end smoke test using a deterministic *synthetic* Reactome fixture.

This avoids the live download in `reactome.py` so the architecture, training
loop, loss, and evaluator can be verified offline. Run it with:

    python -m binn.tests.test_smoke

It exits 0 when the full pipeline runs and all per-head metrics fall in [0, 1].
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from binn.config import Config
from binn.data import (
    list_pathway_columns, load_aligned, make_loaders, positive_weights,
)
from binn.evaluate import metrics, predict, print_report
from binn.main import set_seed
from binn.model import BINN
from binn.reactome import build_layers, build_masks


def _hash_int(s: str, mod: int) -> int:
    return int(hashlib.md5(s.encode()).hexdigest(), 16) % mod


def synthetic_hierarchy(pathway_names, n_hidden_layers: int = 3, fanout: int = 4):
    """Deterministic 4-children-per-parent mock hierarchy keyed off pathway names."""
    input_ids = [f"P0::{n}" for n in pathway_names]
    parent_map = {}
    cur = list(input_ids)
    for depth in range(1, n_hidden_layers):
        n_parents = max(1, len(cur) // fanout)
        next_ids = [f"P{depth}::g{i}" for i in range(n_parents)]
        for c in cur:
            parent_map[c] = {next_ids[_hash_int(c, n_parents)]}
        cur = next_ids

    layers = build_layers(input_ids, parent_map, n_layers=n_hidden_layers)
    masks = build_masks(layers, parent_map)
    return input_ids, parent_map, layers, masks


def main() -> int:
    cfg = Config()
    cfg.max_epochs = 6
    cfg.patience = 6
    cfg.n_hidden_layers = 3
    set_seed(cfg.seed)

    pathway_cols = list_pathway_columns(cfg.scores_csv)
    # Sub-sample for speed: 200 pathways is plenty to exercise the pipeline.
    pathway_cols = pathway_cols[:200]
    print(f"using {len(pathway_cols)} pathways for smoke test")

    input_ids, parent_map, layers, masks = synthetic_hierarchy(
        pathway_cols, n_hidden_layers=cfg.n_hidden_layers
    )
    for i, layer in enumerate(layers):
        print(f"  layer {i}: {len(layer)} nodes")
    for i, m in enumerate(masks):
        print(f"  mask  {i}->{i+1}: {m.shape}  edges={int(m.sum())}")

    X, Y, stage = load_aligned(cfg.scores_csv, cfg.mapping_csv, pathway_cols)

    loaders = make_loaders(
        X, Y, stage, batch_size=cfg.batch_size,
        val_frac=cfg.val_frac, test_frac=cfg.test_frac, seed=cfg.seed,
    )
    pw = positive_weights(Y.values[loaders["splits"]["train"]])
    print("pos weights:", dict(zip(cfg.head_names, pw.tolist())))

    model = BINN(masks=masks, n_heads=len(cfg.head_names), dropout=cfg.dropout)
    print("layer sizes:", model.layer_sizes)
    print("active params:", model.trainable_parameter_count())

    # One forward to make sure shapes/dtype line up before the training loop.
    xb, yb = next(iter(loaders["train"]))
    out = model(xb)
    assert out["prob"].shape == yb.shape, (out["prob"].shape, yb.shape)
    assert torch.all((out["prob"] >= 0) & (out["prob"] <= 1)), "probs out of range"

    from binn.train import fit
    res = fit(model, loaders, pw, cfg, device="cpu", verbose=True)
    assert np.isfinite(res["best_val_loss"]), "val loss became non-finite"

    val_p, val_y = predict(model, loaders["val"])
    test_p, test_y = predict(model, loaders["test"])
    val_m = metrics(val_p, val_y, cfg.head_names)
    test_m = metrics(test_p, test_y, cfg.head_names)
    print_report("VAL", val_m)
    print_report("TEST", test_m)

    for split in (val_m, test_m):
        for head_metrics in split.values():
            for k in ("auroc", "auprc", "f1", "accuracy"):
                v = head_metrics[k]
                assert np.isnan(v) or 0.0 <= v <= 1.0, f"{k}={v} out of [0,1]"

    # Exercise the LaTeX table writers end-to-end and validate their output.
    from binn.reporting import write_table
    tex_dir = cfg.artifacts_dir / "tex_smoke"
    rows = []
    for split_name, m in (("Val", val_m), ("Test", test_m)):
        for head in cfg.head_names:
            v = m[head]
            cm = v["cm"]
            rows.append([head, split_name, v["auroc"], v["auprc"],
                         v["f1"], v["accuracy"],
                         cm["tn"], cm["fp"], cm["fn"], cm["tp"]])
    tex_path = write_table(
        tex_dir / "metrics_smoke.tex",
        headers=["Head", "Split", "AUROC", "AUPRC", "F1", "Acc",
                 "TN", "FP", "FN", "TP"],
        rows=rows,
        caption="Smoke-test BINN metrics (synthetic hierarchy).",
        label="binn-smoke",
    )
    txt = tex_path.read_text()
    for marker in ("\\toprule", "\\midrule", "\\bottomrule", "tab:binn-smoke"):
        assert marker in txt, f"missing {marker} in generated tex"
    print(f"  tex sanity OK ({tex_path})")

    print("\nSMOKE TEST OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
