"""Offline end-to-end smoke test for GraphPath.

Skips the live Reactome download by injecting a deterministic synthetic
adjacency. Verifies forward shapes, finite val loss, metrics in [0,1], and
a well-formed booktabs LaTeX table.

    python -m graphpath.tests.test_smoke
"""
from __future__ import annotations

import hashlib
import sys

import numpy as np
import pandas as pd
import torch

from binn.data import (
    fit_standardizer, list_pathway_columns, load_aligned, positive_weights,
    stratified_split,
)
from binn.evaluate import metrics, predict, print_report
from binn.reporting import write_table

from graphpath.config import Config
from graphpath.model import GraphPath
from graphpath.train import fit


def _h(s: str, mod: int) -> int:
    return int(hashlib.md5(s.encode()).hexdigest(), 16) % mod


def synthetic_adjacency(names, parents_per_node: int = 2) -> np.ndarray:
    """Deterministic sparse adjacency keyed off pathway-name hashes."""
    N = len(names)
    A = np.zeros((N, N), dtype=np.float32)
    for i, n in enumerate(names):
        for k in range(parents_per_node):
            j = _h(f"{n}::{k}", N)
            if j != i:
                A[i, j] = 1.0
                A[j, i] = 1.0
    return A


def main() -> int:
    cfg = Config()
    cfg.max_epochs = 4
    cfg.patience = 4
    cfg.batch_size = 16

    pathway_cols = list_pathway_columns(cfg.scores_csv)[:150]
    print(f"using {len(pathway_cols)} pathways")
    A = synthetic_adjacency(pathway_cols)
    print(f"adjacency edges (undirected): {int(A.sum()/2)}")

    X, Y, stage = load_aligned(cfg.scores_csv, cfg.mapping_csv, pathway_cols)
    train_idx, val_idx, test_idx = stratified_split(
        X, Y, stage, val_frac=cfg.val_frac, test_frac=cfg.test_frac, seed=cfg.seed,
    )
    mu, sd = fit_standardizer(X.values[train_idx])
    Xn = (X.values - mu) / sd
    pw = positive_weights(Y.values[train_idx])

    Xt = torch.from_numpy(Xn).float()
    Yt = torch.from_numpy(Y.values).float()

    def _dl(idx, shuffle):
        ds = torch.utils.data.TensorDataset(Xt[idx], Yt[idx])
        return torch.utils.data.DataLoader(
            ds, batch_size=cfg.batch_size, shuffle=shuffle,
            generator=torch.Generator().manual_seed(cfg.seed) if shuffle else None,
        )
    loaders = {
        "train": _dl(train_idx, True),
        "val":   _dl(val_idx, False),
        "test":  _dl(test_idx, False),
    }

    model = GraphPath(
        n_pathways=len(pathway_cols), adjacency=A,
        embed_dim=cfg.embed_dim, n_heads=cfg.n_heads,
        n_outputs=len(cfg.head_names), dropout=cfg.dropout,
    )
    print(f"params: {model.trainable_parameter_count():,}")

    xb, yb = next(iter(loaders["train"]))
    out = model(xb)
    assert out["prob"].shape == yb.shape
    assert torch.all((out["prob"] >= 0) & (out["prob"] <= 1))

    res = fit(model, loaders, pw, cfg, device="cpu", verbose=True)
    assert np.isfinite(res["best_val_loss"])

    val_p, val_y = predict(model, loaders["val"])
    test_p, test_y = predict(model, loaders["test"])
    val_m = metrics(val_p, val_y, cfg.head_names)
    test_m = metrics(test_p, test_y, cfg.head_names)
    print_report("VAL", val_m)
    print_report("TEST", test_m)

    for split in (val_m, test_m):
        for head in split.values():
            for k in ("auroc", "auprc", "f1", "accuracy"):
                v = head[k]
                assert np.isnan(v) or 0.0 <= v <= 1.0, f"{k}={v}"

    smoke_tex = cfg.artifacts_dir / "tex_smoke" / "graphpath_metrics.tex"
    rows = []
    for split_name, m in (("Val", val_m), ("Test", test_m)):
        for head in cfg.head_names:
            v = m[head]; cm = v["cm"]
            rows.append([head, split_name, v["auroc"], v["auprc"],
                         v["f1"], v["accuracy"],
                         cm["tn"], cm["fp"], cm["fn"], cm["tp"]])
    write_table(
        smoke_tex,
        headers=["Head", "Split", "AUROC", "AUPRC", "F1", "Acc",
                 "TN", "FP", "FN", "TP"],
        rows=rows,
        caption="Smoke-test GraphPath metrics (synthetic adjacency).",
        label="gp-smoke",
    )
    txt = smoke_tex.read_text()
    for marker in ("\\toprule", "\\midrule", "\\bottomrule", "tab:gp-smoke"):
        assert marker in txt
    print(f"  tex sanity OK ({smoke_tex})")

    print("\nGRAPHPATH SMOKE TEST OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
