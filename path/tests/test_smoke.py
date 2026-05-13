"""Offline end-to-end smoke test for the PATH graph transformer.

Skips Reactome download by generating a deterministic weighted adjacency.
Verifies forward shapes, finite val loss, metrics in [0,1], and a well-formed
booktabs LaTeX table.

    python -m path.tests.test_smoke
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

from path.config import Config
from path.model import PathGraphTransformer
from path.train import fit


def _h(s: str, mod: int) -> int:
    return int(hashlib.md5(s.encode()).hexdigest(), 16) % mod


def synthetic_weighted_adjacency(names, neighbors: int = 4) -> np.ndarray:
    """Deterministic weighted adjacency in [0,1], symmetrised."""
    rng = np.random.default_rng(42)
    N = len(names)
    A = np.zeros((N, N), dtype=np.float32)
    for i, n in enumerate(names):
        for k in range(neighbors):
            j = _h(f"{n}::{k}", N)
            if j != i:
                w = float(rng.uniform(0.1, 1.0))
                A[i, j] = max(A[i, j], w)
                A[j, i] = max(A[j, i], w)
    A = A / (A.max() + 1e-8)
    np.fill_diagonal(A, 0.0)
    return A


def main() -> int:
    cfg = Config()
    cfg.max_epochs = 4
    cfg.patience = 4
    cfg.min_epochs = 1
    cfg.batch_size = 16
    cfg.laplacian_k = 8

    pathway_cols = list_pathway_columns(cfg.scores_csv)[:120]
    print(f"using {len(pathway_cols)} pathways")
    A = synthetic_weighted_adjacency(pathway_cols)
    print(f"adjacency edges (undirected): {int((A > 0).sum()/2)}")

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

    model = PathGraphTransformer(
        n_pathways=len(pathway_cols), adjacency=A,
        embed_dim=cfg.embed_dim, n_heads=cfg.n_heads, n_layers=cfg.n_layers,
        n_outputs=len(cfg.head_names), laplacian_k=cfg.laplacian_k,
        soft_mask_penalty=cfg.soft_mask_penalty,
        ffn_expansion=cfg.ffn_expansion, dropout=cfg.dropout,
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
                assert np.isnan(v) or 0.0 <= v <= 1.0

    smoke_tex = cfg.artifacts_dir / "tex_smoke" / "path_metrics.tex"
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
        caption="Smoke-test PATH metrics (synthetic adjacency).",
        label="path-smoke",
    )
    txt = smoke_tex.read_text()
    for marker in ("\\toprule", "\\midrule", "\\bottomrule", "tab:path-smoke"):
        assert marker in txt
    print(f"  tex sanity OK ({smoke_tex})")

    print("\nPATH SMOKE TEST OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
