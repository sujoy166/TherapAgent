"""Per-head metrics: AUROC, AUPRC, accuracy, F1, confusion matrix.

`find_best_threshold` returns the F1-maximising decision threshold per head
(paper PATH §4.3, Eq. 27). `metrics(..., thresholds=...)` applies a per-head
threshold when computing F1, accuracy, and the confusion matrix; AUROC and
AUPRC remain threshold-free.
"""
from __future__ import annotations

from typing import Dict, Mapping, Optional

import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score, confusion_matrix, f1_score,
    precision_recall_curve, roc_auc_score,
)


@torch.no_grad()
def predict(model, loader, device: str = "cpu"):
    model.eval()
    probs, trues = [], []
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        out = model(xb)
        probs.append(out["prob"].cpu().numpy())
        trues.append(yb.numpy())
    return np.concatenate(probs), np.concatenate(trues)


def find_best_threshold(prob: np.ndarray, y: np.ndarray, head_names
                        ) -> Dict[str, float]:
    """Per-head F1-maximising threshold (paper PATH §4.3, Eq. 27).

    ``τ* = argmax_τ 2 · P(τ) · R(τ) / (P(τ) + R(τ) + ε)``

    Heads that are wholly positive or wholly negative on the calibration
    fold fall back to the default 0.5 because no precision/recall curve
    exists there.
    """
    out: Dict[str, float] = {}
    eps = 1e-12
    for h, name in enumerate(head_names):
        p = prob[:, h]
        t = y[:, h].astype(int)
        if t.sum() == 0 or t.sum() == len(t):
            out[name] = 0.5
            continue
        precs, recs, ts = precision_recall_curve(t, p)
        # precision_recall_curve returns len(ts) == len(precs) - 1.
        f1s = 2 * precs * recs / (precs + recs + eps)
        # Drop the last entry which has no threshold associated with it.
        f1s = f1s[:-1] if len(ts) == len(precs) - 1 else f1s
        if len(f1s) == 0:
            out[name] = 0.5
            continue
        idx = int(np.argmax(f1s))
        out[name] = float(ts[idx])
    return out


def metrics(prob: np.ndarray, y: np.ndarray, head_names,
            thresholds: Optional[Mapping[str, float]] = None,
            ) -> Dict[str, dict]:
    """Per-head metrics. When `thresholds` is not given, defaults to 0.5
    for every head (back-compatible behaviour)."""
    out: Dict[str, dict] = {}
    for h, name in enumerate(head_names):
        p, t = prob[:, h], y[:, h].astype(int)
        unique = np.unique(t)
        if unique.size < 2:
            auroc = float("nan")
            auprc = float("nan")
        else:
            auroc = float(roc_auc_score(t, p))
            auprc = float(average_precision_score(t, p))
        thr = float((thresholds or {}).get(name, 0.5))
        pred = (p >= thr).astype(int)
        cm = confusion_matrix(t, pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        out[name] = {
            "auroc": auroc,
            "auprc": auprc,
            "threshold": thr,
            "f1": float(f1_score(t, pred, zero_division=0)),
            "accuracy": float((pred == t).mean()),
            "n_pos": int(t.sum()),
            "n_neg": int((1 - t).sum()),
            "cm": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        }
    return out


def print_report(name: str, m: Dict[str, dict]) -> None:
    print(f"\n  === {name} ===")
    print(f"  {'head':<5} {'thr':>5} {'AUROC':>7} {'AUPRC':>7} {'F1':>7} "
          f"{'Acc':>7}  {'TN':>4} {'FP':>4} {'FN':>4} {'TP':>4}  "
          f"{'n+':>4} {'n-':>4}")
    for head, v in m.items():
        cm = v["cm"]
        print(f"  {head:<5} {v.get('threshold', 0.5):>5.2f} "
              f"{v['auroc']:>7.3f} {v['auprc']:>7.3f} "
              f"{v['f1']:>7.3f} {v['accuracy']:>7.3f}  "
              f"{cm['tn']:>4} {cm['fp']:>4} {cm['fn']:>4} {cm['tp']:>4}  "
              f"{v['n_pos']:>4} {v['n_neg']:>4}")
