"""Per-head metrics: AUROC, AUPRC, accuracy, F1, confusion matrix."""
from __future__ import annotations

from typing import Dict

import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score, confusion_matrix, f1_score, roc_auc_score,
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


def metrics(prob: np.ndarray, y: np.ndarray, head_names) -> Dict[str, dict]:
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
        pred = (p >= 0.5).astype(int)
        cm = confusion_matrix(t, pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        out[name] = {
            "auroc": auroc,
            "auprc": auprc,
            "f1": float(f1_score(t, pred, zero_division=0)),
            "accuracy": float((pred == t).mean()),
            "n_pos": int(t.sum()),
            "n_neg": int((1 - t).sum()),
            "cm": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        }
    return out


def print_report(name: str, m: Dict[str, dict]) -> None:
    print(f"\n  === {name} ===")
    print(f"  {'head':<5} {'AUROC':>7} {'AUPRC':>7} {'F1':>7} {'Acc':>7}  "
          f"{'TN':>4} {'FP':>4} {'FN':>4} {'TP':>4}  {'n+':>4} {'n-':>4}")
    for head, v in m.items():
        cm = v["cm"]
        print(f"  {head:<5} {v['auroc']:>7.3f} {v['auprc']:>7.3f} "
              f"{v['f1']:>7.3f} {v['accuracy']:>7.3f}  "
              f"{cm['tn']:>4} {cm['fp']:>4} {cm['fn']:>4} {cm['tp']:>4}  "
              f"{v['n_pos']:>4} {v['n_neg']:>4}")
