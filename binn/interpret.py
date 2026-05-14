"""Pathway importance scoring via gradient × input.

A fast, model-agnostic post-hoc attribution that approximates SHAP for
linear/near-linear models. For each head ``h``, the importance of
pathway ``p`` is

    importance_p(h) = E_x[ |x_p · (∂ŷ_h / ∂x_p)| ]

averaged over the test set. Higher score → that pathway carries more
signal for that head.

The three trained models in this repository all accept a (B, P) input
tensor and emit a dict containing ``"prob"`` of shape (B, n_heads), so a
single utility works across BINN, GraphPath, and PATH.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch


@torch.enable_grad()
def pathway_importance(model, X: torch.Tensor, head_idx: int) -> np.ndarray:
    """Mean |x ⋅ ∂prob/∂x| per pathway, head_idx-specific.

    Parameters
    ----------
    model
        A trained module whose ``forward(x)`` returns a dict with a
        ``"prob"`` key of shape (B, n_heads).
    X
        Input tensor, shape (B, P). Will be cloned and require_grad-enabled.
    head_idx
        Column of ``prob`` to attribute.

    Returns
    -------
    (P,) numpy array of non-negative importance scores.
    """
    model.eval()
    X = X.detach().clone().requires_grad_(True)
    out = model(X)
    target = out["prob"][:, head_idx].sum()
    target.backward()
    importance = (X * X.grad).abs().mean(dim=0).detach().cpu().numpy()
    return importance


def compute_all_heads(model, X: torch.Tensor, head_names: Sequence[str]
                      ) -> Mapping[str, np.ndarray]:
    """Return {head_name: (P,) importance vector} for all heads."""
    out = {}
    for h_idx, h_name in enumerate(head_names):
        out[h_name] = pathway_importance(model, X, h_idx)
    return out


def top_k_pathways(importance: np.ndarray, pathway_names: Sequence[str],
                   k: int = 20) -> list[tuple[str, float]]:
    """Return list of (pathway_name, score) sorted by importance descending."""
    idx = np.argsort(importance)[::-1][:k]
    return [(pathway_names[i], float(importance[i])) for i in idx]


def save_importance(path: Path, head_importance: Mapping[str, np.ndarray],
                    pathway_names: Iterable[str]) -> Path:
    """Persist as a single .npz with per-head arrays + pathway-name index."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        pathway_names=np.array(list(pathway_names), dtype=object),
        **{h: v for h, v in head_importance.items()},
    )
    return path
