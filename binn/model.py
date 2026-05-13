"""Sparse multi-head BINN.

Architecture follows Hartman et al. 2023 (Nat. Comm.):
  - Sparse linear layers whose mask encodes Reactome parent/child edges.
  - Each hidden output passes Tanh → BatchNorm → Dropout.
  - An auxiliary multi-head classifier is attached after every layer; the
    final probability per head is the mean of all heads' sigmoid outputs
    (Eq. 1 in the paper).
"""
from __future__ import annotations

import math
from typing import List, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedLinear(nn.Module):
    """Linear layer whose weight is multiplied by a fixed {0,1} mask."""

    def __init__(self, mask: np.ndarray):
        super().__init__()
        if mask.ndim != 2:
            raise ValueError("mask must be 2-D (out_features, in_features)")
        out_features, in_features = mask.shape
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.register_buffer(
            "mask", torch.tensor(mask, dtype=torch.float32), persistent=True
        )
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        # Zero out weights that lie outside the mask so the very first
        # forward pass matches the masked structure (cosmetic; the runtime
        # mask multiply enforces sparsity regardless).
        with torch.no_grad():
            self.weight.mul_(self.mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return F.linear(x, self.weight * self.mask, self.bias)

    @property
    def n_active_params(self) -> int:
        return int(self.mask.sum().item()) + self.out_features  # weights + bias


class BINN(nn.Module):
    """Multi-head biologically informed neural network.

    Parameters
    ----------
    masks
        List of {0,1} arrays of shape (size_{l+1}, size_l) connecting
        consecutive Reactome layers (layer 0 = input pathways).
    n_heads
        Number of independent binary heads (e.g. 3 for TMT/RT/OS).
    dropout
        Dropout applied after each hidden layer's Tanh/BatchNorm block.
    """

    def __init__(self, masks: Sequence[np.ndarray], n_heads: int = 3,
                 dropout: float = 0.2):
        super().__init__()
        if not masks:
            raise ValueError("Need at least one mask (one hidden layer).")

        # Sizes: layer 0 = masks[0].shape[1]; layer l>=1 = masks[l-1].shape[0]
        sizes = [int(masks[0].shape[1])] + [int(m.shape[0]) for m in masks]
        self.layer_sizes = sizes
        self.n_layers = len(masks)
        self.n_heads = n_heads

        # Hidden blocks: MaskedLinear → Tanh → BatchNorm → Dropout
        self.masked_layers = nn.ModuleList(MaskedLinear(m) for m in masks)
        self.bns = nn.ModuleList(nn.BatchNorm1d(s) for s in sizes[1:])
        self.dropouts = nn.ModuleList(nn.Dropout(dropout) for _ in masks)

        # Multi-task heads: one per layer (input + each hidden output)
        # producing n_heads logits each.
        self.classifier_heads = nn.ModuleList(
            nn.Linear(s, n_heads) for s in sizes
        )

    def forward(self, x: torch.Tensor) -> dict:
        """Return per-head averaged probabilities AND per-layer logits.

        Outputs
        -------
        dict with keys
          - "prob":        Tensor (B, n_heads) — averaged sigmoid output
          - "layer_logits": List[Tensor]      — one (B, n_heads) per head
          - "activations":  List[Tensor]      — post-activation per layer
        """
        activations: List[torch.Tensor] = [x]
        layer_logits: List[torch.Tensor] = [self.classifier_heads[0](x)]

        h = x
        for layer, bn, drop, head in zip(
            self.masked_layers, self.bns, self.dropouts,
            list(self.classifier_heads)[1:],
        ):
            h = layer(h)
            h = torch.tanh(h)
            h = bn(h)
            h = drop(h)
            activations.append(h)
            layer_logits.append(head(h))

        probs = torch.stack([torch.sigmoid(z) for z in layer_logits], dim=0)
        prob = probs.mean(dim=0)
        # Clamp away from exact 0/1 so the downstream BCE is finite even if
        # all heads saturate identically.
        prob = prob.clamp(min=1e-6, max=1.0 - 1e-6)
        return {"prob": prob, "layer_logits": layer_logits, "activations": activations}

    # ── Convenience ──────────────────────────────────────────────────────
    def trainable_parameter_count(self) -> int:
        """Effective parameter count, treating masked-out weights as inactive."""
        n = sum(m.n_active_params for m in self.masked_layers)
        # batch-norm γ/β
        n += sum(2 * bn.num_features for bn in self.bns)
        # classifier heads
        n += sum(h.weight.numel() + h.bias.numel() for h in self.classifier_heads)
        return n


def weighted_bce(prob: torch.Tensor, target: torch.Tensor,
                 pos_weight: torch.Tensor) -> torch.Tensor:
    """Per-head class-weighted BCE, applied after sigmoid+mean averaging.

    prob:   (B, H) in (0,1)
    target: (B, H) in {0,1}
    pos_weight: (H,) — multiplies the positive-class log term per head.
    """
    eps = 1e-7
    p = prob.clamp(eps, 1.0 - eps)
    loss = -(pos_weight * target * p.log() + (1.0 - target) * (1.0 - p).log())
    return loss.mean()
