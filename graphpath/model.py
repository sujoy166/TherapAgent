"""GraphPath model (Ma & Wang 2024, *Bioinformatics*).

Adapted to pathway-level ssGSEA inputs: each pathway carries a single scalar
(its ssGSEA score) rather than the gene-level multi-omic vector used in the
paper. A learnable per-pathway projection lifts the scalar to the embedding
dimension before the multi-head graph-attention layer; everything downstream
(GAT → tanh readout → FC) is faithful to the paper.

Reference: §2.4–2.5 (model), §2.6 (training).
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GATLayer(nn.Module):
    """One multi-head GAT layer with K independent attention heads, ELU activation.

    Edges come from a fixed {0,1} adjacency that includes self-loops (added in
    forward to ensure every node attends to itself). Outputs are concatenated
    across heads as in the original paper (Eq. 2).
    """

    def __init__(self, in_dim: int, out_dim_per_head: int, n_heads: int,
                 dropout: float = 0.4, negative_slope: float = 0.2):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim_per_head
        self.n_heads = n_heads
        self.negative_slope = negative_slope
        self.dropout = dropout

        # Per-head linear transform (shared across nodes within a head)
        self.W = nn.Parameter(torch.empty(n_heads, in_dim, out_dim_per_head))
        # Per-head attention vector a ∈ R^{2*out_dim_per_head}
        self.a = nn.Parameter(torch.empty(n_heads, 2 * out_dim_per_head))
        for k in range(n_heads):
            nn.init.xavier_uniform_(self.W[k])
            nn.init.xavier_uniform_(self.a[k].unsqueeze(0))

    def forward(self, h: torch.Tensor, adj_mask: torch.Tensor) -> torch.Tensor:
        """
        h        : (B, N, in_dim)
        adj_mask : (N, N) bool — True where edges exist (incl. self-loops)
        returns  : (B, N, n_heads * out_dim_per_head)
        """
        B, N, _ = h.shape
        outs = []
        for k in range(self.n_heads):
            Wh = h @ self.W[k]                              # (B, N, out_dim)
            # Build raw attention logits e_ij = LeakyReLU(a^T [Wh_i || Wh_j])
            a_src, a_dst = self.a[k, :self.out_dim], self.a[k, self.out_dim:]
            e_src = Wh @ a_src                              # (B, N)
            e_dst = Wh @ a_dst                              # (B, N)
            # e_ij = e_src[i] + e_dst[j] → (B, N, N)
            e = e_src.unsqueeze(2) + e_dst.unsqueeze(1)
            e = F.leaky_relu(e, negative_slope=self.negative_slope)
            # Mask non-edges to -inf before softmax
            e = e.masked_fill(~adj_mask.unsqueeze(0), float("-inf"))
            alpha = F.softmax(e, dim=2)                     # row-wise softmax
            alpha = F.dropout(alpha, p=self.dropout, training=self.training)
            outs.append(alpha @ Wh)                         # (B, N, out_dim)
        out = torch.cat(outs, dim=-1)                       # (B, N, K*out_dim)
        return F.elu(out)


class GraphPath(nn.Module):
    """GraphPath: project → multi-head GAT → tanh readout → FC heads.

    Parameters
    ----------
    n_pathways : int
        Number of input pathway nodes (Reactome-matched).
    adjacency  : np.ndarray of shape (n_pathways, n_pathways)
        Symmetric {0,1} adjacency.
    embed_dim  : int (= F' in the paper, per head)
    n_heads    : int (= K = 3 in the paper)
    n_outputs  : int (= 3 for our multi-label heads TMT/RT/OS)
    dropout    : float
    """

    def __init__(self, n_pathways: int, adjacency: np.ndarray,
                 embed_dim: int = 64, n_heads: int = 3,
                 n_outputs: int = 3, dropout: float = 0.4):
        super().__init__()
        if adjacency.shape != (n_pathways, n_pathways):
            raise ValueError("adjacency shape must equal (n_pathways, n_pathways)")
        self.n_pathways = n_pathways
        self.n_outputs = n_outputs

        # Per-pathway 1-D ssGSEA score → embed_dim. We use a shared linear
        # plus a pathway-specific bias so each node retains an identity.
        self.proj_weight = nn.Parameter(torch.empty(1, embed_dim))
        self.proj_bias = nn.Parameter(torch.zeros(n_pathways, embed_dim))
        nn.init.xavier_uniform_(self.proj_weight)

        # Multi-head GAT — concatenates K heads → K * embed_dim
        self.gat = GATLayer(
            in_dim=embed_dim, out_dim_per_head=embed_dim,
            n_heads=n_heads, dropout=dropout,
        )

        # Readout: shared Linear (K*embed_dim → 1) per pathway, tanh.
        # Produces a length-N vector summarising the patient (paper §2.5).
        self.readout = nn.Linear(embed_dim * n_heads, 1)

        # Final fully-connected layer: N pathway scalars → n_outputs logits.
        self.head = nn.Linear(n_pathways, n_outputs)

        self.dropout = nn.Dropout(dropout)

        # Pre-compute the binary adjacency mask with self-loops
        mask = (adjacency > 0).astype(np.bool_)
        np.fill_diagonal(mask, True)
        self.register_buffer("adj_mask", torch.tensor(mask, dtype=torch.bool))

    def forward(self, x: torch.Tensor) -> dict:
        """
        x : (B, n_pathways) ssGSEA scores
        Returns dict with
          - "logits": (B, n_outputs)
          - "prob":   (B, n_outputs) sigmoid of logits
          - "graph_embed": (B, n_pathways) pre-FC pathway scalars
        """
        # Lift each scalar score to embed_dim with a pathway-specific bias
        h = x.unsqueeze(-1) * self.proj_weight + self.proj_bias  # (B, N, d)
        h = torch.tanh(h)
        h = self.dropout(h)

        # Multi-head GAT
        h = self.gat(h, self.adj_mask)                            # (B, N, K*d)

        # Per-pathway readout → scalar
        p = torch.tanh(self.readout(h)).squeeze(-1)               # (B, N)
        p = self.dropout(p)

        logits = self.head(p)                                     # (B, n_outputs)
        prob = torch.sigmoid(logits)
        return {"logits": logits, "prob": prob, "graph_embed": p}

    def trainable_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def weighted_bce_with_logits(logits: torch.Tensor, target: torch.Tensor,
                              pos_weight: torch.Tensor) -> torch.Tensor:
    """Per-head class-weighted BCE, computed from logits for numerical stability."""
    return F.binary_cross_entropy_with_logits(
        logits, target, pos_weight=pos_weight, reduction="mean"
    )
