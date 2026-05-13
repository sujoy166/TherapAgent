"""PATH model — pathway-aware edge-conditioned graph transformer.

Reference: Howlader, Islam, Le (2026), §4.2 *Model architecture*.

Our adaptation
--------------
The paper's Stage 1 (FiLM-modulated gene embeddings) and Stage 2 (within-
pathway attention pooling) consume *gene-level* CNV + mutation data, which
this dataset does not contain — only pathway-level ssGSEA scores. We
therefore replace Stages 1-2 with a learnable per-pathway projection that
lifts each pathway's scalar score to the embedding dimension (with a
pathway-specific bias to preserve identity). Stages 3 (Laplacian positional
encoding + edge-aware graph transformer blocks) and 4 (attention-weighted
readout + MLP classifier) are faithful to the paper.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def laplacian_positional_encoding(A: np.ndarray, k: int) -> np.ndarray:
    """Compute the first k eigenvectors of the symmetric normalised Laplacian
    L = I − D^{-1/2} A D^{-1/2} (paper §4.2, Eq. 9-10).

    Returns an (N, k) float32 array; if N < k, the extras are zero-padded.
    """
    N = A.shape[0]
    deg = A.sum(axis=1)
    deg = np.clip(deg, 1e-8, None)
    d_inv_sqrt = 1.0 / np.sqrt(deg)
    L = np.eye(N) - (A * d_inv_sqrt[:, None]) * d_inv_sqrt[None, :]
    # Use eigh because L is symmetric → real eigenvalues + orthonormal eigenvectors.
    evals, evecs = np.linalg.eigh(L)
    # Drop the trivial near-zero eigenpair, take next k.
    order = np.argsort(evals)
    evecs = evecs[:, order]
    keep = min(k, N - 1)
    pe = np.zeros((N, k), dtype=np.float32)
    pe[:, :keep] = evecs[:, 1:1 + keep].astype(np.float32)
    return pe


class EdgeAwareTransformerBlock(nn.Module):
    """One edge-aware multi-head self-attention block (paper §4.2 Stage 3).

    Combines:
      • Soft structural mask m_struct (Eq. 13): 0 for real edges, −10 elsewhere.
      • Edge-conditioned attention bias φ from a learnable projection of the
        scalar adjacency weight (Eq. 14–16).
      • BatchNorm-wrapped residual MLP FFN with expansion factor 4 (Eq. 18-19).
      • Parallel edge-feature update (Eq. 20).
    """

    def __init__(self, d: int, n_heads: int, ffn_expansion: int = 4,
                 dropout: float = 0.2):
        super().__init__()
        if d % n_heads != 0:
            raise ValueError("embed_dim must be divisible by n_heads")
        self.d = d
        self.h = n_heads
        self.dh = d // n_heads

        self.q_proj = nn.Linear(d, d)
        self.k_proj = nn.Linear(d, d)
        self.v_proj = nn.Linear(d, d)
        self.o_proj = nn.Linear(d, d)

        # Per-head linear projection of the scalar edge feature → gain
        self.edge_gain_w = nn.Parameter(torch.empty(n_heads))
        self.edge_gain_b = nn.Parameter(torch.zeros(n_heads))
        nn.init.normal_(self.edge_gain_w, mean=0.0, std=0.02)

        # FFN (Eq. 18-19)
        self.ffn = nn.Sequential(
            nn.Linear(d, ffn_expansion * d),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_expansion * d, d),
            nn.Dropout(dropout),
        )

        # Per-token BatchNorm (paper reshapes B*P, d so BN normalises features)
        self.bn1 = nn.BatchNorm1d(d)
        self.bn2 = nn.BatchNorm1d(d)

        # Parallel edge feature update (Eq. 20)
        self.edge_mlp = nn.Sequential(
            nn.Linear(1, d),
            nn.GELU(),
            nn.Linear(d, 1),
        )
        self.edge_bn = nn.BatchNorm1d(1)

    def _bn_per_token(self, x: torch.Tensor, bn: nn.BatchNorm1d) -> torch.Tensor:
        B, P, D = x.shape
        if B * P > 1:
            return bn(x.reshape(B * P, D)).reshape(B, P, D)
        # Singleton-safe BN (paper §4.2): use running stats.
        bn.eval()
        out = bn(x.reshape(B * P, D)).reshape(B, P, D)
        bn.train(self.training)
        return out

    def forward(self, x: torch.Tensor, edge_feat: torch.Tensor,
                struct_mask: torch.Tensor) -> tuple:
        """
        x          : (B, P, d)
        edge_feat  : (P, P, 1) — scalar adjacency weights, learnable per layer
        struct_mask: (P, P)    — soft structural mask (0 or −10), float
        """
        B, P, _ = x.shape

        Q = self.q_proj(x).reshape(B, P, self.h, self.dh).transpose(1, 2)  # (B,h,P,dh)
        K = self.k_proj(x).reshape(B, P, self.h, self.dh).transpose(1, 2)
        V = self.v_proj(x).reshape(B, P, self.h, self.dh).transpose(1, 2)

        # Scaled dot-product logits: (B, h, P, P)
        scale = 1.0 / math.sqrt(self.dh)
        logits = torch.einsum("bhpd,bhqd->bhpq", Q, K) * scale

        # Edge-conditioned per-head bias φ (Eq. 14-15). edge_feat: (P,P,1)
        # gain_h(p,q) = softplus(w_h * a_pq + b_h); φ_h(p,q) = log(gain_h + ε)
        ef = edge_feat.squeeze(-1).unsqueeze(0)                  # (1, P, P)
        gain = F.softplus(self.edge_gain_w[:, None, None] * ef
                          + self.edge_gain_b[:, None, None])     # (h, P, P)
        phi = torch.log(gain + 1e-6)                             # (h, P, P)

        # Final mask: M_pq = m_struct + φ, broadcast over heads
        M = struct_mask.unsqueeze(0) + phi                       # (h, P, P)
        logits = logits + M.unsqueeze(0)                          # (B, h, P, P)

        attn = F.softmax(logits, dim=-1)
        out = torch.einsum("bhpq,bhqd->bhpd", attn, V)            # (B, h, P, dh)
        out = out.transpose(1, 2).reshape(B, P, self.d)
        out = self.o_proj(out)

        # Residual + BN + FFN + BN
        x = self._bn_per_token(x + out, self.bn1)
        x = self._bn_per_token(x + self.ffn(x), self.bn2)

        # Edge feature update (Eq. 20)
        ef_flat = edge_feat.reshape(-1, 1)
        ef_upd = self.edge_mlp(ef_flat)
        if ef_flat.size(0) > 1:
            ef_upd = self.edge_bn(ef_upd)
        edge_feat = ef_upd.reshape_as(edge_feat)
        return x, edge_feat


class PathGraphTransformer(nn.Module):
    """End-to-end PATH model (adapted to pathway-level ssGSEA inputs).

    Parameters
    ----------
    n_pathways  : int
    adjacency   : np.ndarray of shape (n_pathways, n_pathways) — weighted
    embed_dim   : int (d)
    n_heads     : int (H)
    n_layers    : int (L)
    n_outputs   : int — number of binary heads (3 for TMT/RT/OS)
    laplacian_k : int — number of PE eigenvectors
    soft_mask_penalty : float — value applied to non-edges in the soft mask
    ffn_expansion : int
    dropout     : float
    """

    def __init__(self, n_pathways: int, adjacency: np.ndarray,
                 embed_dim: int = 64, n_heads: int = 4, n_layers: int = 2,
                 n_outputs: int = 3, laplacian_k: int = 16,
                 soft_mask_penalty: float = -10.0,
                 ffn_expansion: int = 4, dropout: float = 0.2):
        super().__init__()
        if adjacency.shape != (n_pathways, n_pathways):
            raise ValueError("adjacency shape must equal (n_pathways, n_pathways)")
        self.n_pathways = n_pathways
        self.embed_dim = embed_dim
        self.n_outputs = n_outputs

        # Per-pathway projection: scalar ssGSEA → d
        self.proj_weight = nn.Parameter(torch.empty(1, embed_dim))
        self.proj_bias = nn.Parameter(torch.zeros(n_pathways, embed_dim))
        nn.init.xavier_uniform_(self.proj_weight)

        # Laplacian positional encoding (paper Eq. 10)
        pe = laplacian_positional_encoding(adjacency.astype(np.float32),
                                            k=laplacian_k)
        self.register_buffer("pe", torch.tensor(pe))                  # (P, k)
        self.pe_proj = nn.Linear(laplacian_k + 1, embed_dim)

        # Structural mask: 0 where adjacency > 0 or on diagonal, else penalty
        mask = np.full((n_pathways, n_pathways), soft_mask_penalty,
                       dtype=np.float32)
        np.fill_diagonal(mask, 0.0)
        mask[adjacency > 0] = 0.0
        self.register_buffer("struct_mask", torch.tensor(mask))

        # Edge feature (scalar adjacency value) initialised from the graph
        self.register_buffer(
            "edge_init",
            torch.tensor(adjacency.astype(np.float32)).unsqueeze(-1)  # (P, P, 1)
        )
        # Learnable per-layer edge offset
        self.edge_offset = nn.Parameter(
            torch.zeros(n_layers, n_pathways, n_pathways, 1)
        )

        # Pre-compute normalised degree term for PE concat (Eq. 10)
        deg = torch.tensor(adjacency.sum(axis=1), dtype=torch.float32)
        deg = deg / (n_pathways + 1e-8)
        self.register_buffer("deg_norm", deg.unsqueeze(-1))           # (P, 1)

        # Transformer stack
        self.blocks = nn.ModuleList([
            EdgeAwareTransformerBlock(
                d=embed_dim, n_heads=n_heads,
                ffn_expansion=ffn_expansion, dropout=dropout,
            )
            for _ in range(n_layers)
        ])

        # Stage 4 readout (paper Eq. 21-22)
        self.readout_w = nn.Linear(embed_dim, embed_dim // 2)
        self.readout_v = nn.Linear(embed_dim // 2, 1)

        # Classification head (paper Eq. 23) — adapted to n_outputs
        self.head_proj = nn.Linear(embed_dim, embed_dim)
        self.head_bn = nn.BatchNorm1d(embed_dim)
        self.head_dropout = nn.Dropout(dropout)
        self.head_out = nn.Linear(embed_dim, n_outputs)

    def _positional_encoding(self) -> torch.Tensor:
        """(P, d) positional features from Laplacian eigenvectors + degree.

        Applies random sign-flip augmentation during training (Eq. 11).
        """
        if self.training:
            sign = torch.empty(self.pe.size(1), device=self.pe.device).uniform_(-1, 1).sign()
            pe = self.pe * sign.unsqueeze(0)
        else:
            pe = self.pe
        x = torch.cat([pe, self.deg_norm], dim=-1)                    # (P, k+1)
        return self.pe_proj(x)                                         # (P, d)

    def _safe_bn(self, x: torch.Tensor) -> torch.Tensor:
        """Singleton-safe BN over (B, d) — paper §4.2."""
        if x.size(0) > 1:
            return self.head_bn(x)
        was_training = self.head_bn.training
        self.head_bn.eval()
        out = self.head_bn(x)
        self.head_bn.train(was_training)
        return out

    def forward(self, x: torch.Tensor) -> dict:
        """
        x : (B, n_pathways) ssGSEA scores
        Returns dict {"logits", "prob", "graph_attn"}.
        """
        # Per-pathway projection (replaces FiLM + attention pooling on ssGSEA)
        h = x.unsqueeze(-1) * self.proj_weight + self.proj_bias       # (B, P, d)
        h = torch.tanh(h)
        h = h + self._positional_encoding().unsqueeze(0)              # broadcast

        edge_feat = self.edge_init + self.edge_offset[0]              # (P, P, 1)
        for i, block in enumerate(self.blocks):
            if i > 0:
                edge_feat = edge_feat + self.edge_offset[i]
            h, edge_feat = block(h, edge_feat, self.struct_mask)

        # Attention-weighted readout (Eq. 21-22)
        attn_logits = self.readout_v(torch.tanh(self.readout_w(h)))   # (B, P, 1)
        attn = F.softmax(attn_logits, dim=1)                           # (B, P, 1)
        g = (attn * h).sum(dim=1)                                      # (B, d)

        # Classification head (Eq. 23)
        z = self.head_proj(g)
        z = self._safe_bn(z)
        z = F.gelu(z)
        z = self.head_dropout(z)
        logits = self.head_out(z)
        prob = torch.sigmoid(logits)
        return {"logits": logits, "prob": prob,
                "graph_attn": attn.squeeze(-1)}

    def trainable_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def weighted_bce_with_logits(logits: torch.Tensor, target: torch.Tensor,
                              pos_weight: torch.Tensor) -> torch.Tensor:
    """Per-head class-weighted BCE on logits."""
    return F.binary_cross_entropy_with_logits(
        logits, target, pos_weight=pos_weight, reduction="mean"
    )
