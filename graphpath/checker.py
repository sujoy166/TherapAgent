"""Conformance check against Ma & Wang 2024 (Bioinformatics 40:btae165).

Inspects this repository's GraphPath implementation and reports, for each
architectural and training choice the paper specifies, whether the code

  PASS       — matches the paper exactly
  DEVIATION  — documented intentional departure
  BUG        — undocumented departure

Exit code 0 when no BUGs, 1 otherwise.
"""
from __future__ import annotations

import sys
import inspect

import numpy as np
import torch
import torch.nn.functional as F

from graphpath.config import Config
from graphpath.model import GATLayer, GraphPath, weighted_bce_with_logits


CHECKS: list[tuple[str, str, str, str, str]] = []


def check(name, paper, code, verdict, note=""):
    CHECKS.append((name, paper, code, verdict, note))


# ── 1. Multi-head GAT with K=3 heads, ELU activation ───────────────────
cfg = Config()
N = 8
A = np.zeros((N, N), dtype=np.float32); A[np.triu_indices(N, 1)] = 1
A = (A + A.T)
m = GraphPath(n_pathways=N, adjacency=A, embed_dim=cfg.embed_dim,
              n_heads=cfg.n_heads, n_outputs=3, dropout=0.0)

heads_ok = m.gat.n_heads == 3
elu_in_src = "F.elu" in inspect.getsource(GATLayer.forward)
check(
    "K=3 attention heads, ELU activation",
    "Paper §2.4: K=3 independent attention heads; σ(·) = ELU(x). "
    "Eq. 2: h'_i = concat_k σ(Σ_j e^k_ij W^k h_j).",
    f"GATLayer.n_heads = {m.gat.n_heads}; F.elu present in forward: {elu_in_src}.",
    "PASS" if heads_ok and elu_in_src else "BUG",
)


# ── 2. Bahdanau-style attention with LeakyReLU pre-softmax ─────────────
src = inspect.getsource(GATLayer.forward)
has_leaky = "leaky_relu" in src or "LeakyReLU" in src
has_softmax = "softmax" in src
check(
    "Attention coefficient: softmax(LeakyReLU(a^T [Wh_i || Wh_j]))",
    "Paper Eq. 1: e_ij = softmax(LeakyReLU(a^T [Wh_i || Wh_j])).",
    f"GATLayer.forward applies F.leaky_relu ({has_leaky}) then "
    f"F.softmax over neighbours ({has_softmax}).",
    "PASS" if has_leaky and has_softmax else "BUG",
)


# ── 3. Non-edge masking before softmax ─────────────────────────────────
has_mask = "masked_fill" in src and "-inf" in src or "float(\"-inf\")" in src
check(
    "Non-edge entries masked to -∞ before softmax",
    "Standard masked GAT (Veličković 2018); paper §2.4 'aggregate the "
    "1-order neighbours of node i and the node itself (set N_i)'.",
    f"GATLayer.forward masks ~adj_mask to -inf before softmax: {has_mask}.",
    "PASS" if has_mask else "BUG",
)


# ── 4. Self-loops on adjacency ─────────────────────────────────────────
# Verify: in the registered adj_mask buffer, the diagonal is True.
diag_loops = bool(m.adj_mask.diag().all().item())
check(
    "Self-loops added to adjacency",
    "Paper §2.4: 'aggregate the 1-order neighbours of node i and the node "
    "itself' — node attends to itself.",
    f"GraphPath registers adj_mask buffer with diagonal True: {diag_loops}.",
    "PASS" if diag_loops else "BUG",
)


# ── 5. Xavier initialisation of W and a ────────────────────────────────
init_src = inspect.getsource(GATLayer.__init__)
xavier = "xavier_uniform_" in init_src
check(
    "Xavier initialisation",
    "Paper §2.4: 'The weights a and W are randomly initialised using the "
    "Xavier initialisation method.'",
    f"GATLayer.__init__ uses nn.init.xavier_uniform_: {xavier}.",
    "PASS" if xavier else "BUG",
)


# ── 6. Tanh readout (1-D scalar per node) ──────────────────────────────
fwd = inspect.getsource(GraphPath.forward)
tanh_readout = "torch.tanh(self.readout(" in fwd
check(
    "Per-node tanh readout to 1-D scalar",
    "Paper Eq. 4: P_i = Tanh(W_p h'_i).",
    f"GraphPath.forward applies torch.tanh(self.readout(...)): {tanh_readout}.",
    "PASS" if tanh_readout else "BUG",
)


# ── 7. Final FC + sigmoid ─────────────────────────────────────────────
has_sigmoid = "torch.sigmoid" in fwd
has_head_fc = "self.head(" in fwd
check(
    "Output = sigmoid(W_y · P)",
    "Paper Eq. 5: y = Sigmoid(W_y · P), W_y ∈ R^N.",
    f"GraphPath.head is Linear({m.n_pathways} → 3); forward applies "
    f"torch.sigmoid: {has_sigmoid}; FC layer: {has_head_fc}.",
    "PASS" if has_sigmoid and has_head_fc else "BUG",
)


# ── 8. Optimizer & hyperparameters ────────────────────────────────────
opt_ok = (
    abs(cfg.lr - 0.05) < 1e-9 and
    abs(cfg.weight_decay - 0.05) < 1e-9 and
    abs(cfg.dropout - 0.4) < 1e-9 and
    cfg.momentum == 0.9
)
check(
    "SGD(η=0.05, momentum 0.9, weight decay 0.05, dropout 0.4)",
    "Paper §2.6: 'SGD optimiser, initial LR 0.05, dropout 0.4, weight "
    "decay 0.05.'",
    f"Config.lr = {cfg.lr}, momentum = {cfg.momentum}, weight_decay = "
    f"{cfg.weight_decay}, dropout = {cfg.dropout}.",
    "PASS" if opt_ok else "BUG",
)


# ── 9. Output cardinality (binary in paper, multi-label here) ─────────
check(
    "Output cardinality",
    "Paper: single binary primary/metastatic prediction (BCE loss).",
    "GraphPath(n_outputs=3) emits three independent sigmoid heads "
    "(TMT/RT/OS≥180d) trained jointly with class-weighted BCE.",
    "DEVIATION",
    "Multi-label adaptation for the therapy-response setting.",
)


# ── 10. Input feature representation (pathway-level vs gene-level) ────
check(
    "Per-pathway input features",
    "Paper §2.3: 'gene features of multi-omics of pathways are concatenated "
    "to obtain the initial feature matrix' — gene-level CNA+Mutation per "
    "pathway, F = 5004+7552 features.",
    "GraphPath replaces gene-level features with a single ssGSEA scalar "
    "per pathway, lifted to F'=64 by a learnable projection (proj_weight + "
    "per-pathway bias).",
    "DEVIATION",
    "Required because the TherapAgent dataset only provides pathway-level "
    "ssGSEA scores; downstream GAT machinery is untouched.",
)


def main() -> int:
    width = 92
    print("=" * width)
    print(f"{'GraphPath — Ma & Wang 2024':^{width}}")
    print("=" * width)
    for name, paper, code, verdict, note in CHECKS:
        sym = {"PASS": "✓", "DEVIATION": "⚠", "BUG": "✗"}[verdict]
        print(f"\n[{verdict}] {sym}  {name}")
        print(f"   paper : {paper}")
        print(f"   code  : {code}")
        if note:
            print(f"   note  : {note}")
    bugs = [c for c in CHECKS if c[3] == "BUG"]
    devs = [c for c in CHECKS if c[3] == "DEVIATION"]
    passes = [c for c in CHECKS if c[3] == "PASS"]
    print("\n" + "─" * width)
    print(f"  PASS: {len(passes)}    DEVIATION: {len(devs)}    BUG: {len(bugs)}")
    print("─" * width)
    return 1 if bugs else 0


if __name__ == "__main__":
    sys.exit(main())
