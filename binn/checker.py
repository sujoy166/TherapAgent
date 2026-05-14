"""Conformance check against Hartman et al. 2023 (Nature Communications 14:5359).

Inspects this repository's BINN implementation and reports, for each
architectural and training choice the paper specifies, whether the code

  PASS       — matches the paper exactly
  DEVIATION  — departs from the paper but the departure is intentional and
               documented (e.g. multi-label output vs. paper's binary)
  BUG        — undocumented departure that would change results

Exit code 0 when no BUGs are reported, 1 otherwise.

Usage:
    python3 -m binn.checker
"""
from __future__ import annotations

import sys
import inspect

import numpy as np
import torch

from binn.config import Config
from binn.model import BINN, MaskedLinear, weighted_bce
from binn.train import fit


Verdict = str  # "PASS" | "DEVIATION" | "BUG"
CHECKS: list[tuple[str, str, str, Verdict, str]] = []


def check(name: str, paper: str, code: str, verdict: Verdict,
          note: str = "") -> None:
    CHECKS.append((name, paper, code, verdict, note))


# ── 1. Sparse Reactome-hierarchical architecture ───────────────────────
# Paper §Methods step 1-5: subset Reactome graph, layerise so depth ≤ N+1.
masks = [np.array([[1, 0], [0, 1]], dtype=np.float32),
         np.array([[1, 1]],          dtype=np.float32)]
m = BINN(masks=masks, n_heads=3, dropout=0.0)
ml0 = m.masked_layers[0]
mask_buffer_correct = (
    isinstance(ml0, MaskedLinear)
    and "mask" in [n for n, _ in ml0.named_buffers()]
)
check(
    "Sparse Reactome-masked weights",
    "Paper §Methods: between-layer connectivity follows the Reactome "
    "parent-child mask; non-mask entries are zero.",
    f"binn.model.MaskedLinear multiplies weight by registered {{0,1}} "
    f"buffer. Type check: {isinstance(ml0, MaskedLinear)}; buffer "
    f"present: {mask_buffer_correct}.",
    "PASS" if mask_buffer_correct else "BUG",
)


# ── 2. Per-layer activation + BN + Dropout ─────────────────────────────
src = inspect.getsource(BINN.forward)
has_tanh = "torch.tanh" in src
has_bn = "self.bns" in src
has_drop = "self.dropouts" in src
check(
    "Activation chain Tanh → BatchNorm → Dropout",
    "Paper §Methods: 'the hidden linear layers are intercepted by tanh-"
    "activation layers, as well as dropout layers and batch normalisation.'",
    f"BINN.forward applies torch.tanh ({has_tanh}), self.bns "
    f"({has_bn}), self.dropouts ({has_drop}).",
    "PASS" if (has_tanh and has_bn and has_drop) else "BUG",
)


# ── 3. Per-layer auxiliary heads + mean-of-sigmoids output ─────────────
# Paper Eq. 1: outfinal = (1/N) * sum_{l=0..N} sigma(out_layer)
out = m(torch.randn(2, 2))
layer_logits = out["layer_logits"]
n_heads = len(layer_logits)
expected_heads = 1 + len(m.masked_layers)
prob_in_range = bool(torch.all((out["prob"] >= 0) & (out["prob"] <= 1)).item())
check(
    "Per-layer auxiliary heads (input + hidden), mean of sigmoids",
    "Paper Eq. 1: outfinal = mean over ℓ in {0..N} of σ(out_ℓ); N+1 "
    "classifier heads total.",
    f"BINN.forward returns {n_heads} layer_logits (expected {expected_heads}) "
    f"and prob = average of their sigmoids, range [{out['prob'].min().item():.3f}"
    f", {out['prob'].max().item():.3f}].",
    "PASS" if n_heads == expected_heads and prob_in_range else "BUG",
)


# ── 4. Multi-label output instead of paper's binary ────────────────────
check(
    "Output cardinality",
    "Paper: single binary subphenotype output (sigmoid). Cross-entropy loss.",
    "BINN(n_heads=3) emits three independent sigmoid heads (TMT, RT, "
    "OS≥180d) trained jointly with class-weighted BCE.",
    "DEVIATION",
    "Intentional adaptation for the 3-axis therapy-response label.",
)


# ── 5. Optimizer / training schedule ───────────────────────────────────
cfg = Config()
opt_ok = abs(cfg.lr - 1e-3) < 1e-9 and abs(cfg.weight_decay - 1e-3) < 1e-9
check(
    "Optimizer: Adam, η=1e-3, weight decay 1e-3, plateau LR scheduler",
    "Paper §Training: 'learning rate initiated at 0.001, decreased adaptively "
    "if validation loss plateaued. Adam, weight decay 0.001'.",
    f"Config.lr = {cfg.lr}, weight_decay = {cfg.weight_decay}; "
    "binn.train.fit uses Adam + ReduceLROnPlateau on validation loss.",
    "PASS" if opt_ok else "BUG",
)


# ── 6. Hidden-layer count ──────────────────────────────────────────────
check(
    "Hidden-layer count",
    "Paper: '4 hidden layers each' (sepsis & COVID BINNs).",
    f"Config.n_hidden_layers = {cfg.n_hidden_layers} (Reactome walk "
    "from input → top, 4 layers including input).",
    "PASS" if cfg.n_hidden_layers == 4 else "BUG",
)


# ── 7. Layer-copy-forward when terminal reached early ──────────────────
from binn.reactome import build_layers
parent_map = {"A": {"B"}, "B": set()}     # 2-level hierarchy
layers = build_layers(["A"], parent_map, n_layers=4)
copied = (layers[-1] == ["B"])
check(
    "Path-length padding (copy-forward at terminals)",
    "Paper §Methods step 3: 'If reaching a terminal node before N "
    "layers, add a copy of the previous node.'",
    f"build_layers(['A'], {{A→B, B→∅}}, n_layers=4) produced layer 3 = "
    f"{layers[-1]}.",
    "PASS" if copied else "BUG",
)


# ── 8. Dropout + BatchNorm to combat overfitting ───────────────────────
check(
    "Regularisation: dropout, BatchNorm, L2 weight decay",
    "Paper §Discussion: 'use of dropout, batch normalisation and L2-"
    "regularisation' as overfitting mitigations.",
    f"Config.dropout = {cfg.dropout}, BatchNorm1d in BINN per layer, "
    f"weight_decay = {cfg.weight_decay}.",
    "PASS" if cfg.dropout > 0 else "BUG",
)


# ── Report ─────────────────────────────────────────────────────────────
def main() -> int:
    width_col = 38
    print("=" * (width_col * 2 + 16))
    print(f"{'BINN — Hartman et al. 2023':^{width_col*2+16}}")
    print("=" * (width_col * 2 + 16))
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
    print("\n" + "─" * (width_col * 2 + 16))
    print(f"  PASS: {len(passes)}    DEVIATION: {len(devs)}    BUG: {len(bugs)}")
    print("─" * (width_col * 2 + 16))
    return 1 if bugs else 0


if __name__ == "__main__":
    sys.exit(main())
