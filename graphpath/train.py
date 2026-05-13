"""GraphPath training loop (SGD + early stopping)."""
from __future__ import annotations

import copy
from typing import Dict, List

import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau

from .model import GraphPath, weighted_bce_with_logits


def _epoch(model: GraphPath, loader, pos_weight, optim=None, device="cpu"):
    train_mode = optim is not None
    model.train(train_mode)
    total_loss, n = 0.0, 0
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        out = model(xb)
        loss = weighted_bce_with_logits(out["logits"], yb, pos_weight)
        if train_mode:
            optim.zero_grad()
            loss.backward()
            optim.step()
        bs = xb.size(0)
        total_loss += loss.item() * bs
        n += bs
    return {"loss": total_loss / max(n, 1)}


def fit(model: GraphPath, loaders: dict, pos_weight, cfg,
        device: str = "cpu", verbose: bool = True) -> Dict[str, List]:
    pw = torch.as_tensor(pos_weight, dtype=torch.float32, device=device)
    optim = SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum,
                weight_decay=cfg.weight_decay)
    sched = ReduceLROnPlateau(
        optim, mode="min", factor=cfg.plateau_factor,
        patience=cfg.plateau_patience,
    )

    history = {"train_loss": [], "val_loss": [], "lr": []}
    best_val, best_state, patience_left = float("inf"), None, cfg.patience

    for epoch in range(1, cfg.max_epochs + 1):
        tr = _epoch(model, loaders["train"], pw, optim=optim, device=device)
        with torch.no_grad():
            va = _epoch(model, loaders["val"], pw, optim=None, device=device)

        sched.step(va["loss"])
        lr_now = optim.param_groups[0]["lr"]
        history["train_loss"].append(tr["loss"])
        history["val_loss"].append(va["loss"])
        history["lr"].append(lr_now)

        improved = va["loss"] < best_val - 1e-6
        if improved:
            best_val = va["loss"]
            best_state = copy.deepcopy(model.state_dict())
            patience_left = cfg.patience
        else:
            patience_left -= 1

        if verbose and (epoch == 1 or epoch % 5 == 0 or improved):
            star = "*" if improved else " "
            print(f"  ep {epoch:3d} {star}  train {tr['loss']:.4f}  "
                  f"val {va['loss']:.4f}  lr {lr_now:.1e}")

        if patience_left <= 0:
            if verbose:
                print(f"  early stop @ epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return {"history": history, "best_val_loss": best_val}
