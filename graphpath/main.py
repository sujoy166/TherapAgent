"""Phase-driven runner for GraphPath. Mirrors binn/main.py's structure.

Usage:
    python -m graphpath.main reactome   # download Reactome + build adjacency
    python -m graphpath.main data       # split, scale, save splits
    python -m graphpath.main train      # train multi-head GAT model
    python -m graphpath.main evaluate   # final metrics on val + test
    python -m graphpath.main all        # chain reactome → data → train → evaluate
"""
from __future__ import annotations

import argparse
import json
import pickle
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# Reuse data + reporting + metrics from binn (model-agnostic helpers).
from binn.data import (
    alignment_summary, fit_standardizer, list_pathway_columns, load_aligned,
    positive_weights, stratified_split,
)
from binn.evaluate import find_best_threshold, metrics, predict, print_report
from binn.reporting import write_table

from binn.config import COHORT_FILES

from .config import Config
from .model import GraphPath
from .reactome import assemble
from .train import fit


# ── helpers ──────────────────────────────────────────────────────────────
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _reactome_path(cfg): return cfg.artifacts_dir / "reactome.pkl"
def _splits_path(cfg):   return cfg.artifacts_dir / "splits.npz"
def _model_path(cfg):    return cfg.artifacts_dir / "graphpath.pt"
def _results_path(cfg):  return cfg.artifacts_dir / "results.json"


def _tex_dir(cfg: Config) -> Path:
    d = cfg.artifacts_dir / "tex"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ── Phase 1: Environment ─────────────────────────────────────────────────
def phase_env(cfg: Config) -> dict:
    print("PHASE 1: Environment")
    rows = [["Python", sys.version.split()[0]]]
    for name in ("numpy", "pandas", "scipy", "sklearn", "torch"):
        try:
            mod = __import__(name)
            rows.append([name, getattr(mod, "__version__", "?")])
        except Exception:
            rows.append([name, "MISSING"])
    for label, value in rows:
        print(f"  {label:<10} : {value}")
    tex = _tex_dir(cfg) / "01_environment.tex"
    write_table(
        tex, headers=["Package", "Version"], rows=rows,
        caption="Python environment used to train and evaluate the GraphPath model.",
        label="gp-env", align="lr",
    )
    print(f"  wrote LaTeX table → {tex.relative_to(cfg.project_root)}")
    return {"rows": rows}


# ── Phase 2: Reactome adjacency ──────────────────────────────────────────
def phase_reactome(cfg: Config) -> dict:
    print("PHASE 2: Reactome pathway-pathway adjacency")
    pathway_cols = list_pathway_columns(cfg.scores_csv)
    print(f"  pathway rows in scores CSV     : {len(pathway_cols)}")

    reactome = assemble(
        pathway_names=pathway_cols,
        cache_dir=cfg.cache_dir,
        pathways_url=cfg.pathways_url,
        relations_url=cfg.relations_url,
        species_prefix=cfg.species_prefix,
        include_siblings=cfg.include_siblings,
    )
    N = len(reactome["input_ids"])
    E = reactome["n_edges_undirected"]
    deg = reactome["adjacency"].sum(axis=1)
    print(f"  matched pathways               : {N} (missing {reactome['n_missing']})")
    print(f"  undirected edges               : {E}")
    print(f"  degree min/mean/max            : {deg.min():.0f} / {deg.mean():.2f} / {deg.max():.0f}")
    print(f"  siblings included              : {cfg.include_siblings}")

    with open(_reactome_path(cfg), "wb") as f:
        pickle.dump(reactome, f)
    print(f"  saved checkpoint → {_reactome_path(cfg).relative_to(cfg.project_root)}")

    rows = [
        ["Reactome-matched pathways", N],
        ["Unmatched pathway names", reactome["n_missing"]],
        ["Adjacency type", "parent/child + siblings" if cfg.include_siblings else "parent/child only"],
        ["Undirected edges", E],
        ["Average degree", f"{deg.mean():.2f}"],
        ["Max degree", int(deg.max())],
        ["Density (\\%)", f"{100 * E / (N * (N - 1) / 2):.2f}"],
    ]
    write_table(
        _tex_dir(cfg) / "02_pathway_graph.tex",
        headers=["Property", "Value"], rows=rows,
        caption=(
            "Reactome-derived pathway-pathway adjacency used by GraphPath. "
            "Edges connect parent/child pairs and (optionally) siblings that "
            "share a common parent in the Reactome hierarchy."
        ),
        label="gp-graph", align="lr",
    )
    print(f"  wrote LaTeX table → {(_tex_dir(cfg) / '02_pathway_graph.tex').relative_to(cfg.project_root)}")
    return reactome


# ── Phase 3: Data ────────────────────────────────────────────────────────
def phase_data(cfg: Config) -> dict:
    print("PHASE 3: Data preparation")
    if not _reactome_path(cfg).exists():
        raise SystemExit(f"Run phase reactome first — missing {_reactome_path(cfg)}")
    with open(_reactome_path(cfg), "rb") as f:
        reactome = pickle.load(f)

    align = alignment_summary(cfg.scores_csv, cfg.mapping_csv)
    print(f"  scores.csv samples             : {align['scores_samples']}")
    print(f"  mapping.csv labeled samples    : {align['label_samples']}")
    print(f"  intersection (used by model)   : {align['intersect_samples']}")

    X, Y, stage = load_aligned(cfg.scores_csv, cfg.mapping_csv,
                               reactome["input_names"])
    print(f"  X shape                        : {X.shape}")
    print(f"  Y shape                        : {Y.shape}")

    train_idx, val_idx, test_idx = stratified_split(
        X, Y, stage, val_frac=cfg.val_frac, test_frac=cfg.test_frac, seed=cfg.seed,
    )
    mu, sd = fit_standardizer(X.values[train_idx])
    Xn = (X.values - mu) / sd
    pw = positive_weights(Y.values[train_idx])

    print(f"  split (n)                      : train={len(train_idx)} "
          f"val={len(val_idx)} test={len(test_idx)}")
    print(f"  pos weights                    : "
          + ", ".join(f"{h}={w:.2f}" for h, w in zip(cfg.head_names, pw)))

    np.savez(
        _splits_path(cfg),
        X=Xn.astype(np.float32), Y=Y.values.astype(np.float32),
        train_idx=train_idx, val_idx=val_idx, test_idx=test_idx,
        mean=mu, std=sd, pos_weight=pw,
        sample_ids=np.array(X.index.tolist()),
        stage=stage.values,
    )

    rows_align = [
        ["pathway\\_scores.csv – samples",   align["scores_samples"]],
        ["pathway\\_scores.csv – pathways",  align["scores_pathways"]],
        ["pathway\\_phenotype\\_mapping.csv – labeled samples",
         align["label_samples"]],
        ["Reactome-matched input pathways",
         len(reactome["input_names"])],
        ["Samples used by model (intersection)",
         align["intersect_samples"]],
        ["Final X shape (samples × pathways)", f"({X.shape[0]}, {X.shape[1]})"],
        ["Final Y shape (samples × heads)",    f"({Y.shape[0]}, {Y.shape[1]})"],
    ]
    write_table(
        _tex_dir(cfg) / "03_data_alignment.tex",
        headers=["Source / quantity", "Count"], rows=rows_align,
        caption="GraphPath input pipeline: feature/label join cardinalities.",
        label="gp-data-alignment", align="lr",
    )

    rows_split = []
    for name, idx in (("Train", train_idx), ("Val", val_idx), ("Test", test_idx)):
        pos = Y.values[idx].sum(axis=0).astype(int)
        rows_split.append([
            name, len(idx),
            f"{pos[0]} ({pos[0]/len(idx):.1%})",
            f"{pos[1]} ({pos[1]/len(idx):.1%})",
            f"{pos[2]} ({pos[2]/len(idx):.1%})",
        ])
    write_table(
        _tex_dir(cfg) / "03_data_splits.tex",
        headers=["Split", "Samples", "TMT+", "RT+", "OS+"], rows=rows_split,
        caption="GraphPath: stratified 80/10/10 train/val/test split.",
        label="gp-data-splits",
    )

    rows_head = []
    for h, w in zip(cfg.head_names, pw):
        idx_h = cfg.head_names.index(h)
        pos = int(Y.values[train_idx, idx_h].sum())
        neg = len(train_idx) - pos
        rows_head.append([h, pos, neg, f"{pos/(pos+neg):.1%}", float(w)])
    write_table(
        _tex_dir(cfg) / "03_head_distribution.tex",
        headers=["Head", "Pos (train)", "Neg (train)", "Prevalence", "pos\\_weight"],
        rows=rows_head,
        caption="GraphPath head balance and per-head BCE positive-class weight.",
        label="gp-head-weights",
    )
    print(f"  wrote LaTeX tables → {_tex_dir(cfg).relative_to(cfg.project_root)}/")
    return {"n_train": len(train_idx), "n_val": len(val_idx), "n_test": len(test_idx)}


# ── Phase 4: Training ────────────────────────────────────────────────────
def _loaders_from_splits(cfg: Config):
    blob = np.load(_splits_path(cfg), allow_pickle=True)
    Xn = blob["X"]; Y = blob["Y"]; pw = blob["pos_weight"]

    Xt = torch.from_numpy(Xn).float()
    Yt = torch.from_numpy(Y).float()

    def _dl(idx, shuffle):
        ds = torch.utils.data.TensorDataset(Xt[idx], Yt[idx])
        return torch.utils.data.DataLoader(
            ds, batch_size=cfg.batch_size, shuffle=shuffle,
            generator=torch.Generator().manual_seed(cfg.seed) if shuffle else None,
        )

    return {
        "train": _dl(blob["train_idx"], True),
        "val":   _dl(blob["val_idx"], False),
        "test":  _dl(blob["test_idx"], False),
        "splits": {
            "train": blob["train_idx"], "val": blob["val_idx"], "test": blob["test_idx"],
        },
    }, pw


def phase_train(cfg: Config) -> dict:
    print("PHASE 4: Training")
    if not _splits_path(cfg).exists():
        raise SystemExit(f"Run phase data first — missing {_splits_path(cfg)}")
    with open(_reactome_path(cfg), "rb") as f:
        reactome = pickle.load(f)

    set_seed(cfg.seed)
    device = _device()
    print(f"  device                         : {device}")

    loaders, pw = _loaders_from_splits(cfg)
    model = GraphPath(
        n_pathways=len(reactome["input_ids"]),
        adjacency=reactome["adjacency"],
        embed_dim=cfg.embed_dim,
        n_heads=cfg.n_heads,
        n_outputs=len(cfg.head_names),
        dropout=cfg.dropout,
    ).to(device)
    print(f"  pathways                       : {model.n_pathways}")
    print(f"  trainable params               : {model.trainable_parameter_count():,}")

    res = fit(model, loaders, pw, cfg, device=device, verbose=True)

    blob = np.load(_splits_path(cfg), allow_pickle=True)
    torch.save({
        "state_dict": model.state_dict(),
        "n_pathways": model.n_pathways,
        "n_heads": cfg.n_heads,
        "embed_dim": cfg.embed_dim,
        "head_names": list(cfg.head_names),
        "scaler": {"mean": blob["mean"], "std": blob["std"]},
        "input_names": reactome["input_names"],
        "adjacency": reactome["adjacency"],
        "history": res["history"],
        "best_val_loss": float(res["best_val_loss"]),
    }, _model_path(cfg))
    print(f"  saved checkpoint → {_model_path(cfg).relative_to(cfg.project_root)}")

    history = res["history"]
    rows = [
        ["Pathway nodes", model.n_pathways],
        ["Adjacency edges", reactome["n_edges_undirected"]],
        ["Embedding dim (F$'$)", cfg.embed_dim],
        ["Attention heads (K)", cfg.n_heads],
        ["Activation", "ELU"],
        ["Optimizer", "SGD (momentum 0.9)"],
        ["Learning rate (initial)", f"{cfg.lr:.2f}"],
        ["Weight decay (L2)", f"{cfg.weight_decay:.2f}"],
        ["Dropout", cfg.dropout],
        ["Batch size", cfg.batch_size],
        ["Epochs run", len(history["train_loss"])],
        ["Best validation loss", f"{res['best_val_loss']:.4f}"],
        ["Final training loss", f"{history['train_loss'][-1]:.4f}"],
        ["Final validation loss", f"{history['val_loss'][-1]:.4f}"],
        ["Final learning rate", f"{history['lr'][-1]:.1e}"],
        ["Trainable parameters", f"{model.trainable_parameter_count():,}"],
    ]
    write_table(
        _tex_dir(cfg) / "04_training_summary.tex",
        headers=["Setting", "Value"], rows=rows,
        caption=(
            "GraphPath configuration and end-of-run losses. Hyperparameters "
            "follow Ma \\& Wang 2024 \\S2.6; class-weighted BCE replaces the "
            "paper's single-output BCE for our 3-head multi-label setting."
        ),
        label="gp-training", align="lr",
    )
    print(f"  wrote LaTeX table → {(_tex_dir(cfg) / '04_training_summary.tex').relative_to(cfg.project_root)}")
    return res


# ── Phase 5: Evaluation ──────────────────────────────────────────────────
def phase_evaluate(cfg: Config) -> dict:
    print("PHASE 5: Evaluation")
    if not _model_path(cfg).exists():
        raise SystemExit(f"Run phase train first — missing {_model_path(cfg)}")
    ckpt = torch.load(_model_path(cfg), map_location="cpu", weights_only=False)

    model = GraphPath(
        n_pathways=ckpt["n_pathways"],
        adjacency=ckpt["adjacency"],
        embed_dim=ckpt["embed_dim"],
        n_heads=ckpt["n_heads"],
        n_outputs=len(cfg.head_names),
        dropout=cfg.dropout,
    )
    model.load_state_dict(ckpt["state_dict"])
    device = _device()
    model = model.to(device)

    loaders, _ = _loaders_from_splits(cfg)
    val_p, val_y = predict(model, loaders["val"], device=device)
    test_p, test_y = predict(model, loaders["test"], device=device)

    # F1-optimal decision thresholds learned on the validation fold are
    # applied to the test fold (paper PATH §4.3 Eq. 27). AUROC/AUPRC
    # remain threshold-free.
    thresholds = find_best_threshold(val_p, val_y, cfg.head_names)
    val_m  = metrics(val_p,  val_y,  cfg.head_names, thresholds=thresholds)
    test_m = metrics(test_p, test_y, cfg.head_names, thresholds=thresholds)
    print_report("VAL", val_m)
    print_report("TEST", test_m)

    with open(_results_path(cfg), "w") as f:
        json.dump({
            "val": val_m, "test": test_m,
            "best_val_loss": float(ckpt["best_val_loss"]),
            "head_names": list(cfg.head_names),
        }, f, indent=2)
    print(f"  saved results → {_results_path(cfg).relative_to(cfg.project_root)}")

    # Per-sample test predictions for paired bootstrap significance testing.
    np.savez(
        cfg.artifacts_dir / "preds.npz",
        test_p=np.asarray(test_p), test_y=np.asarray(test_y),
        thresholds=np.asarray([thresholds[h] for h in cfg.head_names]),
        head_names=np.asarray(list(cfg.head_names)),
    )

    rows = []
    for split_name, m in (("Val", val_m), ("Test", test_m)):
        for head in cfg.head_names:
            v = m[head]; cm = v["cm"]
            rows.append([head, split_name, v["auroc"], v["auprc"],
                         v["f1"], v["accuracy"],
                         cm["tn"], cm["fp"], cm["fn"], cm["tp"]])
    write_table(
        _tex_dir(cfg) / "05_metrics.tex",
        headers=["Head", "Split", "AUROC", "AUPRC", "F1", "Acc",
                 "TN", "FP", "FN", "TP"],
        rows=rows,
        caption=(
            "Per-head classification performance of GraphPath on the validation "
            "and held-out test splits. AUROC/AUPRC are threshold-free; confusion "
            "matrix counts are reported at a 0.5 probability threshold."
        ),
        label="gp-metrics",
    )
    print(f"  wrote LaTeX table → {(_tex_dir(cfg) / '05_metrics.tex').relative_to(cfg.project_root)}")

    # ── Pathway importance (gradient × input attribution) ───────────
    from binn.interpret import compute_all_heads, save_importance
    # Re-load splits to get the test tensor in input-space, then
    # compute |x · ∂prob/∂x| per pathway per head (gradient × input,
    # a SHAP-analogous post-hoc attribution).
    blob = np.load(_splits_path(cfg), allow_pickle=True)
    X_test = torch.from_numpy(blob["X"][blob["test_idx"]]).float().to(device)
    head_importance = compute_all_heads(model, X_test, cfg.head_names)
    imp_path = cfg.artifacts_dir / "importance.npz"
    save_importance(imp_path,
                    head_importance,
                    pathway_names=ckpt["input_names"])
    print(f"  saved pathway importance → "
          f"{imp_path.relative_to(cfg.project_root)}")
    return {"val": val_m, "test": test_m,
            "importance": head_importance}


# ── CLI ──────────────────────────────────────────────────────────────────
PHASES = {
    "env":      phase_env,
    "reactome": phase_reactome,
    "data":     phase_data,
    "train":    phase_train,
    "evaluate": phase_evaluate,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=list(PHASES.keys()) + ["all"],
                   nargs="?", default="all")
    p.add_argument("--cohort", default="breast",
                   choices=sorted(COHORT_FILES.keys()),
                   help="Which TCGA cohort to train on.")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--epochs", type=int, default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = Config(cohort=args.cohort)
    if args.epochs is not None:
        cfg.max_epochs = args.epochs
    if args.smoke:
        cfg.max_epochs = 5
        cfg.patience = 5
    set_seed(cfg.seed)
    print(f"cohort           : {cfg.cohort}")
    print(f"scores csv       : {cfg.scores_csv}")
    print(f"mapping csv      : {cfg.mapping_csv}")
    print(f"artifacts dir    : {cfg.artifacts_dir}")

    if args.phase == "all":
        for name in ("env", "reactome", "data", "train", "evaluate"):
            PHASES[name](cfg)
            print()
    else:
        PHASES[args.phase](cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
