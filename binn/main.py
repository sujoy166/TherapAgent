"""Phase-driven pipeline runner.

Each phase reads checkpoints from the previous phase, performs its work, then
writes:
  * a binary checkpoint  → `binn/artifacts/`
  * a LaTeX table        → `binn/artifacts/tex/`

Usage:
    python -m binn.main reactome   # build Reactome hierarchy (network access)
    python -m binn.main data       # decode labels, build splits, scale
    python -m binn.main train      # train the multi-head BINN
    python -m binn.main evaluate   # final metrics on val + test
    python -m binn.main all        # run reactome → data → train → evaluate

Each phase is also wrapped by a shell script in `binn/scripts/`.
"""
from __future__ import annotations

import argparse
import json
import pickle
import random
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch

from .config import COHORT_FILES, Config
from .data import (
    alignment_summary, decode_stage, fit_standardizer, list_pathway_columns,
    load_aligned, positive_weights, stratified_split,
)
from .evaluate import find_best_threshold, metrics, predict, print_report
from .model import BINN
from .reactome import assemble
from .reporting import write_table
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


def _reactome_path(cfg: Config) -> Path:
    return cfg.artifacts_dir / "reactome.pkl"


def _splits_path(cfg: Config) -> Path:
    return cfg.artifacts_dir / "splits.npz"


def _model_path(cfg: Config) -> Path:
    return cfg.artifacts_dir / "binn.pt"


def _results_path(cfg: Config) -> Path:
    return cfg.artifacts_dir / "results.json"


def _tex_dir(cfg: Config) -> Path:
    d = cfg.artifacts_dir / "tex"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ── Phase 1: Environment summary ─────────────────────────────────────────
def phase_env(cfg: Config) -> dict:
    print("PHASE 1: Environment")
    rows = [["Python", sys.version.split()[0]]]
    for mod_name in ("numpy", "pandas", "scipy", "sklearn", "torch", "joblib"):
        try:
            mod = __import__(mod_name)
            rows.append([mod_name, getattr(mod, "__version__", "?")])
        except Exception:  # noqa: BLE001
            rows.append([mod_name, "MISSING"])

    for label, value in rows:
        print(f"  {label:<10} : {value}")

    tex_path = _tex_dir(cfg) / "01_environment.tex"
    write_table(
        tex_path,
        headers=["Package", "Version"],
        rows=rows,
        caption=(
            "Python environment used to build, train, and evaluate the "
            "Reactome-informed BINN."
        ),
        label="binn-env",
        align="lr",
    )
    print(f"  wrote LaTeX table → {tex_path.relative_to(cfg.project_root)}")
    return {"rows": rows}


# ── Phase 2: Reactome hierarchy ──────────────────────────────────────────
def phase_reactome(cfg: Config) -> dict:
    print("PHASE 2: Reactome hierarchy")
    pathway_cols = list_pathway_columns(cfg.scores_csv)
    print(f"  pathway rows in scores CSV     : {len(pathway_cols)}")

    reactome = assemble(
        pathway_names=pathway_cols,
        cache_dir=cfg.cache_dir,
        pathways_url=cfg.pathways_url,
        relations_url=cfg.relations_url,
        species_prefix=cfg.species_prefix,
        n_hidden_layers=cfg.n_hidden_layers,
    )

    print(f"  matched pathway nodes          : "
          f"{len(reactome['input_ids'])} / {len(pathway_cols)} "
          f"({reactome['n_missing']} missing)")
    for i, layer in enumerate(reactome["layers"]):
        print(f"  layer {i}                       : {len(layer)} nodes")
    for i, m in enumerate(reactome["masks"]):
        print(f"  mask  {i}→{i+1}                  : {m.shape} "
              f"(active edges={int(m.sum())})")

    out_path = _reactome_path(cfg)
    with open(out_path, "wb") as f:
        pickle.dump(reactome, f)
    print(f"  saved checkpoint → {out_path.relative_to(cfg.project_root)}")

    # LaTeX summary
    rows = []
    for i, (layer, names) in enumerate(zip(reactome["layers"], reactome["layer_names"])):
        if i < len(reactome["masks"]):
            edges = int(reactome["masks"][i].sum())
        else:
            edges = "—"
        sample = "; ".join(names[:3]) + ("…" if len(names) > 3 else "")
        rows.append([f"L{i}", len(layer), edges, sample])

    tex_path = _tex_dir(cfg) / "02_reactome_layers.tex"
    write_table(
        tex_path,
        headers=["Layer", "Nodes", "Outgoing edges", "Example nodes (top 3)"],
        rows=rows,
        caption=(
            f"Reactome-derived hierarchy used by the BINN. "
            f"{len(reactome['input_ids'])} of {len(pathway_cols)} input pathways "
            f"mapped to Reactome IDs ({reactome['n_missing']} unmatched)."
        ),
        label="binn-reactome-layers",
        align="lrrl",
    )
    print(f"  wrote LaTeX table → {tex_path.relative_to(cfg.project_root)}")
    return reactome


# ── Phase 3: Data preparation ────────────────────────────────────────────
def phase_data(cfg: Config) -> dict:
    print("PHASE 3: Data preparation")
    if not _reactome_path(cfg).exists():
        raise SystemExit(
            "Phase 2 (reactome) must run first — missing "
            f"{_reactome_path(cfg).relative_to(cfg.project_root)}"
        )
    with open(_reactome_path(cfg), "rb") as f:
        reactome = pickle.load(f)

    align = alignment_summary(cfg.scores_csv, cfg.mapping_csv)
    print(f"  scores.csv samples             : {align['scores_samples']}")
    print(f"  scores.csv pathways            : {align['scores_pathways']}")
    print(f"  mapping.csv labeled samples    : {align['label_samples']}")
    print(f"  intersection (used by model)   : {align['intersect_samples']}")

    X, Y, stage = load_aligned(cfg.scores_csv, cfg.mapping_csv,
                               reactome["input_names"])
    print(f"  X shape                        : {X.shape}")
    print(f"  Y shape                        : {Y.shape}")

    train_idx, val_idx, test_idx = stratified_split(
        X, Y, stage,
        val_frac=cfg.val_frac, test_frac=cfg.test_frac, seed=cfg.seed,
    )
    mu, sd = fit_standardizer(X.values[train_idx])
    Xn = (X.values - mu) / sd
    pw = positive_weights(Y.values[train_idx])

    print(f"  split (n)                      : "
          f"train={len(train_idx)}  val={len(val_idx)}  test={len(test_idx)}")
    print(f"  pos weights (train)            : "
          + ", ".join(f"{h}={w:.2f}" for h, w in zip(cfg.head_names, pw)))

    np.savez(
        _splits_path(cfg),
        X=Xn.astype(np.float32),
        Y=Y.values.astype(np.float32),
        train_idx=train_idx, val_idx=val_idx, test_idx=test_idx,
        mean=mu, std=sd, pos_weight=pw,
        sample_ids=np.array(X.index.tolist()),
        stage=stage.values,
    )
    print(f"  saved checkpoint → {_splits_path(cfg).relative_to(cfg.project_root)}")

    # LaTeX: data sources and the join cardinalities used to build X / Y
    rows_align = [
        ["pathway\\_scores.csv – samples",   align["scores_samples"]],
        ["pathway\\_scores.csv – pathways",  align["scores_pathways"]],
        ["pathway\\_phenotype\\_mapping.csv – labeled samples", align["label_samples"]],
        ["Reactome-matched input pathways", len(reactome["input_names"])],
        ["Samples used by model (intersection)", align["intersect_samples"]],
        ["Final X shape (samples × pathways)", f"({X.shape[0]}, {X.shape[1]})"],
        ["Final Y shape (samples × heads)",    f"({Y.shape[0]}, {Y.shape[1]})"],
    ]
    write_table(
        _tex_dir(cfg) / "03_data_alignment.tex",
        headers=["Source / quantity", "Count"],
        rows=rows_align,
        caption=(
            "Source CSVs and join cardinalities. Features come from "
            "\\texttt{pathway\\_scores.csv} (transposed to samples \\(\\times\\) "
            "pathways); labels come from \\texttt{pathway\\_phenotype\\_mapping.csv}. "
            "The model trains on the sample-wise intersection."
        ),
        label="binn-data-alignment",
        align="lr",
    )

    # LaTeX: split sizes + per-head positives per split
    def _pos(idx):
        return Y.values[idx].sum(axis=0).astype(int)

    rows_split = []
    for name, idx in (("Train", train_idx), ("Val", val_idx), ("Test", test_idx)):
        pos = _pos(idx)
        rows_split.append([
            name, len(idx),
            f"{pos[0]} ({pos[0]/len(idx):.1%})",
            f"{pos[1]} ({pos[1]/len(idx):.1%})",
            f"{pos[2]} ({pos[2]/len(idx):.1%})",
        ])
    write_table(
        _tex_dir(cfg) / "03_data_splits.tex",
        headers=["Split", "Samples", "TMT+", "RT+", "OS+"],
        rows=rows_split,
        caption=(
            "Stratified 70/15/15 split of the TCGA-BRCA pathway/phenotype cohort. "
            "Per-head positive counts (and prevalence) reported per split."
        ),
        label="binn-data-splits",
    )

    # LaTeX: head class weighting
    rows_head = []
    for h, w in zip(cfg.head_names, pw):
        pos = int(Y.values[train_idx, cfg.head_names.index(h)].sum())
        neg = len(train_idx) - pos
        rows_head.append([h, pos, neg, f"{pos/(pos+neg):.1%}", float(w)])
    write_table(
        _tex_dir(cfg) / "03_head_distribution.tex",
        headers=["Head", "Pos (train)", "Neg (train)", "Prevalence", "pos\\_weight"],
        rows=rows_head,
        caption=(
            "Per-head label distribution in the training fold and the "
            "resulting positive-class loss weights (\\#neg/\\#pos, clipped to [0.1, 20])."
        ),
        label="binn-head-weights",
    )
    print(f"  wrote LaTeX tables → {_tex_dir(cfg).relative_to(cfg.project_root)}/")
    return {"n_train": len(train_idx), "n_val": len(val_idx), "n_test": len(test_idx)}


# ── Phase 4: Training ────────────────────────────────────────────────────
def _loaders_from_splits(cfg: Config) -> Tuple[dict, np.ndarray]:
    blob = np.load(_splits_path(cfg), allow_pickle=True)
    Xn = blob["X"]
    Y = blob["Y"]
    pw = blob["pos_weight"]

    Xt = torch.from_numpy(Xn).float()
    Yt = torch.from_numpy(Y).float()

    def _dl(idx, shuffle):
        ds = torch.utils.data.TensorDataset(Xt[idx], Yt[idx])
        return torch.utils.data.DataLoader(
            ds, batch_size=cfg.batch_size, shuffle=shuffle,
            generator=torch.Generator().manual_seed(cfg.seed) if shuffle else None,
        )

    loaders = {
        "train": _dl(blob["train_idx"], True),
        "val":   _dl(blob["val_idx"], False),
        "test":  _dl(blob["test_idx"], False),
        "splits": {
            "train": blob["train_idx"],
            "val":   blob["val_idx"],
            "test":  blob["test_idx"],
        },
    }
    return loaders, pw


def phase_train(cfg: Config) -> dict:
    print("PHASE 4: Training")
    if not _splits_path(cfg).exists():
        raise SystemExit(
            "Phase 3 (data) must run first — missing "
            f"{_splits_path(cfg).relative_to(cfg.project_root)}"
        )
    with open(_reactome_path(cfg), "rb") as f:
        reactome = pickle.load(f)

    set_seed(cfg.seed)
    device = _device()
    print(f"  device                         : {device}")

    loaders, pw = _loaders_from_splits(cfg)
    model = BINN(
        masks=reactome["masks"],
        n_heads=len(cfg.head_names),
        dropout=cfg.dropout,
    ).to(device)
    print(f"  layer sizes                    : {model.layer_sizes}")
    print(f"  active params                  : {model.trainable_parameter_count():,}")

    res = fit(model, loaders, pw, cfg, device=device, verbose=True)

    blob = np.load(_splits_path(cfg), allow_pickle=True)
    torch.save({
        "state_dict": model.state_dict(),
        "layer_sizes": model.layer_sizes,
        "head_names": list(cfg.head_names),
        "scaler": {"mean": blob["mean"], "std": blob["std"]},
        "input_names": reactome["input_names"],
        "layer_names": reactome["layer_names"],
        "history": res["history"],
        "best_val_loss": float(res["best_val_loss"]),
    }, _model_path(cfg))
    print(f"  saved checkpoint → {_model_path(cfg).relative_to(cfg.project_root)}")

    # LaTeX training summary
    history = res["history"]
    n_epochs = len(history["train_loss"])
    rows = [
        ["Hidden layers", cfg.n_hidden_layers],
        ["Layer sizes", ", ".join(str(s) for s in model.layer_sizes)],
        ["Active parameters", f"{model.trainable_parameter_count():,}"],
        ["Optimizer", "Adam"],
        ["Learning rate (initial)", f"{cfg.lr:.0e}"],
        ["Weight decay (L2)", f"{cfg.weight_decay:.0e}"],
        ["Dropout", cfg.dropout],
        ["Batch size", cfg.batch_size],
        ["Epochs run", n_epochs],
        ["Best validation loss", f"{res['best_val_loss']:.4f}"],
        ["Final training loss", f"{history['train_loss'][-1]:.4f}"],
        ["Final validation loss", f"{history['val_loss'][-1]:.4f}"],
        ["Final learning rate", f"{history['lr'][-1]:.1e}"],
    ]
    write_table(
        _tex_dir(cfg) / "04_training_summary.tex",
        headers=["Setting", "Value"],
        rows=rows,
        caption=(
            "Training configuration and end-of-run loss values for the "
            "Reactome-informed BINN. Hyperparameters follow Hartman et al. (2023) "
            "with class-weighted BCE for the multi-label heads."
        ),
        label="binn-training",
        align="lr",
    )
    print(f"  wrote LaTeX table → {(_tex_dir(cfg) / '04_training_summary.tex').relative_to(cfg.project_root)}")
    return res


# ── Phase 5: Evaluation ──────────────────────────────────────────────────
def phase_evaluate(cfg: Config) -> dict:
    print("PHASE 5: Evaluation")
    if not _model_path(cfg).exists():
        raise SystemExit(
            "Phase 4 (train) must run first — missing "
            f"{_model_path(cfg).relative_to(cfg.project_root)}"
        )
    with open(_reactome_path(cfg), "rb") as f:
        reactome = pickle.load(f)
    ckpt = torch.load(_model_path(cfg), map_location="cpu", weights_only=False)
    device = _device()

    model = BINN(
        masks=reactome["masks"],
        n_heads=len(cfg.head_names),
        dropout=cfg.dropout,
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])

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
            "layer_sizes": ckpt["layer_sizes"],
        }, f, indent=2)
    print(f"  saved results → {_results_path(cfg).relative_to(cfg.project_root)}")

    # Per-sample test predictions for bootstrap significance testing.
    # At a fixed seed the test fold is identical across models, so these
    # arrays support *paired* bootstrap comparison downstream.
    np.savez(
        cfg.artifacts_dir / "preds.npz",
        test_p=np.asarray(test_p), test_y=np.asarray(test_y),
        thresholds=np.asarray([thresholds[h] for h in cfg.head_names]),
        head_names=np.asarray(list(cfg.head_names)),
    )

    # LaTeX: per-head metrics, val vs test
    rows = []
    for split_name, m in (("Val", val_m), ("Test", test_m)):
        for head in cfg.head_names:
            v = m[head]
            cm = v["cm"]
            rows.append([
                head, split_name,
                v["auroc"], v["auprc"], v["f1"], v["accuracy"],
                cm["tn"], cm["fp"], cm["fn"], cm["tp"],
            ])
    write_table(
        _tex_dir(cfg) / "05_metrics.tex",
        headers=["Head", "Split", "AUROC", "AUPRC", "F1", "Acc",
                 "TN", "FP", "FN", "TP"],
        rows=rows,
        caption=(
            "Per-head classification performance of the trained BINN on the "
            "validation and held-out test splits. Confusion matrix counts are "
            "reported at a fixed 0.5 probability threshold; AUROC/AUPRC are "
            "threshold-free."
        ),
        label="binn-metrics",
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
    p.add_argument(
        "phase",
        choices=list(PHASES.keys()) + ["all"],
        nargs="?",
        default="all",
        help="Which pipeline phase to run (default: all).",
    )
    p.add_argument("--cohort", default="breast",
                   choices=sorted(COHORT_FILES.keys()),
                   help="Which TCGA cohort to train on. "
                        "Artifacts land in binn/artifacts/<cohort>/.")
    p.add_argument("--smoke", action="store_true",
                   help="Cap epochs at 5 for a fast sanity run.")
    p.add_argument("--epochs", type=int, default=None,
                   help="Override Config.max_epochs.")
    p.add_argument("--layers", type=int, default=None,
                   help="Override Config.n_hidden_layers (rebuild from phase reactome).")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = Config(cohort=args.cohort)
    if args.epochs is not None:
        cfg.max_epochs = args.epochs
    if args.layers is not None:
        cfg.n_hidden_layers = args.layers
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
