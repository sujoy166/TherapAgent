"""Regenerate every per-phase LaTeX table from saved checkpoints.

Reads `<model>/artifacts/{reactome.pkl, splits.npz, results.json, <m>.pt}`
for `<model>` in `{binn, graphpath, path}` and re-emits the 7 per-phase
LaTeX tables using the updated `binn.reporting` writer, which wraps every
tabular in `\\resizebox{\\linewidth}{!}{...}` and uses `[!ht]` placement
so the typesetter can pack many tables without overlap. The 10-column
metrics table (`05_metrics.tex`) is emitted as a two-column-spanning
`table*` (paper-wide).

No training is rerun.
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from binn.reporting import write_table  # noqa: E402


HEAD_NAMES = ("TMT", "RT", "OS")


def _load(model: str) -> dict:
    art = PROJECT_ROOT / model / "artifacts"
    with open(art / "reactome.pkl", "rb") as f:
        reactome = pickle.load(f)
    blob = np.load(art / "splits.npz", allow_pickle=True)
    results = json.loads((art / "results.json").read_text())
    return {"model": model, "art": art, "reactome": reactome,
            "blob": blob, "results": results}


def _tex_dir(art: Path) -> Path:
    d = art / "tex"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ── Phase 1: env (we keep what's already there, since it's runtime-derived)


def regen_binn_reactome(ctx: dict) -> None:
    r = ctx["reactome"]
    rows = []
    for i, (layer, names) in enumerate(zip(r["layers"], r["layer_names"])):
        edges = int(r["masks"][i].sum()) if i < len(r["masks"]) else "—"
        sample = "; ".join(names[:3]) + ("…" if len(names) > 3 else "")
        rows.append([f"L{i}", len(layer), edges, sample])
    write_table(
        _tex_dir(ctx["art"]) / "02_reactome_layers.tex",
        headers=["Layer", "Nodes", "Outgoing edges", "Example nodes (top 3)"],
        rows=rows,
        caption=(
            f"Reactome-derived hierarchy used by the BINN. "
            f"{len(r['input_ids'])} of {len(r['input_ids']) + r['n_missing']} "
            f"input pathways mapped to Reactome IDs."
        ),
        label="binn-reactome-layers",
        align="lrrl",
        full_width=True,
    )


def regen_graph_reactome(ctx: dict, model_label: str, prefix: str) -> None:
    r = ctx["reactome"]
    A = r["adjacency"]
    if model_label == "graphpath":
        deg = A.sum(axis=1)
        rows = [
            ["Reactome-matched pathways", len(r["input_ids"])],
            ["Unmatched pathway names", r["n_missing"]],
            ["Adjacency type",
             "parent/child + siblings (KEGG-analogue)"],
            ["Undirected edges", r["n_edges_undirected"]],
            ["Average degree", f"{deg.mean():.2f}"],
            ["Max degree", int(deg.max())],
            ["Density (\\%)",
             f"{100 * r['n_edges_undirected'] / (len(r['input_ids']) * (len(r['input_ids']) - 1) / 2):.2f}"],
        ]
        caption = (
            "Reactome-derived pathway-pathway adjacency used by GraphPath. "
            "Edges connect parent/child pairs and siblings sharing a parent."
        )
    else:  # path
        deg = (A > 0).sum(axis=1)
        rows = [
            ["Pathways considered", len(r["input_names"]) + r["n_missing"]],
            ["Pathways with $\\ge$15 genes (matched)", len(r["input_names"])],
            ["Unmatched pathway names", r["n_missing"]],
            ["Adjacency type",
             "Jaccard of Reactome gene memberships (Eq.~2)"],
            ["Undirected edges", r["n_edges_undirected"]],
            ["Max Jaccard (after norm)", f"{r['max_jaccard']:.3f}"],
            ["Median $|G_p|$", int(r["median_pathway_size"])],
            ["Average degree", f"{deg.mean():.2f}"],
            ["Max degree", int(deg.max())],
        ]
        caption = (
            "Reactome-derived pathway-pathway Jaccard adjacency used by PATH "
            "(paper Eq.~2)."
        )
    write_table(
        _tex_dir(ctx["art"]) / "02_pathway_graph.tex",
        headers=["Property", "Value"], rows=rows,
        caption=caption,
        label=f"{prefix}-graph", align="lr",
    )


def regen_data_tables(ctx: dict, prefix: str, full_caption: str) -> None:
    blob = ctx["blob"]
    Y = blob["Y"]
    train_idx, val_idx, test_idx = blob["train_idx"], blob["val_idx"], blob["test_idx"]
    pw = blob["pos_weight"]
    r = ctx["reactome"]

    # alignment table — re-derive from scores.csv & mapping.csv headers
    import pandas as pd
    scores_csv = PROJECT_ROOT / "pathway_scores.csv"
    mapping_csv = PROJECT_ROOT / "pathway_phenotype_mapping.csv"
    score_samples = pd.read_csv(scores_csv, nrows=0).columns.tolist()[1:]
    score_pathways = pd.read_csv(scores_csv, usecols=[0]).iloc[:, 0].tolist()
    label_samples = pd.read_csv(mapping_csv, usecols=["sample", "stage"],
                                index_col=0).dropna(subset=["stage"]).index.tolist()
    common = sorted(set(score_samples) & set(label_samples))
    matched_pathways = (r["input_ids"] if "input_ids" in r and r["input_ids"]
                        else r["input_names"])

    rows_align = [
        ["pathway\\_scores.csv – samples",   len(score_samples)],
        ["pathway\\_scores.csv – pathways",  len(score_pathways)],
        ["pathway\\_phenotype\\_mapping.csv – labeled samples", len(label_samples)],
        ["Reactome-matched input pathways", len(matched_pathways)],
        ["Samples used by model (intersection)", len(common)],
        ["Final X shape (samples $\\times$ pathways)",
         f"({len(common)}, {len(matched_pathways)})"],
        ["Final Y shape (samples $\\times$ heads)",
         f"({len(common)}, {len(HEAD_NAMES)})"],
    ]
    write_table(
        _tex_dir(ctx["art"]) / "03_data_alignment.tex",
        headers=["Source / quantity", "Count"], rows=rows_align,
        caption=full_caption + " Source CSVs and join cardinalities.",
        label=f"{prefix}-data-alignment", align="lr",
    )

    # splits table
    rows_split = []
    for name, idx in (("Train", train_idx), ("Val", val_idx), ("Test", test_idx)):
        pos = Y[idx].sum(axis=0).astype(int)
        rows_split.append([
            name, int(len(idx)),
            f"{pos[0]} ({pos[0]/len(idx):.1%})",
            f"{pos[1]} ({pos[1]/len(idx):.1%})",
            f"{pos[2]} ({pos[2]/len(idx):.1%})",
        ])
    write_table(
        _tex_dir(ctx["art"]) / "03_data_splits.tex",
        headers=["Split", "Samples", "TMT+", "RT+", "OS+"], rows=rows_split,
        caption=full_caption + " Stratified train/val/test split.",
        label=f"{prefix}-data-splits",
        full_width=True,
    )

    # head distribution
    rows_head = []
    for h, w in zip(HEAD_NAMES, pw):
        idx_h = HEAD_NAMES.index(h)
        pos = int(Y[train_idx, idx_h].sum())
        neg = len(train_idx) - pos
        rows_head.append([h, pos, neg, f"{pos/(pos+neg):.1%}", float(w)])
    write_table(
        _tex_dir(ctx["art"]) / "03_head_distribution.tex",
        headers=["Head", "Pos (train)", "Neg (train)", "Prevalence",
                 "pos\\_weight"],
        rows=rows_head,
        caption=full_caption + " Per-head positive-class BCE weighting.",
        label=f"{prefix}-head-weights",
        full_width=True,
    )


def regen_training_summary(ctx: dict, prefix: str, label_caption: str) -> None:
    model = ctx["model"]
    pt_name = {"binn": "binn.pt", "graphpath": "graphpath.pt",
               "path": "path.pt"}[model]
    ckpt = torch.load(ctx["art"] / pt_name, map_location="cpu",
                      weights_only=False)
    history = ckpt["history"]

    if model == "binn":
        rows = [
            ["Hidden layers", len(ckpt["layer_sizes"]) - 1],
            ["Layer sizes", ", ".join(str(s) for s in ckpt["layer_sizes"])],
            ["Active parameters",
             "{:,}".format(ckpt.get("active_params", "—") if "active_params" in ckpt
                           else sum(ckpt["state_dict"][k].numel()
                                    for k in ckpt["state_dict"]
                                    if k.endswith(".weight")))],
            ["Optimizer", "Adam"],
            ["Learning rate (initial)", "1e-03"],
            ["Weight decay (L2)", "1e-03"],
            ["Dropout", 0.2],
            ["Batch size", 32],
            ["Epochs run", len(history["train_loss"])],
            ["Best validation loss", f"{ckpt['best_val_loss']:.4f}"],
            ["Final training loss", f"{history['train_loss'][-1]:.4f}"],
            ["Final validation loss", f"{history['val_loss'][-1]:.4f}"],
            ["Final learning rate", f"{history['lr'][-1]:.1e}"],
        ]
    elif model == "graphpath":
        n_params = sum(v.numel() for v in ckpt["state_dict"].values())
        rows = [
            ["Pathway nodes", ckpt["n_pathways"]],
            ["Embedding dim ($F'$)", ckpt["embed_dim"]],
            ["Attention heads ($K$)", ckpt["n_heads"]],
            ["Activation", "ELU"],
            ["Optimizer", "SGD (momentum 0.9)"],
            ["Learning rate (initial)", "5.0e-02"],
            ["Weight decay (L2)", "5.0e-02"],
            ["Dropout", 0.4],
            ["Batch size", 32],
            ["Epochs run", len(history["train_loss"])],
            ["Best validation loss", f"{ckpt['best_val_loss']:.4f}"],
            ["Final training loss", f"{history['train_loss'][-1]:.4f}"],
            ["Final validation loss", f"{history['val_loss'][-1]:.4f}"],
            ["Final learning rate", f"{history['lr'][-1]:.1e}"],
            ["Trainable parameters", "{:,}".format(n_params)],
        ]
    else:  # path
        n_params = sum(v.numel() for v in ckpt["state_dict"].values())
        rows = [
            ["Pathway nodes ($P$)", ckpt["n_pathways"]],
            ["Embedding dim ($d$)", ckpt["embed_dim"]],
            ["Attention heads ($H$)", ckpt["n_heads"]],
            ["Transformer layers ($L$)", ckpt["n_layers"]],
            ["Laplacian PE eigenvectors ($k$)", ckpt["laplacian_k"]],
            ["FFN expansion factor", ckpt["ffn_expansion"]],
            ["Soft-mask penalty", ckpt["soft_mask_penalty"]],
            ["Optimizer", "AdamW"],
            ["Learning rate (initial)", "1.0e-04"],
            ["Weight decay", "5.0e-04"],
            ["Gradient clip", 2.0],
            ["Dropout", 0.2],
            ["Batch size", 16],
            ["Epochs run", len(history["train_loss"])],
            ["Best validation loss", f"{ckpt['best_val_loss']:.4f}"],
            ["Final training loss", f"{history['train_loss'][-1]:.4f}"],
            ["Final validation loss", f"{history['val_loss'][-1]:.4f}"],
            ["Final learning rate", f"{history['lr'][-1]:.1e}"],
            ["Trainable parameters", "{:,}".format(n_params)],
        ]
    write_table(
        _tex_dir(ctx["art"]) / "04_training_summary.tex",
        headers=["Setting", "Value"], rows=rows,
        caption=label_caption + " end-of-run loss values.",
        label=f"{prefix}-training", align="lr",
    )


def regen_metrics(ctx: dict, prefix: str, label_caption: str) -> None:
    res = ctx["results"]
    rows = []
    for split_name, split_data in (("Val", res["val"]), ("Test", res["test"])):
        for head in HEAD_NAMES:
            v = split_data[head]; cm = v["cm"]
            rows.append([head, split_name, v["auroc"], v["auprc"],
                         v["f1"], v["accuracy"],
                         cm["tn"], cm["fp"], cm["fn"], cm["tp"]])
    write_table(
        _tex_dir(ctx["art"]) / "05_metrics.tex",
        headers=["Head", "Split", "AUROC", "AUPRC", "F1", "Acc",
                 "TN", "FP", "FN", "TP"],
        rows=rows,
        caption=label_caption + (
            " Per-head classification performance on the validation and "
            "held-out test splits. AUROC/AUPRC are threshold-free; "
            "confusion-matrix counts at the 0.5 threshold."
        ),
        label=f"{prefix}-metrics",
        full_width=True,
    )


def main() -> None:
    for model, prefix, caption in [
        ("binn", "binn", "BINN."),
        ("graphpath", "gp", "GraphPath."),
        ("path", "path", "PATH."),
    ]:
        ctx = _load(model)
        print(f">> regenerating {model}/artifacts/tex/")
        # Phase 2 — Reactome
        if model == "binn":
            regen_binn_reactome(ctx)
        else:
            regen_graph_reactome(ctx, model, prefix)
        # Phase 3 — data
        regen_data_tables(ctx, prefix, caption)
        # Phase 4 — training
        regen_training_summary(ctx, prefix, caption)
        # Phase 5 — metrics
        regen_metrics(ctx, prefix, caption)
    print(">> done.")


if __name__ == "__main__":
    main()
