"""Dataset loader for the TherapAgent pathway score / phenotype tables.

The two CSV files serve distinct roles:

* ``pathway_scores.csv``           — canonical **feature matrix**:
                                      rows = 1706 Reactome pathways,
                                      columns = 1218 TCGA-BRCA samples.
* ``pathway_phenotype_mapping.csv``— **label table**:
                                      rows = 618 samples with valid clinical
                                      stage (subset of the 1218 in scores.csv).
                                      Pathway-score columns are redundant
                                      (we ignore them here).

The labeled cohort is therefore the intersection of (scores.csv columns)
∩ (mapping.csv rows) = 618 samples. Anything in scores.csv without a label
is unusable for supervised training and is dropped.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset


def decode_stage(stage: pd.Series) -> pd.DataFrame:
    """Stage = 4*TMT + 2*RT + 1*OS≥180d  →  3-column DataFrame of {0,1}."""
    s = stage.astype(int)
    return pd.DataFrame({
        "TMT": (s // 4) % 2,
        "RT":  (s // 2) % 2,
        "OS":  s % 2,
    }, index=stage.index)


def list_pathway_columns(scores_csv: Path) -> List[str]:
    """Pathway names = row index of ``scores_csv`` (its first column)."""
    df = pd.read_csv(scores_csv, usecols=[0])
    return df.iloc[:, 0].astype(str).tolist()


def alignment_summary(scores_csv: Path, mapping_csv: Path) -> dict:
    """Compute the feature/label join cardinalities (for reporting).

    Reads only the headers (no data) so it's O(file-size of one line).
    """
    score_samples = pd.read_csv(scores_csv, nrows=0).columns.tolist()[1:]
    score_pathways = list_pathway_columns(scores_csv)
    label_samples = pd.read_csv(
        mapping_csv, usecols=["sample", "stage"], index_col=0
    ).dropna(subset=["stage"]).index.tolist()
    common = sorted(set(score_samples) & set(label_samples))
    return {
        "scores_samples": len(score_samples),
        "scores_pathways": len(score_pathways),
        "label_samples": len(label_samples),
        "intersect_samples": len(common),
        "common_sample_ids": common,
    }


def load_aligned(scores_csv: Path,
                 mapping_csv: Path,
                 restrict_pathways: List[str]
                 ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Build the (X, Y, stage) trio for supervised training.

    X — features from ``scores_csv``, transposed to (samples × pathways) and
        restricted to ``restrict_pathways`` (the Reactome-matched subset).
    Y — 3 binary heads decoded from ``mapping_csv``'s ``stage`` column.
    stage — the raw 0..7 stage code, used by the stratified splitter.

    Indexes of all three frames are aligned to the intersection of
    (scores.csv columns) ∩ (mapping.csv rows with a valid stage).
    """
    # Features ────────────────────────────────────────────────────────────
    scores = pd.read_csv(scores_csv, index_col=0)              # pathways × samples
    scores.index = scores.index.astype(str)
    scores = scores.T                                          # → samples × pathways
    scores.index.name = "sample"

    missing_p = [p for p in restrict_pathways if p not in scores.columns]
    if missing_p:
        raise ValueError(
            f"{len(missing_p)} pathways requested but absent from "
            f"{scores_csv.name}. First few: {missing_p[:3]}"
        )
    scores = scores[restrict_pathways].astype(np.float32)

    # Labels ──────────────────────────────────────────────────────────────
    labels = pd.read_csv(mapping_csv, usecols=["sample", "stage"], index_col=0)
    labels = labels.dropna(subset=["stage"])
    labels["stage"] = labels["stage"].astype(int)

    # Intersect samples ───────────────────────────────────────────────────
    common = scores.index.intersection(labels.index)
    if len(common) == 0:
        raise ValueError(
            f"No samples are present in both {scores_csv.name} and "
            f"{mapping_csv.name}"
        )
    # Deterministic order, identical across pandas versions.
    common = sorted(common)

    X = scores.loc[common]
    stage = labels.loc[common, "stage"]
    Y = decode_stage(stage).astype(np.float32)
    return X, Y, stage


def stratified_split(X: pd.DataFrame, Y: pd.DataFrame, stage: pd.Series,
                     val_frac: float, test_frac: float, seed: int):
    """Three-way split stratified on the 8-class stage where feasible.

    Falls back to non-stratified splits if any stratum has <2 samples
    (our cohort has stage==2 with a single sample).
    """
    idx = np.arange(len(X))
    s = stage.values

    counts = pd.Series(s).value_counts()
    can_stratify = (counts >= 2).all() and (counts.min() * test_frac >= 1)
    stratify1 = s if can_stratify else None
    rest_idx, test_idx = train_test_split(
        idx, test_size=test_frac, random_state=seed, stratify=stratify1
    )

    s_rest = s[rest_idx]
    counts_rest = pd.Series(s_rest).value_counts()
    val_ratio = val_frac / (1.0 - test_frac)
    can_stratify2 = (counts_rest >= 2).all() and (counts_rest.min() * val_ratio >= 1)
    stratify2 = s_rest if can_stratify2 else None
    train_idx, val_idx = train_test_split(
        rest_idx, test_size=val_ratio, random_state=seed, stratify=stratify2
    )
    return train_idx, val_idx, test_idx


def fit_standardizer(X_train: np.ndarray):
    """Per-feature mean/std on the training fold; std clamped above 1e-6."""
    mu = X_train.mean(axis=0)
    sd = X_train.std(axis=0)
    sd = np.where(sd < 1e-6, 1.0, sd)
    return mu.astype(np.float32), sd.astype(np.float32)


def make_loaders(X: pd.DataFrame, Y: pd.DataFrame, stage: pd.Series,
                 batch_size: int, val_frac: float, test_frac: float, seed: int
                 ) -> dict:
    train_idx, val_idx, test_idx = stratified_split(
        X, Y, stage, val_frac=val_frac, test_frac=test_frac, seed=seed
    )
    Xv, Yv = X.values, Y.values

    mu, sd = fit_standardizer(Xv[train_idx])
    Xn = (Xv - mu) / sd

    Xt = torch.from_numpy(Xn).float()
    Yt = torch.from_numpy(Yv).float()

    train_ds = TensorDataset(Xt[train_idx], Yt[train_idx])
    val_ds   = TensorDataset(Xt[val_idx],   Yt[val_idx])
    test_ds  = TensorDataset(Xt[test_idx],  Yt[test_idx])

    g = torch.Generator().manual_seed(seed)
    return {
        "train": DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=g),
        "val":   DataLoader(val_ds,   batch_size=batch_size, shuffle=False),
        "test":  DataLoader(test_ds,  batch_size=batch_size, shuffle=False),
        "splits": {"train": train_idx, "val": val_idx, "test": test_idx},
        "scaler": {"mean": mu, "std": sd},
        "sample_ids": X.index.tolist(),
    }


def positive_weights(Y_train: np.ndarray) -> np.ndarray:
    """w_pos[h] = (#neg / #pos) per head; clipped to [0.1, 20] for stability."""
    pos = Y_train.sum(axis=0)
    neg = Y_train.shape[0] - pos
    w = np.where(pos > 0, neg / np.maximum(pos, 1.0), 1.0)
    return np.clip(w, 0.1, 20.0).astype(np.float32)
