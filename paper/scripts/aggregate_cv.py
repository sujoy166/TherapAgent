#!/usr/bin/env python3
"""Aggregate the unified identical-split cross-validation sweep.

Reads the per-seed snapshots written by ``slurm/20_cv.sbatch``:

    results_cv/seed<seed>/<model>__<cohort>.json        (summary metrics)
    results_cv/seed<seed>/<model>__<cohort>.preds.npz   (per-sample test preds)

and produces, under ``results_cv/``:

  * ``summary.json``   -- test AUROC/AUPRC/F1 mean and 95% CI over seeds,
                          per (model, cohort, head).
  * ``cv_table.tex``   -- LaTeX table of test AUROC (mean +/- 95% CI) with
                          the three models side by side per cohort/head.
  * ``boot_tests.tex`` -- paired-bootstrap significance tests comparing the
                          models head-to-head on the *identical* test folds.

Because every model is evaluated on byte-identical test folds at each seed,
model predictions are aligned sample-for-sample, so we use a **paired
bootstrap**: we resample held-out samples with replacement, recompute the
metric for both models on the same resampled indices, and take the
distribution of the difference. The two-sided bootstrap p-value is
``2 * min(P(delta<=0), P(delta>=0))`` and the 95% CI is the 2.5/97.5
percentiles of the resampled difference.

Only numpy + scikit-learn are required.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import zlib
from collections import defaultdict

import numpy as np
from sklearn.metrics import (average_precision_score, f1_score,
                             roc_auc_score)

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CV_DIR = os.path.join(REPO, "results_cv")
MODELS = ["binn", "graphpath", "path"]
MODEL_TeX = {"binn": "BINN", "graphpath": "GraphPath", "path": "PATH"}
COHORTS = ["breast", "lung", "prostate", "head_neck", "thyroid"]
COHORT_TeX = {"breast": "Breast", "lung": "Lung", "prostate": "Prostate",
              "head_neck": "Head \\& Neck", "thyroid": "Thyroid"}
HEADS = ["TMT", "RT", "OS"]
N_BOOT = 10000
MIN_MINORITY = 15       # pooled minority-class count below which a cell is
                        # flagged underpowered (bootstrap p unreliable)
BOOT_SEED = 12345       # base seed; each contrast draws a deterministic
                        # sub-seed so p-values are stable across reruns.


# ----------------------------------------------------------------------
# metric helpers
# ----------------------------------------------------------------------
def _auroc(y, s):
    # Fast rank-based AUROC (Mann-Whitney U); avoids sklearn overhead in the
    # inner bootstrap loop. Falls back to NaN for single-class samples.
    y = np.asarray(y)
    npos = int(y.sum()); nneg = y.size - npos
    if npos == 0 or nneg == 0:
        return np.nan
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(y.size, dtype=float)
    ranks[order] = np.arange(1, y.size + 1)
    # average ranks for ties
    s_sorted = s[order]
    i = 0
    while i < y.size:
        j = i
        while j + 1 < y.size and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    return (ranks[y == 1].sum() - npos * (npos + 1) / 2.0) / (npos * nneg)


def _auprc(y, s):
    return average_precision_score(y, s) if len(np.unique(y)) == 2 else np.nan


def _f1(y, pred):
    return f1_score(y, pred, zero_division=0)


def _metric(kind, y, s, pred):
    if kind == "auroc":
        return _auroc(y, s)
    if kind == "auprc":
        return _auprc(y, s)
    return _f1(y, pred)


def mean_ci(vals):
    """Mean and 95% normal-approx CI over seeds (nan-aware)."""
    v = np.asarray([x for x in vals if not np.isnan(x)], dtype=float)
    if v.size == 0:
        return np.nan, np.nan, np.nan
    m = float(v.mean())
    if v.size == 1:
        return m, np.nan, np.nan
    se = float(v.std(ddof=1) / np.sqrt(v.size))
    return m, m - 1.96 * se, m + 1.96 * se


# ----------------------------------------------------------------------
# load snapshots
# ----------------------------------------------------------------------
def load_summary(seeds):
    """metrics[(model,cohort,head,metric)] = [per-seed test value]."""
    metrics = defaultdict(list)
    for seed in seeds:
        for model in MODELS:
            for cohort in COHORTS:
                fp = os.path.join(CV_DIR, f"seed{seed}", f"{model}__{cohort}.json")
                if not os.path.exists(fp):
                    continue
                with open(fp) as f:
                    blob = json.load(f)
                test = blob.get("test", {})
                for head in HEADS:
                    hv = test.get(head, {})
                    for met in ("auroc", "auprc", "f1"):
                        metrics[(model, cohort, head, met)].append(
                            float(hv.get(met, np.nan)))
    return metrics


def load_preds(seeds):
    """preds[(model,cohort,head)] = list of (y, score, pred) per seed."""
    preds = defaultdict(list)
    for seed in seeds:
        for model in MODELS:
            for cohort in COHORTS:
                fp = os.path.join(CV_DIR, f"seed{seed}",
                                  f"{model}__{cohort}.preds.npz")
                if not os.path.exists(fp):
                    continue
                d = np.load(fp, allow_pickle=True)
                P, Y = np.asarray(d["test_p"]), np.asarray(d["test_y"])
                thr = np.asarray(d["thresholds"], dtype=float)
                names = [str(x) for x in d["head_names"]]
                for h, name in enumerate(names):
                    if name not in HEADS:
                        continue
                    y = Y[:, h].astype(int)
                    s = P[:, h].astype(float)
                    pred = (s >= thr[h]).astype(int)
                    preds[(model, cohort, name)].append((y, s, pred))
    return preds


# ----------------------------------------------------------------------
# paired bootstrap between two models on identical folds
# ----------------------------------------------------------------------
def paired_bootstrap(preds, model_a, model_b, cohort, head, kind="auroc"):
    """Pool the per-seed identical test folds, then paired-bootstrap the
    metric difference (model_a - model_b)."""
    sa = preds.get((model_a, cohort, head), [])
    sb = preds.get((model_b, cohort, head), [])
    if not sa or not sb or len(sa) != len(sb):
        return None
    # Concatenate seed-by-seed; y must match (identical fold per seed).
    ya, sca, pra, yb, scb, prb = [], [], [], [], [], []
    for (y1, s1, p1), (y2, s2, p2) in zip(sa, sb):
        if len(y1) != len(y2) or not np.array_equal(y1, y2):
            # Folds not aligned for this seed; skip defensively.
            continue
        ya.append(y1); sca.append(s1); pra.append(p1)
        yb.append(y2); scb.append(s2); prb.append(p2)
    if not ya:
        return None
    y = np.concatenate(ya)
    sca, pra = np.concatenate(sca), np.concatenate(pra)
    scb, prb = np.concatenate(scb), np.concatenate(prb)
    n = len(y)
    n_pos = int(y.sum())
    n_min = min(n_pos, n - n_pos)          # minority-class count in pooled fold
    underpowered = n_min < MIN_MINORITY

    # Deterministic per-contrast RNG so p-values are reproducible regardless
    # of evaluation order or metric-implementation changes.
    key = f"{model_a}/{model_b}/{cohort}/{head}/{kind}".encode()
    rng = np.random.default_rng(BOOT_SEED + zlib.crc32(key))

    obs = _metric(kind, y, sca, pra) - _metric(kind, y, scb, prb)
    diffs = np.empty(N_BOOT)
    valid = 0
    for b in range(N_BOOT):
        idx = rng.integers(0, n, n)
        yb_ = y[idx]
        if len(np.unique(yb_)) < 2:
            diffs[b] = np.nan
            continue
        da = _metric(kind, yb_, sca[idx], pra[idx])
        db = _metric(kind, yb_, scb[idx], prb[idx])
        diffs[b] = da - db
        valid += 1
    d = diffs[~np.isnan(diffs)]
    if d.size == 0:
        return None
    lo, hi = np.percentile(d, [2.5, 97.5])
    p_gt = float(np.mean(d >= 0))
    p_lt = float(np.mean(d <= 0))
    pval = float(min(1.0, 2 * min(p_gt, p_lt)))
    return {"kind": kind, "obs_diff": float(obs), "ci_lo": float(lo),
            "ci_hi": float(hi), "pval": pval, "n": int(n),
            "n_min": int(n_min), "underpowered": bool(underpowered),
            "n_boot": int(d.size)}


# ----------------------------------------------------------------------
# LaTeX writers
# ----------------------------------------------------------------------
def _fmt(m, lo, hi):
    if np.isnan(m):
        return "--"
    if np.isnan(lo):
        return f"{m:.2f}"
    return f"{m:.2f}\\,[{lo:.2f},{hi:.2f}]"


def write_cv_table(metrics, seeds, path):
    lines = [
        "% Auto-generated by paper/scripts/aggregate_cv.py",
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Held-out test AUROC (mean\\,[95\\% CI] over "
        f"{len(seeds)} identical-split repeats) for the three architectures "
        "on every cohort and clinical head. At a fixed repeat all three "
        "models are evaluated on byte-identical 80/10/10 stratified folds.}",
        "\\label{tab:cv}",
        "\\small\\setlength{\\tabcolsep}{4pt}",
        "\\begin{tabular}{@{}llccc@{}}",
        "\\toprule",
        "Cohort & Head & BINN & GraphPath & PATH \\\\",
        "\\midrule",
    ]
    for cohort in COHORTS:
        has_data = any(metrics.get((mo, cohort, he, "auroc"))
                       for mo in MODELS for he in HEADS)
        if not has_data:
            continue
        for hi_, head in enumerate(HEADS):
            cells = []
            for model in MODELS:
                m, lo, hi = mean_ci(metrics.get((model, cohort, head, "auroc"), []))
                cells.append(_fmt(m, lo, hi))
            label = COHORT_TeX[cohort] if hi_ == 0 else ""
            lines.append(f"{label} & {head} & " + " & ".join(cells) + " \\\\")
        lines.append("\\midrule")
    lines[-1] = "\\bottomrule"
    lines += ["\\end{tabular}", "\\end{table*}", ""]
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"  wrote {os.path.relpath(path, REPO)}")


def write_boot_table(preds, path):
    """Paired-bootstrap AUROC comparison of the best vs each other model,
    per cohort/head. We report every pairwise contrast."""
    pairs = [("path", "binn"), ("path", "graphpath"), ("graphpath", "binn")]
    lines = [
        "% Auto-generated by paper/scripts/aggregate_cv.py",
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Paired-bootstrap significance tests on test AUROC "
        "($10^4$ resamples of the identical held-out folds). "
        "$\\Delta$AUROC is (model A $-$ model B); a 95\\% CI excluding 0 "
        "(equivalently $p<0.05$, shown in bold) indicates a significant "
        "difference. $^{\\dagger}$ marks underpowered cells with fewer than "
        f"{MIN_MINORITY} pooled minority-class test samples (thyroid and the "
        "near-universal-prevalence OS cohorts), whose bootstrap estimates are "
        "unstable and are not interpreted as significant.}",
        "\\label{tab:boot}",
        "\\small\\setlength{\\tabcolsep}{4pt}",
        "\\begin{tabular}{@{}llccc@{}}",
        "\\toprule",
        "Cohort & Head & Contrast (A vs B) & $\\Delta$AUROC [95\\% CI] & $p$ \\\\",
        "\\midrule",
    ]
    for cohort in COHORTS:
        block = []
        for head in HEADS:
            for a, b in pairs:
                r = paired_bootstrap(preds, a, b, cohort, head, "auroc")
                if r is None:
                    continue
                # Underpowered cells are never rendered as significant.
                signif = (r["pval"] < 0.05) and not r["underpowered"]
                sig = "\\textbf{" if signif else ""
                end = "}" if signif else ""
                p_str = "<0.001" if r["pval"] < 1e-3 else f"{r['pval']:.3f}"
                dag = "$^{\\dagger}$" if r["underpowered"] else ""
                block.append(
                    f"{COHORT_TeX[cohort]} & {head} & "
                    f"{MODEL_TeX[a]} vs {MODEL_TeX[b]} & "
                    f"{sig}{r['obs_diff']:+.3f} "
                    f"[{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}]{end} & {p_str}{dag} \\\\")
        if block:
            lines.extend(block)
            lines.append("\\midrule")
    lines[-1] = "\\bottomrule"
    lines += ["\\end{tabular}", "\\end{table*}", ""]
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"  wrote {os.path.relpath(path, REPO)}")


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    args = ap.parse_args()
    seeds = args.seeds
    print(f"[aggregate] seeds = {seeds}")

    metrics = load_summary(seeds)
    preds = load_preds(seeds)

    # summary.json
    summary = {}
    for (model, cohort, head, met), vals in metrics.items():
        m, lo, hi = mean_ci(vals)
        summary[f"{model}/{cohort}/{head}/{met}"] = {
            "mean": m, "ci_lo": lo, "ci_hi": hi,
            "n_seeds": int(np.sum(~np.isnan(vals))), "values": vals}
    os.makedirs(CV_DIR, exist_ok=True)
    with open(os.path.join(CV_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  wrote {os.path.relpath(os.path.join(CV_DIR, 'summary.json'), REPO)}")

    write_cv_table(metrics, seeds, os.path.join(CV_DIR, "cv_table.tex"))
    write_boot_table(preds, os.path.join(CV_DIR, "boot_tests.tex"))
    print("[aggregate] done.")


if __name__ == "__main__":
    main()
