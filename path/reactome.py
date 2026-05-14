"""Reactome → weighted pathway-pathway adjacency for PATH.

Adjacency follows paper §4.1: A_pq is the Jaccard similarity of pathway gene
memberships taken from `ReactomePathways.gmt`, restricted to *Homo sapiens*,
and symmetrically normalised by the matrix max (Eq. 2 in the paper).
"""
from __future__ import annotations

import io
import urllib.request
import zipfile
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np


def _fetch(url: str, dest: Path) -> Path:
    if not dest.exists():
        print(f"  downloading {url}")
        urllib.request.urlretrieve(url, dest)
    return dest


def load_gmt(cache_dir: Path, gmt_url: str,
             min_pathway_size: int = 15) -> Dict[str, Set[str]]:
    """Pathway name → set of gene symbols, filtered to |G_p| >= min_pathway_size."""
    zip_dest = cache_dir / "ReactomePathways.gmt.zip"
    _fetch(gmt_url, zip_dest)

    gene_sets: Dict[str, Set[str]] = {}
    with zipfile.ZipFile(zip_dest) as z:
        gmt_name = next(n for n in z.namelist() if n.endswith(".gmt"))
        text = z.read(gmt_name).decode("utf-8")
    for line in text.strip().split("\n"):
        parts = line.strip().split("\t")
        if len(parts) < 3:
            continue
        name = parts[0].strip()
        genes = {g.strip() for g in parts[2:] if g.strip()}
        if len(genes) >= min_pathway_size:
            gene_sets[name] = genes
    return gene_sets


def jaccard_adjacency(input_names: List[str],
                      gene_sets: Dict[str, Set[str]]
                      ) -> Tuple[List[str], np.ndarray]:
    """Symmetric, max-normalised Jaccard adjacency over input_names.

    Pathways missing from `gene_sets` are dropped (returned name list reflects
    only the survivors). Self-loops are zeroed.
    """
    matched_names = [n for n in input_names if n in gene_sets]
    if not matched_names:
        raise RuntimeError("No pathway names matched the Reactome GMT.")

    sets = [gene_sets[n] for n in matched_names]
    sizes = np.array([len(s) for s in sets], dtype=np.float32)
    N = len(matched_names)
    A = np.zeros((N, N), dtype=np.float32)

    # Precompute hash-based set intersection efficiently
    for i in range(N):
        si = sets[i]
        for j in range(i + 1, N):
            inter = len(si & sets[j])
            if inter == 0:
                continue
            union = sizes[i] + sizes[j] - inter
            jac = inter / union
            A[i, j] = jac
            A[j, i] = jac

    # Paper Eq. 2 — symmetric (already) + max-normalise
    eps = 1e-8
    A = A / (A.max() + eps)
    np.fill_diagonal(A, 0.0)
    return matched_names, A


def assemble(pathway_names: List[str],
             cache_dir: Path,
             gmt_url: str,
             min_pathway_size: int,
             jaccard_threshold: float = 0.0
             ) -> dict:
    gene_sets = load_gmt(cache_dir, gmt_url, min_pathway_size=min_pathway_size)
    matched_names, A = jaccard_adjacency(pathway_names, gene_sets)

    if jaccard_threshold > 0.0:
        A = np.where(A >= jaccard_threshold, A, 0.0).astype(np.float32)

    n_edges = int((A > 0).sum() / 2)
    return {
        "input_names": matched_names,
        "adjacency": A,
        "n_missing": len(pathway_names) - len(matched_names),
        "n_edges_undirected": n_edges,
        "max_jaccard": float(A.max()) if n_edges else 0.0,
        "median_pathway_size": float(np.median([len(gene_sets[n])
                                                 for n in matched_names])),
    }
