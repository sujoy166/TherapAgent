"""Reactome → pathway-pathway adjacency matrix for GraphPath.

The published GraphPath paper builds adjacency from KEGG pathway-in-pathway
references. Reactome's closest analog is its parent/child curation: two
pathways are adjacent if (a) one is a parent of the other, or (b) they share
a parent (siblings). Sibling edges can be disabled via Config.include_siblings.
"""
from __future__ import annotations

import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np


def _fetch(url: str, dest: Path) -> Path:
    if not dest.exists():
        print(f"  downloading {url}")
        urllib.request.urlretrieve(url, dest)
    return dest


def load_reactome(cache_dir: Path, pathways_url: str, relations_url: str,
                  species_prefix: str = "R-HSA-"
                  ) -> Tuple[Dict[str, str], Dict[str, str],
                              Dict[str, Set[str]], Dict[str, Set[str]]]:
    """Return (id→name, name→id, parents[child]→{parents}, children[parent]→{children})."""
    pw_file = _fetch(pathways_url, cache_dir / "ReactomePathways.txt")
    rel_file = _fetch(relations_url, cache_dir / "ReactomePathwaysRelation.txt")

    id2name: Dict[str, str] = {}
    name2id: Dict[str, str] = {}
    with open(pw_file, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            pid, name, species = parts[0], parts[1], parts[2]
            if not pid.startswith(species_prefix):
                continue
            id2name[pid] = name
            if name not in name2id or pid < name2id[name]:
                name2id[name] = pid

    parents: Dict[str, Set[str]] = defaultdict(set)
    children: Dict[str, Set[str]] = defaultdict(set)
    with open(rel_file, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            parent, child = parts[0], parts[1]
            if parent.startswith(species_prefix) and child.startswith(species_prefix):
                parents[child].add(parent)
                children[parent].add(child)

    return id2name, name2id, parents, children


def build_adjacency(input_names: List[str],
                    name2id: Dict[str, str],
                    parents: Dict[str, Set[str]],
                    children: Dict[str, Set[str]],
                    include_siblings: bool = True
                    ) -> Tuple[List[str], List[str], np.ndarray]:
    """Build a symmetric {0,1} adjacency over Reactome-matched pathways.

    Inputs that don't map to any Reactome ID are dropped (their indices vanish
    from the returned name/id lists).
    """
    matched_names, matched_ids = [], []
    for n in input_names:
        pid = name2id.get(n)
        if pid is not None:
            matched_names.append(n)
            matched_ids.append(pid)
    if not matched_ids:
        raise RuntimeError("No pathway names matched Reactome.")

    idx_of = {pid: i for i, pid in enumerate(matched_ids)}
    N = len(matched_ids)
    A = np.zeros((N, N), dtype=np.float32)

    # Parent/child edges
    for child, plist in parents.items():
        if child not in idx_of:
            continue
        i = idx_of[child]
        for parent in plist:
            j = idx_of.get(parent)
            if j is not None:
                A[i, j] = 1.0
                A[j, i] = 1.0

    if include_siblings:
        # Two pathways are siblings if they share a parent
        for parent, kids in children.items():
            present = [idx_of[k] for k in kids if k in idx_of]
            for a in range(len(present)):
                for b in range(a + 1, len(present)):
                    A[present[a], present[b]] = 1.0
                    A[present[b], present[a]] = 1.0

    np.fill_diagonal(A, 0.0)
    return matched_names, matched_ids, A


def assemble(pathway_names: List[str],
             cache_dir: Path,
             pathways_url: str,
             relations_url: str,
             species_prefix: str,
             include_siblings: bool
             ) -> dict:
    id2name, name2id, parents, children = load_reactome(
        cache_dir, pathways_url, relations_url, species_prefix
    )
    matched_names, matched_ids, A = build_adjacency(
        pathway_names, name2id, parents, children, include_siblings
    )
    return {
        "input_names": matched_names,
        "input_ids": matched_ids,
        "adjacency": A,
        "n_missing": len(pathway_names) - len(matched_ids),
        "n_edges_undirected": int(A.sum() / 2),
    }
