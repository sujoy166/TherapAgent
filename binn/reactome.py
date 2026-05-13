"""Reactome hierarchy → BINN layer specification.

Builds the sparse parent/child layout used by the model. Layer 0 holds the
input pathways present in our dataset; each successive layer holds their
Reactome parents. Nodes that hit a terminal before reaching the requested
depth are copied forward (paper §Methods, step 3).
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
                  ) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, Set[str]]]:
    """Return (id→name, name→id, child_id→{parent_ids}) restricted to species_prefix."""
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
            # If a name collides across IDs, keep the lexicographically smallest ID
            # for determinism (rare for HSA, but defensive).
            if name not in name2id or pid < name2id[name]:
                name2id[name] = pid

    parents: Dict[str, Set[str]] = defaultdict(set)
    with open(rel_file, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            parent, child = parts[0], parts[1]
            if parent.startswith(species_prefix) and child.startswith(species_prefix):
                parents[child].add(parent)

    return id2name, name2id, parents


def build_layers(input_ids: List[str],
                 parent_map: Dict[str, Set[str]],
                 n_layers: int = 4
                 ) -> List[List[str]]:
    """Walk parents upward to produce n_layers ordered node lists.

    Layer 0 = input_ids (deduped, order-preserved).
    Layer i+1 = union of parents of layer i; nodes with no parent get copied
    forward so every path keeps a node at every depth.
    """
    seen = set()
    layer0 = []
    for n in input_ids:
        if n not in seen:
            seen.add(n)
            layer0.append(n)

    layers: List[List[str]] = [layer0]
    for _ in range(n_layers - 1):
        prev = layers[-1]
        nxt: List[str] = []
        seen = set()
        for c in prev:
            ps = parent_map.get(c, set())
            if not ps:
                # Terminal – copy forward to preserve path
                if c not in seen:
                    seen.add(c)
                    nxt.append(c)
            else:
                for p in sorted(ps):  # sort for determinism
                    if p not in seen:
                        seen.add(p)
                        nxt.append(p)
        layers.append(nxt)
    return layers


def build_masks(layers: List[List[str]],
                parent_map: Dict[str, Set[str]]
                ) -> List[np.ndarray]:
    """For each adjacent (in_layer, out_layer) build a {0,1} mask of shape
    (out_features, in_features). mask[i, j] = 1 iff out_layer[i] is a parent
    of in_layer[j], or i==j when in_layer[j] is a terminal copied forward.
    """
    masks: List[np.ndarray] = []
    for l in range(len(layers) - 1):
        layer_in, layer_out = layers[l], layers[l + 1]
        out_idx = {n: i for i, n in enumerate(layer_out)}
        mask = np.zeros((len(layer_out), len(layer_in)), dtype=np.float32)
        for j, child in enumerate(layer_in):
            parents = parent_map.get(child, set())
            if not parents and child in out_idx:
                mask[out_idx[child], j] = 1.0
            for p in parents:
                if p in out_idx:
                    mask[out_idx[p], j] = 1.0
        masks.append(mask)
    return masks


def assemble(pathway_names: List[str],
             cache_dir: Path,
             pathways_url: str,
             relations_url: str,
             species_prefix: str,
             n_hidden_layers: int
             ) -> dict:
    """End-to-end Reactome → layers/masks builder.

    Returns dict with:
      - input_names: list of original-CSV pathway names actually used (Reactome-matched)
      - input_ids:   matching Reactome IDs
      - layers:      list of n_hidden_layers ID lists (layer 0 = inputs, layer N-1 = top)
      - layer_names: same as layers but resolved to human names
      - masks:       list of np.float32 arrays connecting consecutive layers
    """
    id2name, name2id, parent_map = load_reactome(
        cache_dir, pathways_url, relations_url, species_prefix
    )

    matched_names, matched_ids, missing = [], [], []
    for name in pathway_names:
        pid = name2id.get(name)
        if pid is None:
            missing.append(name)
        else:
            matched_names.append(name)
            matched_ids.append(pid)

    if not matched_ids:
        raise RuntimeError(
            "No pathway names from the CSV matched Reactome. "
            f"First few unmatched: {missing[:5]}"
        )

    layers = build_layers(matched_ids, parent_map, n_layers=n_hidden_layers)
    masks = build_masks(layers, parent_map)
    layer_names = [[id2name.get(i, i) for i in layer] for layer in layers]

    return {
        "input_names": matched_names,
        "input_ids": matched_ids,
        "layers": layers,
        "layer_names": layer_names,
        "masks": masks,
        "n_missing": len(missing),
        "missing_examples": missing[:10],
    }
