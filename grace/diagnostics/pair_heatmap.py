"""Per-prompt-pair similarity matrix — paper §6 (cluster diagnostic).

For each layer we form the P×P matrix where entry ``[i, j]`` is the cosine
similarity between the per-pair mean directions ``pair_vec_i`` and
``pair_vec_j`` (each averaged across all questions). A ``cluster`` vector
construction is a useful remedy when this matrix shows persistent block
structure across many layers — i.e. a subset of pairs that mutually agree
but disagree with the others.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from grace.activations.io import load_activations
from grace.vectors._common import n_pairs_in, per_pair_means


def per_pair_similarity(
    model_name: str,
    concept: str,
    activation_type: str = "response_avg",
    activations_root: str = "activations",
) -> dict[int, torch.Tensor]:
    """Return ``{layer: P×P cosine-similarity matrix}``."""
    data = load_activations(model_name, concept, root=activations_root)
    layer_list = list(data["layer_list"])
    abq = data["activations_by_question"]
    P = n_pairs_in(abq)
    out: dict[int, torch.Tensor] = {}
    for layer in layer_list:
        pair_vecs = per_pair_means(abq, activation_type, layer, P)
        if any(v is None for v in pair_vecs):
            continue
        stacked = F.normalize(torch.stack([v for v in pair_vecs if v is not None]).float(), dim=1)
        out[int(layer)] = stacked @ stacked.T
    return out


def detect_block_structure(
    sim_by_layer: dict[int, torch.Tensor],
    high: float = 0.7,
    low: float = 0.3,
    min_layer_fraction: float = 0.4,
) -> tuple[bool, dict[str, float | list[int]]]:
    """Return ``(detected, info)``.

    Block structure is "detected" when, on at least `min_layer_fraction` of layers,
    there exists a strict subset of pairs whose intra-block mean cosine sim > `high`
    while at least one cross-block cosine sim < `low`.
    """
    if not sim_by_layer:
        return False, {"layers_with_block": 0, "total_layers": 0, "fraction": 0.0}

    layers_with_block: list[int] = []
    pair_outlier_counts: dict[int, int] = {}

    for layer, sim in sim_by_layer.items():
        P = sim.shape[0]
        if P < 3:
            continue
        # Heuristic: find any pair index with strong outlier behaviour
        # (very low similarity to ≥ ceil(P/2) other pairs).
        had_block = False
        for i in range(P):
            n_low = int(((sim[i] < low) & (torch.arange(P) != i)).sum().item())
            if n_low >= max(1, (P - 1) // 2):
                # check there's at least one strong cross-pair somewhere else
                others = [j for j in range(P) if j != i]
                if len(others) >= 2:
                    sub = sim[others][:, others]
                    iu = torch.triu_indices(len(others), len(others), offset=1)
                    if len(iu[0]) > 0 and float(sub[iu[0], iu[1]].mean()) > high:
                        had_block = True
                        pair_outlier_counts[i] = pair_outlier_counts.get(i, 0) + 1
        if had_block:
            layers_with_block.append(int(layer))

    total = len(sim_by_layer)
    frac = len(layers_with_block) / total if total else 0.0
    info: dict[str, float | list[int]] = {
        "layers_with_block": layers_with_block,
        "total_layers": total,
        "fraction": frac,
        "outlier_pair_counts": pair_outlier_counts,
    }
    return frac >= min_layer_fraction, info
