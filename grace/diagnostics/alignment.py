"""Prompt-boundary alignment 𝒜_c(ℓ) — paper §3, eq. (3).

For concept c at layer ℓ, ``𝒜_c(ℓ)`` is the average pairwise cosine similarity
of the unit-normalized prompt-boundary difference vectors. High alignment
indicates a broadly consistent direction at that layer.

We don't recompute pairwise cosines here — they're already stored in the
per-concept statistics dict under ``prompt_paired.by_layer[ℓ][activation_type].
avg_cosine_sim``. We just expose them under a paper-faithful interface.
"""
from __future__ import annotations

from grace.activations.statistics import load_statistics


def alignment_per_layer(
    model_name: str,
    concept: str,
    activation_type: str = "prompt_last",
    statistics_root: str = "results/statistics",
) -> dict[int, float]:
    """Return ``{layer: 𝒜_c(layer)}``.

    Default is `prompt_last` (the paper's PL variant). Pass `response_avg` to
    get the RA-variant alignment used in the fragmentation diagnostic.
    """
    stats = load_statistics(model_name, concept, root=statistics_root)
    out: dict[int, float] = {}
    for layer, by_at in stats["prompt_paired"]["by_layer"].items():
        if activation_type in by_at:
            v = by_at[activation_type].get("avg_cosine_sim")
            if v is not None:
                out[int(layer)] = float(v)
    return out


def top_k_layers(alignment: dict[int, float], k: int) -> list[int]:
    """Return the top-k layers by alignment, descending."""
    ordered = sorted(alignment.items(), key=lambda kv: kv[1], reverse=True)
    return [layer for layer, _ in ordered[:k]]
