"""Helpers shared by grid + Optuna search."""
from __future__ import annotations

from grace.eval.results import build_results_cache as _build_results_cache


def build_results_cache(
    model_name: str,
    concept: str,
    method: str,
    judge_tag: str = "judge_gemma3_12b",
    root: str = "results/steering_evaluations",
) -> dict[tuple[int, float], float]:
    """Re-export so the search package has a stable import path."""
    return _build_results_cache(model_name, concept, method, judge_tag=judge_tag, root=root)


def restrict_to_top_k_layers(layers: list[int], alignment: dict[int, float], k: int) -> list[int]:
    """Return the top-k layers from `layers` ranked by `alignment[layer]` descending.

    `alignment` is the per-layer prompt-boundary alignment 𝒜_c(ℓ) (or whichever
    alignment metric the caller wants to use). Layers without an entry are
    skipped silently.
    """
    scored = [(ell, alignment[ell]) for ell in layers if ell in alignment]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [ell for ell, _ in scored[:k]]
