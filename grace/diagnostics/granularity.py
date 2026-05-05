"""Concept granularity G_c — paper §5, eqs. (5)–(6).

Decomposes prompt-boundary alignment 𝒜_c(ℓ) into:

- γ_c(ℓ): average pairwise cosine sim across **within-question** prompt-pairs
          (same q, different p).
- λ_c(ℓ): average pairwise cosine sim across **cross-question** prompt-pairs
          (different q).

Granularity is then ``G_c(ℓ) = γ_c(ℓ) / 𝒜_c(ℓ)``. Aggregated across layers we
report ``mean_ℓ G_c(ℓ)`` as the per-concept granularity scalar.
"""
from __future__ import annotations

import statistics as _stats

from grace.activations.statistics import load_statistics


def gamma_per_layer(
    model_name: str,
    concept: str,
    activation_type: str = "prompt_last",
    statistics_root: str = "results/statistics",
) -> dict[int, float]:
    """γ_c(ℓ): mean of per-question avg_cosine_sim values at each layer."""
    stats = load_statistics(model_name, concept, root=statistics_root)
    out: dict[int, float] = {}
    for layer, by_at in stats["prompt_paired"]["by_layer"].items():
        layer_data = by_at.get(activation_type, {})
        per_q = layer_data.get("per_question", {})
        cs = [
            q_stats.get("avg_cosine_sim")
            for q_stats in per_q.values()
            if q_stats.get("avg_cosine_sim") is not None
        ]
        if cs:
            out[int(layer)] = float(sum(cs) / len(cs))
    return out


def lambda_per_layer(
    model_name: str,
    concept: str,
    activation_type: str = "prompt_last",
    statistics_root: str = "results/statistics",
) -> dict[int, float]:
    """λ_c(ℓ): the cross-question component, derived from the alignment decomp.

    With weights w_W = N_W / N_T and w_C = N_C / N_T, eq. (5) gives:
        𝒜 = w_W * γ + w_C * λ   ⇒   λ = (𝒜 - w_W * γ) / w_C.

    For paper's setup (P=5, Q=100) we have N_W = P(P-1)/2 * Q = 1000,
    N_T = (P*Q choose 2) = 124750, so w_W ≈ 0.008 and w_C ≈ 0.992.
    """
    from grace.diagnostics.alignment import alignment_per_layer
    A = alignment_per_layer(model_name, concept, activation_type, statistics_root)
    g = gamma_per_layer(model_name, concept, activation_type, statistics_root)

    stats = load_statistics(model_name, concept, root=statistics_root)
    sample_layer = next(iter(stats["prompt_paired"]["by_layer"]))
    layer_data = stats["prompt_paired"]["by_layer"][sample_layer][activation_type]
    n_t = layer_data.get("n_vectors", 0)
    if n_t < 2:
        return {}
    per_q = layer_data.get("per_question", {})
    n_w = sum(
        (q.get("n_vectors", 0) * (q.get("n_vectors", 0) - 1)) // 2
        for q in per_q.values()
    )
    n_total_pairs = (n_t * (n_t - 1)) // 2
    w_w = n_w / n_total_pairs if n_total_pairs else 0.0
    w_c = 1.0 - w_w
    if w_c <= 0:
        return {}

    out: dict[int, float] = {}
    for ell, alignment in A.items():
        if ell in g:
            out[ell] = (alignment - w_w * g[ell]) / w_c
    return out


def granularity(
    model_name: str,
    concept: str,
    activation_type: str = "prompt_last",
    statistics_root: str = "results/statistics",
) -> tuple[float, dict[int, float]]:
    """Return ``(G_c, {layer: G_c(layer)})`` for one concept.

    `G_c(ℓ)` is `γ_c(ℓ) / 𝒜_c(ℓ)`; the scalar `G_c` is `mean_ℓ G_c(ℓ)`.
    """
    A = alignment_per_layer_or_zero(model_name, concept, activation_type, statistics_root)
    g = gamma_per_layer(model_name, concept, activation_type, statistics_root)
    per_layer = {
        ell: g[ell] / A[ell]
        for ell in g
        if ell in A and A[ell] != 0
    }
    if not per_layer:
        return float("nan"), {}
    return float(_stats.mean(per_layer.values())), per_layer


def alignment_per_layer_or_zero(model_name, concept, activation_type, statistics_root):
    from grace.diagnostics.alignment import alignment_per_layer
    return alignment_per_layer(model_name, concept, activation_type, statistics_root)
