"""Representational fragmentation diagnostic — paper §6 (`app:grace_constrained`).

Pearson correlation across layers between the prompt-boundary alignment
profile 𝒜_c^PL(ℓ) and the response-averaged alignment profile 𝒜_c^RA(ℓ).
When the correlation is below 0.2 the two variants disagree about which
layers are best, and constraining the search to top-15 PL alone risks
missing the true peak. The remedy is to expand the search space to the
union of top-15 PL ∪ top-15 RA layers.
"""
from __future__ import annotations

from grace.diagnostics.alignment import alignment_per_layer


def pl_ra_correlation(
    model_name: str,
    concept: str,
    statistics_root: str = "results/statistics",
) -> tuple[float, dict[int, float], dict[int, float]]:
    """Return ``(pearson_r, A_PL, A_RA)`` for this concept.

    - `A_PL[ℓ]` = prompt-boundary alignment 𝒜_c^PL(ℓ)
    - `A_RA[ℓ]` = response-averaged alignment 𝒜_c^RA(ℓ)
    - `pearson_r` = Pearson correlation between the two profiles across layers.
    """
    pl = alignment_per_layer(model_name, concept, activation_type="prompt_last", statistics_root=statistics_root)
    ra = alignment_per_layer(model_name, concept, activation_type="response_avg", statistics_root=statistics_root)
    common = sorted(set(pl) & set(ra))
    if len(common) < 2:
        return float("nan"), pl, ra
    pl_v = [pl[ell] for ell in common]
    ra_v = [ra[ell] for ell in common]
    return _pearson(pl_v, ra_v), pl, ra


def _pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = sum((x - mx) ** 2 for x in xs) ** 0.5
    dy = sum((y - my) ** 2 for y in ys) ** 0.5
    if dx == 0 or dy == 0:
        return float("nan")
    return num / (dx * dy)
