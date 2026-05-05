"""Magnitude coefficient of variation — paper §6 (Unit Mean diagnostic).

The CV of difference-vector magnitudes across (P, Q) pairs at a given layer.
A high CV (heavy-tailed magnitude distribution) is the signal that a few
high-magnitude samples may dominate the PV average; this is when Unit Mean
construction is expected to help.
"""
from __future__ import annotations

import statistics as _stats

from grace.activations.statistics import load_statistics


def magnitude_cv(
    model_name: str,
    concept: str,
    activation_type: str = "response_avg",
    statistics_root: str = "results/statistics",
) -> tuple[float, dict[int, float]]:
    """Return ``(concept_level_CV, {layer: CV(layer)})`` for the chosen activation type.

    Per-layer CV is ``magnitude_std / avg_magnitude``. The concept-level scalar
    is the mean of per-layer CVs.
    """
    stats = load_statistics(model_name, concept, root=statistics_root)
    out: dict[int, float] = {}
    for layer, by_at in stats["prompt_paired"]["by_layer"].items():
        ld = by_at.get(activation_type, {})
        avg = ld.get("avg_magnitude")
        std = ld.get("magnitude_std")
        if avg and avg > 0 and std is not None:
            out[int(layer)] = float(std) / float(avg)
    if not out:
        return float("nan"), {}
    return float(_stats.mean(out.values())), out
