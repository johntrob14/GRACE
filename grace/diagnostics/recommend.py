"""GRACE recommender — encode the §6 workflow as a single function.

Given cached statistics + activations for one (model, concept), produce a
recommendation block:

- which vector construction to use (`pv` / `unit_mean` / `cluster`)
- which layer set to constrain the search to (`top15_pl` / `union_pl_ra`)
- the diagnostic numbers that drove each choice

Recommendation rules (paper §6):

1. Magnitude CV above the model-specific threshold → prefer `unit_mean`.
   Threshold from paper §H: 0.45 on Gemma-3-27B; signal does not generalize
   on the other paper models, so the threshold is +∞ (i.e. `pv` by default).

2. Per-pair similarity matrix shows persistent block structure (a subset of
   prompt pairs disagrees with the others on ≥ 40% of layers) → prefer
   `cluster`. Without a small pilot eval we apply this conservatively, only
   when both the block fraction is ≥ 60% (very strong signal).

3. Pearson correlation between PL and RA alignment profiles is below 0.2 →
   widen the layer set to the union of top-15 PL ∪ top-15 RA, otherwise stay
   on top-15 PL.

The recommender is also importable for use in notebooks.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

from grace.diagnostics.alignment import alignment_per_layer, top_k_layers
from grace.diagnostics.fragmentation import pl_ra_correlation
from grace.diagnostics.granularity import granularity
from grace.diagnostics.magnitude import magnitude_cv
from grace.diagnostics.pair_heatmap import detect_block_structure, per_pair_similarity

# Paper §H reports unit_mean's signal generalizing only on Gemma-3-27B.
_UNIT_MEAN_CV_THRESHOLD: dict[str, float] = {
    "google/gemma-3-27b-it": 0.45,
    "google/gemma-2-2b-it": float("inf"),
    "meta-llama/Llama-3.3-70B-Instruct": float("inf"),
}

_PL_RA_FRAGMENTATION_THRESHOLD = 0.2
_BLOCK_STRUCTURE_LAYER_FRACTION = 0.40
_BLOCK_STRUCTURE_STRONG_FRACTION = 0.60


@dataclass
class Recommendation:
    model_name: str
    concept: str
    granularity: float
    top15_pl_layers: list[int]
    pl_ra_correlation: float
    magnitude_cv: float
    block_structure_fraction: float
    block_structure_outlier_pairs: list[int]
    construction: str             # "pv" | "unit_mean" | "cluster"
    search_space: str             # "top15_pl" | "union_pl_ra"
    search_layers: list[int]
    rationale: list[str]

    def as_dict(self) -> dict:
        return asdict(self)


def recommend_for_concept(
    model_name: str,
    concept: str,
    activations_root: str = "activations",
    statistics_root: str = "results/statistics",
) -> Recommendation:
    """Run the full diagnostic suite and emit a Recommendation."""
    rationale: list[str] = []

    # --- Geometry --------------------------------------------------------
    A_pl = alignment_per_layer(model_name, concept, "prompt_last", statistics_root)
    top15_pl = top_k_layers(A_pl, k=15)

    G, _ = granularity(model_name, concept, activation_type="prompt_last", statistics_root=statistics_root)
    rationale.append(f"granularity G_c = {G:.3f}")

    # --- Fragmentation (PL/RA correlation) -------------------------------
    pl_ra_r, _, A_ra = pl_ra_correlation(model_name, concept, statistics_root=statistics_root)
    if pl_ra_r != pl_ra_r:  # nan
        rationale.append("PL/RA correlation could not be computed; defaulting to top-15 PL.")
        search_space = "top15_pl"
        search_layers = top15_pl
    elif pl_ra_r < _PL_RA_FRAGMENTATION_THRESHOLD:
        top15_ra = top_k_layers(A_ra, k=15)
        union = sorted(set(top15_pl) | set(top15_ra))
        rationale.append(
            f"PL/RA correlation {pl_ra_r:.2f} < {_PL_RA_FRAGMENTATION_THRESHOLD}; "
            f"widening search to union of top-15 PL ∪ top-15 RA ({len(union)} layers)."
        )
        search_space = "union_pl_ra"
        search_layers = union
    else:
        rationale.append(
            f"PL/RA correlation {pl_ra_r:.2f} ≥ {_PL_RA_FRAGMENTATION_THRESHOLD}; PL alone is sufficient."
        )
        search_space = "top15_pl"
        search_layers = top15_pl

    # --- Magnitude CV (unit_mean trigger) --------------------------------
    cv, _ = magnitude_cv(model_name, concept, "response_avg", statistics_root)
    cv_threshold = _UNIT_MEAN_CV_THRESHOLD.get(model_name, float("inf"))

    # --- Per-pair block structure (cluster trigger) ----------------------
    sim_by_layer = per_pair_similarity(model_name, concept, "response_avg", activations_root)
    detected, info = detect_block_structure(sim_by_layer)
    block_fraction = float(info.get("fraction", 0.0))
    outlier_pairs: list[int] = sorted(info.get("outlier_pair_counts", {}))

    construction = "pv"
    if detected and block_fraction >= _BLOCK_STRUCTURE_STRONG_FRACTION:
        construction = "cluster"
        rationale.append(
            f"persistent block structure on {block_fraction:.0%} of layers — prefer cluster."
        )
    elif detected:
        rationale.append(
            f"weak block structure on {block_fraction:.0%} of layers — cluster is borderline; "
            f"run a small pilot eval before switching off pv."
        )
    if cv > cv_threshold and construction == "pv":
        construction = "unit_mean"
        rationale.append(
            f"magnitude CV {cv:.2f} > model threshold {cv_threshold:.2f} — prefer unit_mean."
        )
    elif cv > cv_threshold and construction == "cluster":
        rationale.append(
            f"magnitude CV {cv:.2f} also high but cluster takes priority over unit_mean here."
        )

    return Recommendation(
        model_name=model_name,
        concept=concept,
        granularity=G,
        top15_pl_layers=top15_pl,
        pl_ra_correlation=pl_ra_r,
        magnitude_cv=cv,
        block_structure_fraction=block_fraction,
        block_structure_outlier_pairs=outlier_pairs,
        construction=construction,
        search_space=search_space,
        search_layers=search_layers,
        rationale=rationale,
    )


def format_recommendation(rec: Recommendation) -> str:
    """Render a Recommendation as the printed block shown in the docs."""
    lines = [
        f"CONCEPT: {rec.concept}  ({rec.model_name})",
        f"  Granularity G_c     = {rec.granularity:.3f}",
        f"  PL/RA correlation   = {rec.pl_ra_correlation:.2f}",
        f"  Magnitude CV        = {rec.magnitude_cv:.2f}",
        f"  Block-structure fr. = {rec.block_structure_fraction:.0%}"
        + (f"  outlier pairs: {rec.block_structure_outlier_pairs}" if rec.block_structure_outlier_pairs else ""),
        f"  Top-15 PL layers    = {rec.top15_pl_layers}",
        "  >>> Recommendation:",
        f"        construction:  {rec.construction}",
        f"        search space:  {rec.search_space}  ({len(rec.search_layers)} layers)",
        f"        search budget: 50 trials × 3 seeds",
        "  Rationale:",
        *(f"    - {r}" for r in rec.rationale),
    ]
    return "\n".join(lines)
