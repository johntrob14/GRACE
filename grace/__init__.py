"""GRACE: Granularity- and Representation-Aware Concept Engineering.

Companion code for the NeurIPS submission "When Is Rank-1 Steering Cheap?
Geometry, Granularity, and Budgeted Search."

The package is organized as:

    grace.data            Concept-data generation (GPT-5 → JSON pairs/questions/rubric).
    grace.activations     Extract residual-stream activations and per-layer statistics.
    grace.vectors         Three rank-1 vector constructions: pv, unit_mean, cluster.
    grace.steering        Inference-time activation steering (forward hook).
    grace.eval            LLM-as-judge evaluation; concept score + coherence.
    grace.search          Grid + TPE search over (layer, coefficient).
    grace.diagnostics     The GRACE diagnostic suite (alignment, granularity,
                          magnitude CV, PL/RA correlation, per-pair heatmap, ANOVA,
                          and the workflow recommender).
    grace.analysis        Cross-concept aggregation helpers used by the notebooks.
"""

from grace import config as _config  # noqa: F401  (ensures .env is loaded on import)
