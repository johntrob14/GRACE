"""The GRACE diagnostic suite (paper §3, §5, §6, Appendix §C, §H).

All diagnostics consume cached activations or per-layer statistics — none of
them require a GPU or an LLM judge.
"""

from grace.diagnostics.alignment import alignment_per_layer, top_k_layers
from grace.diagnostics.granularity import granularity, gamma_per_layer, lambda_per_layer
from grace.diagnostics.magnitude import magnitude_cv
from grace.diagnostics.fragmentation import pl_ra_correlation
from grace.diagnostics.pair_heatmap import per_pair_similarity, detect_block_structure
from grace.diagnostics.recommend import recommend_for_concept

__all__ = [
    "alignment_per_layer",
    "top_k_layers",
    "granularity",
    "gamma_per_layer",
    "lambda_per_layer",
    "magnitude_cv",
    "pl_ra_correlation",
    "per_pair_similarity",
    "detect_block_structure",
    "recommend_for_concept",
]
