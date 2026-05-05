"""Two-way ANOVA on contrastive difference vectors — paper Appendix §C.4.

For each concept × layer we have a balanced P × Q × D matrix of unit
difference vectors. A two-way ANOVA decomposes total directional variance
into three additive components:

- η²_prompt       : variance attributable to the instruction-template effect
- η²_question     : variance attributable to the evaluation-context effect
- η²_interaction  : residual prompt × question variance

The paper reports these averaged across layers, per model. The Question
effect is the binding constraint on rank-1 steering: when η²_question is high,
no single direction can satisfy all input contexts simultaneously.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from grace.activations.io import load_activations
from grace.vectors._common import n_pairs_in


def _layer_eta_squared(
    activations_by_question: dict,
    activation_type: str,
    layer: int,
) -> dict[str, float] | None:
    """Compute (η²_prompt, η²_question, η²_interaction) at one layer.

    Treats each unit-normalized difference vector as a multivariate sample.
    Total SS = Σ ||v_pq − v̄||²; component SSs are computed by the standard
    two-way fixed-effects decomposition.
    """
    P = n_pairs_in(activations_by_question)
    Q = len(activations_by_question)
    matrix: list[list[torch.Tensor]] = [[None] * Q for _ in range(P)]  # type: ignore
    for q_idx, qd in enumerate(activations_by_question.values()):
        pos_list, neg_list = qd.get("pos", []), qd.get("neg", [])
        for p in range(min(P, len(pos_list), len(neg_list))):
            if (
                activation_type in pos_list[p]
                and activation_type in neg_list[p]
                and layer in pos_list[p][activation_type]
            ):
                v = pos_list[p][activation_type][layer] - neg_list[p][activation_type][layer]
                matrix[p][q_idx] = F.normalize(v.float(), dim=0)
    # Drop columns/rows with missing entries.
    keep_cols = [q for q in range(Q) if all(matrix[p][q] is not None for p in range(P))]
    if len(keep_cols) < 2:
        return None
    Q = len(keep_cols)
    cells = torch.stack([torch.stack([matrix[p][q] for q in keep_cols]) for p in range(P)])  # [P, Q, D]
    grand_mean = cells.mean(dim=(0, 1))
    prompt_mean = cells.mean(dim=1)  # [P, D]
    question_mean = cells.mean(dim=0)  # [Q, D]

    ss_total = float(((cells - grand_mean) ** 2).sum())
    ss_prompt = float(Q * ((prompt_mean - grand_mean) ** 2).sum())
    ss_question = float(P * ((question_mean - grand_mean) ** 2).sum())
    ss_interaction = max(ss_total - ss_prompt - ss_question, 0.0)
    if ss_total == 0:
        return None
    return {
        "eta_sq_prompt": ss_prompt / ss_total,
        "eta_sq_question": ss_question / ss_total,
        "eta_sq_interaction": ss_interaction / ss_total,
    }


def anova_decomposition(
    model_name: str,
    concept: str,
    activation_type: str = "response_avg",
    activations_root: str = "activations",
) -> dict[str, float | dict[int, dict[str, float]]]:
    """Return per-layer η² components plus the layer-averaged means."""
    data = load_activations(model_name, concept, root=activations_root)
    layer_list = list(data["layer_list"])
    abq = data["activations_by_question"]

    per_layer: dict[int, dict[str, float]] = {}
    for layer in layer_list:
        components = _layer_eta_squared(abq, activation_type, layer)
        if components is not None:
            per_layer[int(layer)] = components

    if not per_layer:
        return {"per_layer": {}, "mean_eta_sq_prompt": float("nan"),
                "mean_eta_sq_question": float("nan"),
                "mean_eta_sq_interaction": float("nan")}

    keys = ("eta_sq_prompt", "eta_sq_question", "eta_sq_interaction")
    means = {f"mean_{k}": sum(v[k] for v in per_layer.values()) / len(per_layer) for k in keys}
    return {"per_layer": per_layer, **means}
