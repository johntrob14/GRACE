"""Fixed-interval grid search over (layer, coefficient) — paper §3.

For each concept the paper sweeps every-5-layers × {1, 2, 3} coefficients with
response-averaged steering vectors. The output of this sweep is what the
alignment notebooks (``01_alignment_predicts_layers.ipynb``) plot against
prompt-boundary alignment.
"""
from __future__ import annotations

from itertools import product
from pathlib import Path

from grace.eval.results import build_results_cache
from grace.eval.runner import evaluate_one


def grid_search(
    model,
    tokenizer,
    judge,
    *,
    model_name: str,
    concept: str,
    method: str,
    layers: list[int],
    coefs: list[float],
    judge_tag: str = "judge_gemma3_12b",
    n_questions: int | None = None,
    out_root: str | Path = "results/steering_evaluations",
    vectors_root: str | Path = "vectors",
    overwrite: bool = False,
) -> list[dict]:
    """Evaluate every (layer, coef) combination. Skips already-cached configs unless overwrite=True."""
    cache = (
        build_results_cache(model_name, concept, method, judge_tag=judge_tag, root=str(out_root))
        if not overwrite else {}
    )
    summaries = []
    for layer, coef in product(layers, coefs):
        if (layer, coef) in cache:
            summaries.append({"concept": concept, "method": method, "layer": layer, "coef": coef, "mean_utility": cache[(layer, coef)], "cached": True})
            continue
        s = evaluate_one(
            model, tokenizer, judge,
            model_name=model_name, concept=concept, method=method,
            layer=layer, coef=coef,
            judge_tag=judge_tag, n_questions=n_questions, out_root=out_root,
            vectors_root=vectors_root,
            overwrite=overwrite,
        )
        s["cached"] = False
        summaries.append(s)
    return summaries
