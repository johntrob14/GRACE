"""Fixed-interval grid search over (layer, coefficient) — paper §3.

For each concept the paper sweeps every-5-layers x {1, 2, 3} coefficients with
response-averaged steering vectors. The output of this sweep is what the
alignment notebooks (``01_alignment_predicts_layers.ipynb``) plot against
prompt-boundary alignment.

Generation and judging run as two passes so the steered model and the judge
never reside in GPU memory at the same time. The caller drives the lifecycle:
load the steered model, call ``generate_grid``, free the model, load the judge,
call ``score_grid``, free the judge.
"""
from __future__ import annotations

from itertools import product
from pathlib import Path

from grace.eval.results import build_results_cache
from grace.eval.runner import generate_responses_one, score_responses_one


def generate_grid(
    model,
    tokenizer,
    *,
    model_name: str,
    concept: str,
    method: str,
    layers: list[int],
    coefs: list[float],
    judge_tag: str = "judge_gemma3_12b",
    n_questions: int | None = None,
    max_new_tokens: int = 256,
    out_root: str | Path = "results/steering_evaluations",
    vectors_root: str | Path = "vectors",
    overwrite: bool = False,
) -> list[Path]:
    """Generate steered responses for every (layer, coef) and write per_sample CSVs."""
    paths = []
    for layer, coef in product(layers, coefs):
        p = generate_responses_one(
            model, tokenizer,
            model_name=model_name, concept=concept, method=method,
            layer=layer, coef=coef,
            judge_tag=judge_tag, n_questions=n_questions, max_new_tokens=max_new_tokens,
            out_root=out_root, vectors_root=vectors_root,
            overwrite=overwrite,
        )
        paths.append(p)
    return paths


def score_grid(
    judge,
    *,
    model_name: str,
    concept: str,
    method: str,
    layers: list[int],
    coefs: list[float],
    judge_tag: str = "judge_gemma3_12b",
    out_root: str | Path = "results/steering_evaluations",
    overwrite: bool = False,
) -> list[dict]:
    """Score the per_sample CSVs produced by ``generate_grid`` and write summaries.

    Reads cached summaries when present (skip-if-exists unless ``overwrite``).
    """
    cache = (
        build_results_cache(model_name, concept, method, judge_tag=judge_tag, root=str(out_root))
        if not overwrite else {}
    )
    summaries = []
    for layer, coef in product(layers, coefs):
        if (layer, coef) in cache:
            summaries.append({
                "concept": concept, "method": method, "layer": layer, "coef": coef,
                "mean_utility": cache[(layer, coef)], "cached": True,
            })
            continue
        s = score_responses_one(
            judge,
            model_name=model_name, concept=concept, method=method,
            layer=layer, coef=coef,
            judge_tag=judge_tag, out_root=out_root,
            overwrite=overwrite,
        )
        s["cached"] = False
        summaries.append(s)
    return summaries
