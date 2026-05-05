#!/usr/bin/env python3
"""Optuna TPE search over (layer, coef) — paper §4.

Three search modes: ``unconstrained``, ``top15_pl``, ``union_pl_ra``.
The mode determines which layers TPE samples from; everything else is shared.

Each trial loads the steered model, generates, frees the model, loads the
judge, scores, and frees the judge — generation and judging never share GPU
residency. See ``grace.search.optuna_search`` for the per-trial protocol.
"""
import argparse

import yaml

from grace.config import MODEL_CONFIGS
from grace.diagnostics.alignment import alignment_per_layer, top_k_layers
from grace.search.optuna_search import optuna_search


def _num_layers_from_registry(model_name: str) -> int:
    cfg = MODEL_CONFIGS.get(model_name)
    if cfg is None:
        raise KeyError(
            f"{model_name!r} not in grace.config.MODEL_CONFIGS; add it (with num_layers) "
            f"or extend this script to load the model briefly."
        )
    return int(cfg["num_layers"])


def _layers_for_mode(model_name: str, mode: str, concept: str, statistics_root: str) -> list[int]:
    n_layers = _num_layers_from_registry(model_name)
    full = list(range(1, n_layers + 1))
    if mode == "unconstrained":
        return full
    pl = alignment_per_layer(model_name, concept, "prompt_last", statistics_root)
    pl_top = top_k_layers(pl, k=15)
    if mode == "top15_pl":
        return pl_top
    if mode == "union_pl_ra":
        ra = alignment_per_layer(model_name, concept, "response_avg", statistics_root)
        ra_top = top_k_layers(ra, k=15)
        return sorted(set(pl_top) | set(ra_top))
    raise ValueError(f"Unknown mode: {mode}")


def _apply_overrides(cfg: dict, overrides: list[str]) -> dict:
    for kv in overrides:
        if "=" not in kv:
            continue
        k, v = kv.split("=", 1)
        try:
            v = int(v)
        except ValueError:
            try:
                v = float(v)
            except ValueError:
                pass
        cfg[k] = v
    return cfg


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True, help="Path to a configs/search/*.yaml file.")
    p.add_argument("--concept", required=True)
    p.add_argument("--override", nargs="*", default=[], help="key=value overrides.")
    p.add_argument("--overwrite", action="store_true",
                   help="Re-run and overwrite existing per-config eval CSVs that the TPE objective re-hits.")
    args = p.parse_args()

    cfg = yaml.safe_load(open(args.config))
    cfg = _apply_overrides(cfg, args.override)

    model_name = cfg["model"]
    method = cfg["method"]
    mode = cfg["mode"]
    coefs = list(cfg["coefs"])
    n_trials = int(cfg.get("n_trials", 50))
    n_seeds = int(cfg.get("n_seeds", 3))
    judge_tag = cfg.get("judge_tag", "judge_gemma3_12b")
    judge_model = cfg.get("judge_model", "google/gemma-3-12b-it")
    questions = cfg.get("questions")
    statistics_root = cfg.get("statistics_root", "results/statistics")
    eval_root = cfg.get("eval_root", "results/steering_evaluations")
    out_root = cfg.get("out_root", "results/optuna")
    vectors_root = cfg.get("vectors_root", "vectors")

    layers = _layers_for_mode(model_name, mode, args.concept, statistics_root)

    history = optuna_search(
        model_name=model_name, judge_model_name=judge_model,
        concept=args.concept, method=method,
        layers=layers, coefs=coefs, mode=mode,
        n_trials=n_trials, n_seeds=n_seeds,
        judge_tag=judge_tag, n_questions=questions,
        out_root=out_root, eval_root=eval_root,
        vectors_root=vectors_root,
        overwrite=args.overwrite,
    )
    print(f"Wrote {len(history)} trial rows to {out_root}/")


if __name__ == "__main__":
    main()
