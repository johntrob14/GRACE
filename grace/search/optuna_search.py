"""Optuna TPE search over (layer, coefficient) — paper §4.

50 trials × 3 seeds is the default; see configs/search/tpe_*.yaml.
The search supports three layer-restriction modes:

- `unconstrained`: TPE samples from `layers` (the model's full layer range).
- `top15_pl`:      TPE samples from the top-15 layers by prompt-boundary
                    alignment (paper §4.2).
- `union_pl_ra`:   TPE samples from the union of top-15 PL + top-15 RA
                    (paper §6 fragmentation remedy).

The caller passes the layer set in `layers`; this module is agnostic to how
that set was selected. It just runs TPE and writes a trial-history CSV per
seed so the notebooks can compute ``T_{95}`` and convergence curves.
"""
from __future__ import annotations

import csv
from pathlib import Path

import optuna

from grace.eval.results import build_results_cache
from grace.eval.runner import evaluate_one
from grace.paths import optuna_dir


def optuna_search(
    model,
    tokenizer,
    judge,
    *,
    model_name: str,
    concept: str,
    method: str,
    layers: list[int],
    coefs: list[float],
    mode: str = "unconstrained",
    n_trials: int = 50,
    n_seeds: int = 3,
    judge_tag: str = "judge_gemma3_12b",
    n_questions: int | None = None,
    out_root: str | Path = "results/optuna",
    eval_root: str | Path = "results/steering_evaluations",
    vectors_root: str | Path = "vectors",
    overwrite: bool = False,
) -> list[dict]:
    """Run TPE for `n_seeds` seeds; return per-trial history rows for all seeds.

    Per-config eval CSVs reuse cache by default (skip-if-exists). Pass
    `overwrite=True` to re-evaluate and overwrite cached configs.
    """
    out_dir = optuna_dir(model_name, concept, method, mode, root=out_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    layers = [L for L in layers if L >= 1]
    if not layers:
        raise ValueError("No valid layers (>=1) to search; layer 0 is the embedding output.")

    all_history: list[dict] = []
    for seed in range(n_seeds):
        history_path = out_dir / f"trial_history_seed{seed}.csv"
        if not overwrite and history_path.exists():
            continue
        cache = (
            build_results_cache(model_name, concept, method, judge_tag=judge_tag, root=str(eval_root))
            if not overwrite else {}
        )
        sampler = optuna.samplers.TPESampler(seed=seed)
        study = optuna.create_study(direction="maximize", sampler=sampler)

        def objective(trial: optuna.Trial) -> float:
            layer = trial.suggest_categorical("layer", layers)
            coef = trial.suggest_categorical("coef", coefs)
            key = (layer, coef)
            if key in cache:
                return cache[key]
            summary = evaluate_one(
                model, tokenizer, judge,
                model_name=model_name, concept=concept, method=method,
                layer=layer, coef=coef,
                judge_tag=judge_tag, n_questions=n_questions, out_root=eval_root,
                vectors_root=vectors_root,
                overwrite=overwrite,
            )
            value = summary.get("mean_utility")
            if value is None:
                return 0.0
            cache[key] = value
            return value

        study.optimize(objective, n_trials=n_trials)

        with history_path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["trial", "layer", "coef", "value", "best_value_so_far"])
            best = -float("inf")
            for trial in study.trials:
                value = trial.value if trial.value is not None else float("nan")
                if value == value:  # not nan
                    best = max(best, value)
                row = {
                    "trial": trial.number,
                    "layer": trial.params.get("layer"),
                    "coef": trial.params.get("coef"),
                    "value": value,
                    "best_value_so_far": best,
                    "seed": seed,
                }
                w.writerow([row[k] for k in ("trial", "layer", "coef", "value", "best_value_so_far")])
                all_history.append(row)
    return all_history
