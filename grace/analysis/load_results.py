"""Load summary CSVs and Optuna trial histories across many concepts."""
from __future__ import annotations

import csv
from pathlib import Path

from grace.eval.results import read_summary
from grace.paths import optuna_dir, steering_eval_dir


def load_summary_results(
    model_name: str,
    concept: str,
    judge_tag: str = "judge_gemma3_12b",
    method: str | None = None,
    root: str = "results/steering_evaluations",
) -> list[dict]:
    """Walk one concept's eval directory and return one row per (method, layer, coef)."""
    eval_dir = steering_eval_dir(model_name, judge_tag, concept, root=root)
    if not eval_dir.exists():
        return []
    pattern = f"{concept}_{method}_layer*_coef*_summary.csv" if method else f"{concept}_*_layer*_coef*_summary.csv"
    rows: list[dict] = []
    for path in eval_dir.glob(pattern):
        row = read_summary(path)
        if row is None:
            continue
        rows.append(row)
    return rows


def load_optuna_history(
    model_name: str,
    concept: str,
    method: str,
    mode: str = "unconstrained",
    root: str = "results/optuna",
) -> list[dict]:
    """Concatenate trial-history CSVs across seeds for one (concept, method, mode)."""
    out_dir = optuna_dir(model_name, concept, method, mode, root=root)
    if not out_dir.exists():
        return []
    rows: list[dict] = []
    for path in sorted(out_dir.glob("trial_history_seed*.csv")):
        seed = int(path.stem.split("seed")[-1])
        with path.open() as f:
            reader = csv.DictReader(f)
            for row in reader:
                row["seed"] = seed
                row["trial"] = int(row["trial"])
                row["layer"] = int(row["layer"]) if row["layer"] not in ("", "None") else None
                row["coef"] = float(row["coef"]) if row["coef"] not in ("", "None") else None
                row["value"] = float(row["value"]) if row["value"] not in ("", "None") else None
                row["best_value_so_far"] = float(row["best_value_so_far"]) if row["best_value_so_far"] not in ("", "None") else None
                rows.append(row)
    return rows
