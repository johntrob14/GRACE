"""T_95: trials needed to recover 95 % of best-found utility — paper §4 (eq. 5)."""
from __future__ import annotations

from grace.analysis.load_results import load_optuna_history


def _t95_one_seed(rows: list[dict]) -> int | None:
    """Return the smallest trial index whose best_value_so_far ≥ 0.95 × max."""
    if not rows:
        return None
    rows = sorted(rows, key=lambda r: r["trial"])
    best = max(r["best_value_so_far"] for r in rows if r["best_value_so_far"] is not None)
    target = 0.95 * best
    for r in rows:
        if r["best_value_so_far"] is not None and r["best_value_so_far"] >= target:
            return int(r["trial"]) + 1  # 1-indexed trial count
    return None


def t95(
    model_name: str,
    concept: str,
    method: str,
    mode: str = "unconstrained",
    root: str = "results/optuna",
) -> tuple[float | None, dict[int, int]]:
    """Return ``(mean_T95_across_seeds, per_seed_T95)`` for one (concept, method, mode)."""
    rows = load_optuna_history(model_name, concept, method, mode, root=root)
    by_seed: dict[int, list[dict]] = {}
    for r in rows:
        by_seed.setdefault(r["seed"], []).append(r)
    per_seed: dict[int, int] = {}
    for seed, seed_rows in by_seed.items():
        v = _t95_one_seed(seed_rows)
        if v is not None:
            per_seed[seed] = v
    if not per_seed:
        return None, {}
    return sum(per_seed.values()) / len(per_seed), per_seed
