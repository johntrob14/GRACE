"""Read summary CSVs and build a (layer, coef) → utility result cache for search reuse.

Two summary schemas are supported, both with (concept, method, layer, coef)
recovered from the filename ``{concept}_{method}_layer{L}_coef{C}_summary.csv``:

- **Compact**: one header row + one data row, columns include
  ``mean_concept_score``, ``mean_coherence``, ``mean_utility``.
- **Multi-row**: the ``metric`` column is one of ``concept_score_raw``,
  ``coherence_raw``, etc. and the ``level`` column is one of ``overall`` or
  ``per_question``. We parse the ``overall`` rows.
"""
from __future__ import annotations

import csv
import re
from pathlib import Path

from grace.paths import VECTOR_METHODS, steering_eval_dir

# Anchor the method to the known set so multi-word concepts (e.g. `culinary_centric`)
# don't get split by the multi-word method `unit_mean`.
_METHOD_ALT = "|".join(re.escape(m) for m in VECTOR_METHODS)
_FILENAME_RE = re.compile(
    rf"^(?P<concept>.+)_(?P<method>{_METHOD_ALT})_layer(?P<layer>-?\d+)_coef(?P<coef>-?\d+(?:\.\d+)?)_summary\.csv$"
)


def _parse_filename(path: Path) -> dict[str, str | int | float] | None:
    m = _FILENAME_RE.match(path.name)
    if not m:
        return None
    g = m.groupdict()
    return {
        "concept": g["concept"],
        "method": g["method"],
        "layer": int(g["layer"]),
        "coef": float(g["coef"]),
    }


def read_summary(summary_path: Path) -> dict | None:
    if not summary_path.exists():
        return None
    with summary_path.open() as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None

    base = _parse_filename(summary_path) or {}

    # Compact schema: one data row with a mean_utility column.
    if len(rows) == 1 and "mean_utility" in rows[0]:
        row = rows[0]
        out = dict(base)
        for k in ("mean_concept_score", "mean_coherence", "mean_utility", "coef"):
            if k in row and row[k] != "":
                try:
                    out[k] = float(row[k])
                except ValueError:
                    pass
        for k in ("layer", "n_questions", "n_scored"):
            if k in row and row[k] != "":
                try:
                    out[k] = int(float(row[k]))
                except ValueError:
                    pass
        return out

    # Multi-row schema: scan for `metric` column with overall rows.
    out = dict(base)
    for row in rows:
        metric = (row.get("metric") or "").strip()
        level = (row.get("level") or "").strip()
        if level != "overall":
            continue
        try:
            mean = float(row.get("mean") or "")
        except ValueError:
            continue
        if metric == "concept_score_raw":
            out["mean_concept_score"] = mean
        elif metric == "coherence_raw":
            out["mean_coherence"] = mean
    if "mean_concept_score" in out and "mean_coherence" in out:
        out["mean_utility"] = (out["mean_concept_score"] + out["mean_coherence"]) / 2
    return out if "mean_utility" in out else None


def build_results_cache(
    model_name: str,
    concept: str,
    method: str,
    judge_tag: str = "judge_gemma3_12b",
    root: str = "results/steering_evaluations",
) -> dict[tuple[int, float], float]:
    """Walk the per-concept eval directory and return ``{(layer, coef): mean_utility}``."""
    eval_dir = steering_eval_dir(model_name, judge_tag, concept, root=root)
    if not eval_dir.exists():
        return {}
    cache: dict[tuple[int, float], float] = {}
    for path in eval_dir.glob(f"{concept}_{method}_layer*_coef*_summary.csv"):
        row = read_summary(path)
        if row is None:
            continue
        layer = row.get("layer")
        coef = row.get("coef")
        u = row.get("mean_utility")
        if layer is not None and coef is not None and u is not None:
            cache[(int(layer), float(coef))] = float(u)
    return cache
