"""Path helpers for the artifacts the pipeline produces."""
from __future__ import annotations

from pathlib import Path

VECTOR_METHODS = ("pv", "unit_mean", "cluster")
ACTIVATION_TYPES = ("response_avg", "prompt_last", "prompt_avg")


def model_safe(model_name: str) -> str:
    """`google/gemma-2-2b-it` → `gemma-2-2b-it`."""
    return model_name.split("/")[-1] if "/" in model_name else model_name


def activations_path(model_name: str, concept: str, root: str | Path = "activations") -> Path:
    return Path(root) / model_safe(model_name) / concept / "activations_by_question.pt"


def statistics_path(model_name: str, concept: str, root: str | Path = "results/statistics") -> Path:
    return Path(root) / model_safe(model_name) / f"{concept}_statistics.pt"


def vector_path(
    model_name: str,
    concept: str,
    method: str,
    activation_type: str = "response_avg",
    root: str | Path = "vectors",
) -> Path:
    if method not in VECTOR_METHODS:
        raise ValueError(f"method must be one of {VECTOR_METHODS}, got {method!r}")
    if activation_type not in ACTIVATION_TYPES:
        raise ValueError(f"activation_type must be one of {ACTIVATION_TYPES}, got {activation_type!r}")
    return Path(root) / model_safe(model_name) / concept / method / f"{activation_type}.pt"


def cluster_selections_path(
    model_name: str,
    concept: str,
    root: str | Path = "vectors",
) -> Path:
    """Per-concept JSON recording which prompt-pair indices the cluster vector kept."""
    return Path(root) / model_safe(model_name) / concept / "cluster" / "selections.json"


def steering_eval_dir(
    model_name: str,
    judge_tag: str,
    concept: str,
    root: str | Path = "results/steering_evaluations",
) -> Path:
    """`judge_tag` is one of: `judge_gemma3_12b`, `judge_gpt4_1_mini`, `judge_nova_2_lite`."""
    return Path(root) / model_safe(model_name) / judge_tag / concept


def optuna_dir(
    model_name: str,
    concept: str,
    method: str,
    mode: str,
    root: str | Path = "results/optuna",
) -> Path:
    """`mode` is one of: `unconstrained`, `top15_pl`, `union_pl_ra`."""
    return Path(root) / model_safe(model_name) / concept / f"{method}_{mode}"
