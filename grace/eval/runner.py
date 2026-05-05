"""End-to-end eval for one (concept, method, layer, coef) configuration.

Generates responses on the held-out evaluation question set and scores each
with the configured judge along two axes (concept score, coherence). The
primary steering utility ``U_c`` reported throughout the paper is the
arithmetic mean of these two (paper §A.1, eq. 4).

Outputs two CSVs per (concept, method, layer, coef):

- {tag}_per_sample.csv:  one row per held-out question
- {tag}_summary.csv:     one row with the aggregate metrics

These match the schema the public-repo notebooks load.

Layer-index convention
----------------------
Throughout this codebase ``layer`` is the *hidden_states* index, matching the
extracted activations file: ``hidden_states[0]`` is the embedding output and
``hidden_states[L]`` for ``L >= 1`` is the output of transformer block
``model.layers[L-1]``. Saved steering vectors are keyed by this same index.
When applying a vector at layer ``L`` we therefore hook ``model.layers[L-1]``.
Layer 0 has no transformer-block hook position and is rejected.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import torch

from grace.eval.generate import generate_responses
from grace.eval.judge import LocalJudge, score_responses
from grace.eval.prompts import COHERENCE_PROMPT, build_concept_prompt
from grace.paths import steering_eval_dir, vector_path


def _load_concept_eval(concept: str, root: Path = Path("concepts/gpt-5/eval")) -> dict:
    return json.loads((root / f"{concept}.json").read_text())


def _read_summary_csv(path: Path) -> dict:
    with path.open() as f:
        rows = list(csv.reader(f))
    if len(rows) < 2:
        return {}
    keys, vals = rows[0], rows[1]
    out: dict = {}
    for k, v in zip(keys, vals):
        if v == "" or v == "None":
            out[k] = None
            continue
        try:
            out[k] = int(v) if k in {"layer", "n_questions", "n_scored"} else float(v)
        except ValueError:
            out[k] = v
    return out


def _load_steering_vector(
    model_name: str, concept: str, method: str, layer: int,
    vectors_root: str | Path = "vectors",
) -> torch.Tensor:
    vp = vector_path(model_name, concept, method, root=vectors_root)
    vectors = torch.load(vp, weights_only=False)
    if layer not in vectors:
        raise KeyError(f"layer {layer} missing from {vp}; available: {sorted(vectors)}")
    return vectors[layer]


def evaluate_one(
    model,
    tokenizer,
    judge,
    *,
    model_name: str,
    concept: str,
    method: str,
    layer: int,
    coef: float,
    judge_tag: str = "judge_gemma3_12b",
    n_questions: int | None = None,
    max_new_tokens: int = 256,
    out_root: Path | str = "results/steering_evaluations",
    vectors_root: Path | str = "vectors",
    overwrite: bool = False,
) -> dict:
    if layer < 1:
        raise ValueError(
            f"layer must be >= 1 (hidden_states convention; layer 0 is the "
            f"embedding output and has no transformer-block hook position). "
            f"Got layer={layer}."
        )

    out_dir = steering_eval_dir(model_name, judge_tag, concept, root=out_root)
    tag = f"{concept}_{method}_layer{layer}_coef{coef}"
    per_sample_path = out_dir / f"{tag}_per_sample.csv"
    summary_path = out_dir / f"{tag}_summary.csv"
    if not overwrite and per_sample_path.exists() and summary_path.exists():
        return _read_summary_csv(summary_path)

    spec = _load_concept_eval(concept)
    rubric = spec["eval_prompt"]
    questions = spec["questions"]
    if n_questions is not None:
        questions = questions[:n_questions]

    vector = _load_steering_vector(model_name, concept, method, layer, vectors_root=vectors_root)
    responses = generate_responses(
        model, tokenizer, questions,
        steering_vector=vector, layer_idx=layer - 1, coef=coef,
        max_new_tokens=max_new_tokens,
    )

    concept_prompts = [build_concept_prompt(concept, rubric, q, a) for q, a in zip(questions, responses)]
    coherence_prompts = [COHERENCE_PROMPT.format(question=q, answer=a) for q, a in zip(questions, responses)]
    concept_scores = score_responses(judge, concept_prompts)
    coherence_scores = score_responses(judge, coherence_prompts)
    utility_per_q = [
        ((c + h) / 2.0) if (c is not None and h is not None) else None
        for c, h in zip(concept_scores, coherence_scores)
    ]

    out_dir.mkdir(parents=True, exist_ok=True)

    with per_sample_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["question_id", "question", "answer", "concept_score_raw", "coherence_raw", "utility"])
        for i, (q, a, cs, ch, u) in enumerate(zip(questions, responses, concept_scores, coherence_scores, utility_per_q)):
            w.writerow([i, q, a, cs, ch, u])

    valid_u = [u for u in utility_per_q if u is not None]
    valid_cs = [c for c in concept_scores if c is not None]
    valid_ch = [h for h in coherence_scores if h is not None]
    summary = {
        "concept": concept, "method": method, "layer": layer, "coef": coef,
        "n_questions": len(questions),
        "n_scored": len(valid_u),
        "mean_concept_score": sum(valid_cs) / len(valid_cs) if valid_cs else None,
        "mean_coherence": sum(valid_ch) / len(valid_ch) if valid_ch else None,
        "mean_utility": sum(valid_u) / len(valid_u) if valid_u else None,
    }
    with summary_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(list(summary.keys()))
        w.writerow(list(summary.values()))
    return summary


def get_default_judge() -> LocalJudge:
    return LocalJudge()
