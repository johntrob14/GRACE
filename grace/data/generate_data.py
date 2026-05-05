"""Generate concept data via GPT-5 (Responses API).

Given a concept name and a one-paragraph description, GPT-5 produces 5
positive/negative instruction pairs, 200 questions (split 100 extraction +
100 held-out evaluation), and a per-concept rubric template. The output JSON
schema is the one consumed by `grace.activations.extract` and `grace.eval`.

This is a leaner port of the original `core/data/generate_data.py`. The local-
model paths and cost-tracking telemetry have been dropped — paper data was
generated exclusively with GPT-5.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import unicodedata
from pathlib import Path

from openai import OpenAI

from grace.config import require_env
from grace.data.prompts import PROMPTS

_REPLACEMENTS = {
    "’": "'", "‘": "'", "“": '"', "”": '"',
    "„": '"', "′": "'", "″": '"',
    "–": "-", "—": "--", "―": "--", "−": "-",
    "…": "...", " ": " ", "•": "*",
    "​": "", "‌": "", "‍": "", "﻿": "",
}


def normalize_unicode(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    for src, dst in _REPLACEMENTS.items():
        text = text.replace(src, dst)
    return "".join(c for c in text if c.isprintable() or c in "\n\r\t")


def parse_json_from_text(text: str) -> dict:
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        text = m.group(1)
    else:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            text = m.group(0)
    return json.loads(text)


def validate(data: dict) -> None:
    if not isinstance(data, dict):
        raise ValueError("Top-level output is not a dict.")
    if "instruction" not in data or len(data["instruction"]) != 5:
        raise ValueError("`instruction` must be a list of exactly 5 pos/neg pairs.")
    for i, pair in enumerate(data["instruction"]):
        if "pos" not in pair or "neg" not in pair:
            raise ValueError(f"instruction[{i}] is missing pos or neg.")
    if "questions" not in data or not (190 <= len(data["questions"]) <= 210):
        raise ValueError("`questions` must contain 190-210 items.")
    if "eval_prompt" not in data or not isinstance(data["eval_prompt"], str):
        raise ValueError("`eval_prompt` must be a string.")


def generate(concept: str, description: str, max_tokens: int = 12000) -> dict:
    api_key = require_env("OPENAI_API_KEY")
    client = OpenAI(api_key=api_key)
    formatted = PROMPTS["generate_concept"].format(
        CONCEPT=concept,
        concept_instruction=description,
        question_instruction="",
    )
    response = client.responses.create(
        model="gpt-5",
        input=formatted,
        max_output_tokens=max_tokens,
        reasoning={"effort": "low"},
    )
    raw = response.output_text
    data = parse_json_from_text(raw)

    # Normalize text fields.
    data["questions"] = [normalize_unicode(q) for q in data["questions"]]
    for pair in data["instruction"]:
        pair["pos"] = normalize_unicode(pair["pos"])
        pair["neg"] = normalize_unicode(pair["neg"])
    data["eval_prompt"] = normalize_unicode(data["eval_prompt"])

    validate(data)
    return data


def split_and_save(
    data: dict,
    concept: str,
    out_root: Path = Path("concepts/gpt-5"),
    seed: int | None = 0,
) -> tuple[Path, Path]:
    """Split 200 questions into 100 extract + 100 eval; write the two JSONs."""
    if seed is not None:
        random.seed(seed)
    questions = list(data["questions"])
    if len(questions) > 200:
        questions = questions[:200]
    if len(questions) < 200:
        raise ValueError(f"Need >=200 questions to split, got {len(questions)}.")
    random.shuffle(questions)

    extract = {"instruction": data["instruction"], "questions": questions[:100], "eval_prompt": data["eval_prompt"]}
    evaluate = {"instruction": data["instruction"], "questions": questions[100:200], "eval_prompt": data["eval_prompt"]}

    extract_path = out_root / "extract" / f"{concept}.json"
    eval_path = out_root / "eval" / f"{concept}.json"
    extract_path.parent.mkdir(parents=True, exist_ok=True)
    eval_path.parent.mkdir(parents=True, exist_ok=True)
    extract_path.write_text(json.dumps(extract, indent=2))
    eval_path.write_text(json.dumps(evaluate, indent=2))
    return extract_path, eval_path


def main():
    parser = argparse.ArgumentParser(description="Generate concept data via GPT-5.")
    parser.add_argument("--concept", required=True, help="Concept name (used as filename).")
    parser.add_argument("--description", required=True, help="One-paragraph human-written concept description.")
    parser.add_argument("--out-root", default="concepts/gpt-5")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=12000)
    args = parser.parse_args()

    data = generate(args.concept, args.description, max_tokens=args.max_tokens)
    extract_path, eval_path = split_and_save(data, args.concept, Path(args.out_root), args.seed)
    print(f"Wrote {extract_path}")
    print(f"Wrote {eval_path}")


if __name__ == "__main__":
    main()
