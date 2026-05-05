#!/usr/bin/env python3
"""Re-score cached responses with a different judge — cross-judge robustness (§B).

Reads existing per_sample.csv files and runs them through one of the three
judges (local Gemma, OpenAI GPT-4.1-mini, AWS Bedrock Nova-2-Lite). Writes the
new scores under a new judge_tag directory.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from grace.eval.judge import BedrockJudge, LocalJudge, OpenAIJudge, clear_local_judge_cache, score_responses
from grace.eval.prompts import COHERENCE_PROMPT, build_concept_prompt


_JUDGE_TAG_BY_BACKEND = {
    "local": "judge_gemma3_12b",
    "openai": "judge_gpt4_1_mini",
    "bedrock": "judge_nova_2_lite",
}


def _make_judge(backend: str, model_id: str | None):
    if backend == "local":
        return LocalJudge(model_name=model_id or "google/gemma-3-12b-it")
    if backend == "openai":
        return OpenAIJudge(model_name=model_id or "gpt-4.1-mini")
    if backend == "bedrock":
        return BedrockJudge(model_id=model_id or "amazon.nova-lite-v1:0")
    raise ValueError(f"Unknown backend {backend!r}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--responses-dir", required=True,
                   help="Path to a directory containing *_per_sample.csv files.")
    p.add_argument("--backend", choices=("local", "openai", "bedrock"), default="local")
    p.add_argument("--judge-model", default=None)
    p.add_argument("--concept", required=True,
                   help="Concept name (used to fetch the rubric template).")
    p.add_argument("--rubric", required=True,
                   help="One-paragraph rubric to fill into the concept-score template.")
    p.add_argument("--dry-run-without-creds-for", default="",
                   help="Comma list of backends to skip silently if credentials are missing.")
    p.add_argument("--overwrite", action="store_true",
                   help="Re-score and overwrite existing per_sample.csv outputs in the target judge dir.")
    args = p.parse_args()

    skip = {b.strip() for b in args.dry_run_without_creds_for.split(",") if b.strip()}
    try:
        judge = _make_judge(args.backend, args.judge_model)
    except (RuntimeError, ImportError) as e:
        if args.backend in skip:
            print(f"[skip] {args.backend} judge unavailable: {e}")
            return
        raise

    out_tag = _JUDGE_TAG_BY_BACKEND[args.backend]
    responses_dir = Path(args.responses_dir)
    out_dir = responses_dir.parent.parent / out_tag / responses_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        for src in sorted(responses_dir.glob(f"{args.concept}_*_per_sample.csv")):
            dst = out_dir / src.name
            if not args.overwrite and dst.exists():
                print(f"[skip] {dst} (exists; pass --overwrite to re-score)")
                continue
            rows = list(csv.DictReader(src.open()))
            if not rows:
                continue
            questions = [r["question"] for r in rows]
            answers = [r["answer"] for r in rows]
            concept_prompts = [build_concept_prompt(args.concept, args.rubric, q, a) for q, a in zip(questions, answers)]
            coherence_prompts = [COHERENCE_PROMPT.format(question=q, answer=a) for q, a in zip(questions, answers)]
            cs = score_responses(judge, concept_prompts)
            ch = score_responses(judge, coherence_prompts)

            with dst.open("w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["question_id", "question", "answer", "concept_score_raw", "coherence_raw", "utility"])
                for i, (q, a, c, h) in enumerate(zip(questions, answers, cs, ch)):
                    u = ((c + h) / 2) if (c is not None and h is not None) else None
                    w.writerow([i, q, a, c, h, u])
            print(f"Wrote {dst}")
    finally:
        del judge
        if args.backend == "local":
            clear_local_judge_cache()


if __name__ == "__main__":
    main()
