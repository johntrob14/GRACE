#!/usr/bin/env python3
"""Evaluate one (concept, method, layer, coef) configuration.

Generation and judging run in two phases. The steered model is freed before
the judge is loaded so the two never share GPU residency.
"""
import argparse
import json

from grace.eval.judge import LocalJudge, clear_local_judge_cache, free_gpu_memory
from grace.eval.runner import generate_responses_one, score_responses_one
from grace.steering import load_model


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--concept", required=True)
    p.add_argument("--method", choices=("pv", "unit_mean", "cluster"), default="pv")
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--coef", type=float, required=True)
    p.add_argument("--judge-model", default="google/gemma-3-12b-it")
    p.add_argument("--judge-tag", default="judge_gemma3_12b")
    p.add_argument("--questions", type=int, default=None)
    p.add_argument("--out-root", default="results/steering_evaluations")
    p.add_argument("--vectors-root", default="vectors")
    p.add_argument("--overwrite", action="store_true",
                   help="Re-run and overwrite existing per-config eval CSVs.")
    args = p.parse_args()

    model, tokenizer = load_model(args.model)
    try:
        generate_responses_one(
            model, tokenizer,
            model_name=args.model, concept=args.concept, method=args.method,
            layer=args.layer, coef=args.coef,
            judge_tag=args.judge_tag, n_questions=args.questions, out_root=args.out_root,
            vectors_root=args.vectors_root,
            overwrite=args.overwrite,
        )
    finally:
        if hasattr(model, "cpu"):
            try:
                model.cpu()
            except Exception:
                pass
        del model, tokenizer
        free_gpu_memory()

    judge = LocalJudge(model_name=args.judge_model)
    try:
        summary = score_responses_one(
            judge,
            model_name=args.model, concept=args.concept, method=args.method,
            layer=args.layer, coef=args.coef,
            judge_tag=args.judge_tag, out_root=args.out_root,
            overwrite=args.overwrite,
        )
    finally:
        del judge
        clear_local_judge_cache()

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
