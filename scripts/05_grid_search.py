#!/usr/bin/env python3
"""Grid sweep over (layer, coef) — paper §3.

Generation runs to completion across all (layer, coef) configs, the steered
model is freed, then the judge is loaded and scoring runs across all configs.
The two models never share GPU residency.
"""
import argparse

from grace.eval.judge import LocalJudge, clear_local_judge_cache, free_gpu_memory
from grace.search.grid import generate_grid, score_grid
from grace.steering import load_model


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--concept", required=True)
    p.add_argument("--method", choices=("pv", "unit_mean", "cluster"), default="pv")
    p.add_argument("--layer-step", type=int, default=5,
                   help="Sweep every N layers (paper default: 5).")
    p.add_argument("--coefs", default="1.0,2.0,3.0",
                   help="Comma-separated coefficients (paper default).")
    p.add_argument("--judge-model", default="google/gemma-3-12b-it")
    p.add_argument("--judge-tag", default="judge_gemma3_12b")
    p.add_argument("--questions", type=int, default=None)
    p.add_argument("--out-root", default="results/steering_evaluations")
    p.add_argument("--vectors-root", default="vectors")
    p.add_argument("--overwrite", action="store_true",
                   help="Re-run and overwrite existing per-config eval CSVs.")
    args = p.parse_args()

    coefs = [float(x) for x in args.coefs.split(",")]

    model, tokenizer = load_model(args.model)
    n_layers = model.config.num_hidden_layers if hasattr(model.config, "num_hidden_layers") \
        else model.config.text_config.num_hidden_layers
    layers = list(range(1, n_layers + 1, args.layer_step))

    try:
        generate_grid(
            model, tokenizer,
            model_name=args.model, concept=args.concept, method=args.method,
            layers=layers, coefs=coefs,
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
        summaries = score_grid(
            judge,
            model_name=args.model, concept=args.concept, method=args.method,
            layers=layers, coefs=coefs,
            judge_tag=args.judge_tag, out_root=args.out_root,
            overwrite=args.overwrite,
        )
    finally:
        del judge
        clear_local_judge_cache()

    print(f"Wrote {len(summaries)} (layer, coef) summaries.")


if __name__ == "__main__":
    main()
