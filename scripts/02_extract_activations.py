#!/usr/bin/env python3
"""Extract residual-stream activations for one (model, concept).

Greedily generates pos/neg responses for every (prompt-pair, question), records
prompt_avg, prompt_last, response_avg activations at every layer, and saves the
activations dict to ``activations/{model}/{concept}/activations_by_question.pt``.
"""
import argparse
from pathlib import Path

from grace.activations.extract import extract_concept_activations


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, help="HF model id, e.g. google/gemma-2-2b-it")
    p.add_argument("--concept", required=True)
    p.add_argument("--data-model", default="gpt-5", help="Subdirectory under concepts/ (default: gpt-5)")
    p.add_argument("--concepts-root", default="concepts", help="Top-level concepts directory")
    p.add_argument("--out-root", default="activations")
    p.add_argument("--questions", type=int, default=None, help="Cap on questions (smoke tests).")
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--overwrite", action="store_true",
                   help="Re-extract and overwrite an existing activations file.")
    args = p.parse_args()

    extract_root = Path(args.concepts_root) / args.data_model / "extract"
    path = extract_concept_activations(
        model_name=args.model,
        concept=args.concept,
        concept_root=extract_root,
        out_root=Path(args.out_root),
        max_questions=args.questions,
        max_new_tokens=args.max_new_tokens,
        overwrite=args.overwrite,
    )
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
