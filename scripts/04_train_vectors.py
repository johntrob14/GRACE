#!/usr/bin/env python3
"""Train one of the three steering-vector constructions for one (model, concept)."""
import argparse

from grace.vectors.train import train


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--concept", required=True)
    p.add_argument("--method", choices=("pv", "unit_mean", "cluster"), required=True)
    p.add_argument("--activation-type", default="response_avg",
                   choices=("response_avg", "prompt_last", "prompt_avg"))
    p.add_argument("--activations-root", default="activations")
    p.add_argument("--vectors-root", default="vectors")
    p.add_argument("--overwrite", action="store_true",
                   help="Re-train and overwrite an existing vector file.")
    args = p.parse_args()

    path = train(
        model_name=args.model,
        concept=args.concept,
        method=args.method,
        activation_type=args.activation_type,
        activations_root=args.activations_root,
        vectors_root=args.vectors_root,
        overwrite=args.overwrite,
    )
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
