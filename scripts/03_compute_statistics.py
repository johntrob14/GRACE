#!/usr/bin/env python3
"""Compute per-layer statistics from cached activations."""
import argparse

from grace.activations.statistics import compute_statistics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--concept", required=True)
    p.add_argument("--activations-root", default="activations")
    p.add_argument("--out-root", default="results/statistics")
    p.add_argument("--overwrite", action="store_true",
                   help="Re-compute and overwrite an existing statistics file.")
    args = p.parse_args()

    path = compute_statistics(
        model_name=args.model,
        concept=args.concept,
        activations_root=args.activations_root,
        out_root=args.out_root,
        overwrite=args.overwrite,
    )
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
