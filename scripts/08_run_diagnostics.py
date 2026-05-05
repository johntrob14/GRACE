#!/usr/bin/env python3
"""Run the full GRACE diagnostic suite from cached activations + statistics.

For each (model, concept) pair this:
1. Computes 𝒜_c(ℓ), G_c, magnitude CV, PL/RA correlation, per-pair heatmap.
2. Emits a recommendation block (which construction + which search space).
3. Writes everything as JSON to ``results/diagnostics/{model}/{concept}.json``.

No GPU is required — everything is computed from cached `*.pt` files.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from grace.diagnostics.recommend import format_recommendation, recommend_for_concept
from grace.paths import model_safe


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--concepts", nargs="+", required=True)
    p.add_argument("--activations-root", default="activations")
    p.add_argument("--statistics-root", default="results/statistics")
    p.add_argument("--output", default="results/diagnostics")
    p.add_argument("--overwrite", action="store_true",
                   help="Re-run and overwrite existing diagnostic JSONs.")
    args = p.parse_args()

    out_dir = Path(args.output) / model_safe(args.model)
    out_dir.mkdir(parents=True, exist_ok=True)

    for concept in args.concepts:
        out_path = out_dir / f"{concept}.json"
        if not args.overwrite and out_path.exists():
            print(f"[skip] {out_path} (exists; pass --overwrite to recompute)")
            continue
        rec = recommend_for_concept(
            model_name=args.model,
            concept=concept,
            activations_root=args.activations_root,
            statistics_root=args.statistics_root,
        )
        print(format_recommendation(rec))
        print()
        out_path.write_text(json.dumps(rec.as_dict(), indent=2))


if __name__ == "__main__":
    main()
