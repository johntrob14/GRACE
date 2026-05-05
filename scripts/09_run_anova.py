#!/usr/bin/env python3
"""Two-way ANOVA decomposition (paper Appendix §C.4).

Computes η²_prompt, η²_question, η²_interaction per layer and reports the
layer-averaged means for each concept.

Optionally also builds the three §C.4 ablation vectors (prompt-weighted,
drop-worst-prompt, Question-SVD) for sanity-checking the §C.4 deltas.
Pass ``--build-ablation-vectors`` to enable.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from grace.activations.io import load_activations
from grace.diagnostics.anova import anova_decomposition
from grace.paths import model_safe


def _build_ablation_vectors(
    model_name: str, concept: str, activations_root: str, vectors_root: str,
    overwrite: bool = False,
) -> dict:
    """Compute the three Appendix §C.4 ablation vectors at every layer.

    Returns a dict ``{ablation_name: {layer: vector}}``. The ablations are:
    - ``prompt_weighted``:    inverse-interaction-weighted average across prompt pairs.
                              Empirically near-identical to PV (cos > 0.99 in paper).
    - ``drop_worst_prompt``:  remove the prompt pair with the highest interaction with
                              other prompt pairs, then average.
    - ``question_svd``:       top singular vector of the centered question-mean matrix.
                              Captures the axis of cross-question variation, *not* the concept.
    """
    from grace.vectors._common import n_pairs_in
    data = load_activations(model_name, concept, root=activations_root)
    layer_list = list(data["layer_list"])
    abq = data["activations_by_question"]
    P = n_pairs_in(abq)
    activation_type = "response_avg"

    out: dict[str, dict[int, torch.Tensor]] = {
        "prompt_weighted": {}, "drop_worst_prompt": {}, "question_svd": {},
    }

    for layer in layer_list:
        # Per-pair, per-question difference matrix: [P, Q, D]
        cells: list[list[torch.Tensor]] = [[] for _ in range(P)]
        for qd in abq.values():
            for p in range(P):
                if (
                    p < len(qd["pos"]) and p < len(qd["neg"])
                    and activation_type in qd["pos"][p]
                    and layer in qd["pos"][p][activation_type]
                ):
                    diff = qd["pos"][p][activation_type][layer] - qd["neg"][p][activation_type][layer]
                    cells[p].append(diff.float())
        if not all(cells):
            continue
        # Pad to common Q.
        Q = min(len(c) for c in cells)
        cells = [c[:Q] for c in cells]
        diff_tensor = torch.stack([torch.stack(c) for c in cells])  # [P, Q, D]
        prompt_means = diff_tensor.mean(dim=1)                       # [P, D]
        question_means = diff_tensor.mean(dim=0)                     # [Q, D]
        grand_mean = diff_tensor.mean(dim=(0, 1))                    # [D]

        # interaction term per (p, q): cells - prompt_means - question_means + grand_mean
        interaction = (
            diff_tensor
            - prompt_means.unsqueeze(1)
            - question_means.unsqueeze(0)
            + grand_mean
        )  # [P, Q, D]

        # prompt_weighted: 1 / (1 + ||interaction_p||²)
        per_prompt_inter_norm = interaction.flatten(1).norm(dim=1)  # [P]
        weights = 1.0 / (1.0 + per_prompt_inter_norm)
        weights = weights / weights.sum()
        out["prompt_weighted"][layer] = (weights.unsqueeze(1) * prompt_means).sum(dim=0)

        # drop_worst_prompt: drop the prompt with the highest interaction norm.
        worst = int(per_prompt_inter_norm.argmax())
        keep = [p for p in range(P) if p != worst]
        out["drop_worst_prompt"][layer] = prompt_means[keep].mean(dim=0)

        # question_svd: top singular vector of the centered question-mean matrix.
        centered = question_means - grand_mean
        try:
            _, _, vh = torch.linalg.svd(centered, full_matrices=False)
            out["question_svd"][layer] = vh[0]
        except Exception:
            pass

    if vectors_root:
        out_dir = Path(vectors_root) / model_safe(model_name) / concept / "anova"
        out_dir.mkdir(parents=True, exist_ok=True)
        for name, vectors in out.items():
            target = out_dir / f"{name}.pt"
            if not overwrite and target.exists():
                continue
            torch.save(vectors, target)

    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--concepts", nargs="+", required=True)
    p.add_argument("--activations-root", default="activations")
    p.add_argument("--out-root", default="results/anova")
    p.add_argument("--build-ablation-vectors", action="store_true")
    p.add_argument("--vectors-root", default="vectors")
    p.add_argument("--overwrite", action="store_true",
                   help="Re-run and overwrite existing ANOVA JSON / ablation vectors.")
    args = p.parse_args()

    out_dir = Path(args.out_root) / model_safe(args.model)
    out_dir.mkdir(parents=True, exist_ok=True)

    for concept in args.concepts:
        path = out_dir / f"{concept}.json"
        if not args.overwrite and path.exists():
            print(f"[skip] {path} (exists; pass --overwrite to recompute)")
        else:
            decomp = anova_decomposition(
                model_name=args.model,
                concept=concept,
                activation_type="response_avg",
                activations_root=args.activations_root,
            )
            path.write_text(json.dumps(decomp, indent=2))
            print(f"{concept:25s} mean η²:  prompt={decomp['mean_eta_sq_prompt']:.3f}  "
                  f"question={decomp['mean_eta_sq_question']:.3f}  "
                  f"interaction={decomp['mean_eta_sq_interaction']:.3f}")
        if args.build_ablation_vectors:
            _build_ablation_vectors(
                args.model, concept,
                activations_root=args.activations_root,
                vectors_root=args.vectors_root,
                overwrite=args.overwrite,
            )


if __name__ == "__main__":
    main()
