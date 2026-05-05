# GRACE: Granularity- and Representation-Aware Concept Engineering

Code for the for our research titled **"When Is Rank-1 Steering Cheap?
Geometry, Granularity, and Budgeted Search."**

GRACE is a diagnostic workflow for rank-1 activation steering. It uses cheap
geometric statistics computed from contrastive activations — no steering
search needed — to predict where useful steering directions emerge in a model,
how expensive a concept will be to optimize, and which of three vector
constructions (`pv`, `unit_mean`, `cluster`) and search strategies are likely
to help.

## What's included

- **`grace/`** — the importable Python package.
  - `data/`        Concept-data generation (GPT-5 → JSON pairs/questions/rubric).
  - `activations/` Residual-stream extraction (prompt-boundary + response-averaged) and per-layer statistics.
  - `vectors/`     Three vector constructions: `pv`, `unit_mean`, `cluster`.
  - `steering/`    Inference-time activation steering (forward hook).
  - `eval/`        Local Gemma-3-12b-it judge by default; optional GPT-4.1-mini and Nova-2-Lite plugins.
  - `search/`      Grid sweep (§3) and Optuna TPE search (§4).
  - `diagnostics/` Alignment 𝒜, granularity G, magnitude CV, PL/RA correlation, per-pair heatmap, ANOVA, and the workflow recommender.
  - `analysis/`    Cross-concept aggregation helpers used by the notebooks.
- **`scripts/`** — 10 CLI entry points, one per pipeline stage.
- **`configs/`** — YAML configs for data generation, vector training, search, and eval.
- **`concepts/gpt-5/`** — the 20 concept JSONs (5 prompt pairs × 100 extraction + 100 held-out questions).
- **`results/`** — pre-computed per-concept statistics, summary CSVs, and Optuna trial histories so the notebooks reproduce all paper figures without re-running the (expensive) generation/judging pipeline.
- **`notebooks/`** — six notebooks that produce every figure in the paper.
- **`docs/`** — `reproducing_the_paper.md`, `diagnostic_workflow.md`, `adding_new_concepts.md`.
- **`smoke_test.sh`** — full mini-pipeline on Gemma-2-2B + Gemma-3-12B judge, two concepts, ~15 min on one GPU.

## Setup

```bash
uv sync
cp .env.example .env
# edit .env to add HF_TOKEN (required) and OPENAI_API_KEY / AWS_* (optional)
```

The Bedrock judge is gated behind an extra:

```bash
uv sync --extra bedrock
```

## Smoke test

```bash
bash smoke_test.sh
```

Runs the full pipeline (extract → statistics → train all 3 vectors → eval one
config → mini grid sweep → mini TPE search → ANOVA → diagnostics → cross-judge)
on one GPU in ~15 minutes. It only checks that every stage runs to completion,
not numerical reproducibility.

## Reproducing paper figures

The notebooks load from `results/` so they run without re-doing any
generation. Each notebook header lists the figures it produces:

| Notebook | Paper figures |
|----------|---------------|
| `notebooks/01_alignment_predicts_layers.ipynb` | Figs. 2, 5, 6 |
| `notebooks/02_search_landscape_and_geo_search.ipynb` | Figs. 1, 3, 9 |
| `notebooks/03_granularity_correlations.ipynb` | Figs. 4, 7, 8 |
| `notebooks/04_anova_decomposition.ipynb` | Appendix Table 6 |
| `notebooks/05_grace_diagnostics.ipynb` | Figs. 10, 11, 12 |
| `notebooks/06_judge_robustness.ipynb` | Tables 4, 5 |

See [`docs/reproducing_the_paper.md`](docs/reproducing_the_paper.md) for the
full pipeline-from-scratch recipe (extracting activations, training vectors,
running TPE sweeps, etc.).

## Diagnostic workflow

If you have steering vectors trained on a new concept and want to know
whether to (a) prefer `unit_mean`/`cluster` over `pv`, (b) restrict the
search to the top-15 alignment-ranked layers, or (c) widen the search space
because of representational fragmentation, run:

```bash
uv run python scripts/08_run_diagnostics.py \
    --model google/gemma-3-27b-it \
    --concepts evil sycophantic golden_gate_centric \
    --output results/diagnostics/
```

This reads cached activations, computes the full GRACE diagnostic suite, and
prints a per-concept recommendation. See
[`docs/diagnostic_workflow.md`](docs/diagnostic_workflow.md) for details.

## Citation

```bibtex
@inproceedings{grace2026,
  title  = {When Is Rank-1 Steering Cheap? Geometry, Granularity, and Budgeted Search},
  author = {Anonymous},
  year   = {2026},
}
```

## Notes

- Granularity is the pairwise-cosine formulation
  G_c(ℓ) = γ_c(ℓ) / 𝒜_c(ℓ), implemented in
  [grace/diagnostics/granularity.py](grace/diagnostics/granularity.py).
- The figure file `Images/per-model-gran-steerabilitty.png` keeps the typo
  ("steerabilitty") in its filename to match the paper's `\includegraphics`
  reference.
