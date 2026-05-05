# Reproducing the paper

This walks through the full pipeline that produced every figure and table in
the NeurIPS submission. All commands are run from the repo root.

## Setup

```bash
uv sync                           # creates .venv from uv.lock
cp .env.example .env              # then edit to add HF_TOKEN
```

Optional extras: `uv sync --extra bedrock` enables the Nova-Lite cross-judge.

## Stage 1: Activation extraction (paper §A, §3)

For each of the 20 concepts, generate pos/neg responses with all 5 prompt
pairs × 100 extraction questions and cache the residual-stream activations:

```bash
for c in $(ls concepts/gpt-5/extract | sed 's/.json//'); do
  uv run python scripts/02_extract_activations.py \
    --model google/gemma-2-2b-it --concept "$c"
done
```

Repeat with `google/gemma-3-27b-it` and `meta-llama/Llama-3.3-70B-Instruct`.
Each model produces ~3.5 GB of cached activations per concept; you'll likely
want to point `--out-root` at fast local storage.

## Stage 2: Per-layer statistics (paper §3, §5, §6)

```bash
for c in $(ls concepts/gpt-5/extract | sed 's/.json//'); do
  uv run python scripts/03_compute_statistics.py \
    --model google/gemma-2-2b-it --concept "$c"
done
```

Statistics land under `results/statistics/{model}/{concept}_statistics.pt`
and are what every diagnostic + notebook reads.

## Stage 3: Vector training (paper §6)

Train all three vector variants:

```bash
for c in $(ls concepts/gpt-5/extract | sed 's/.json//'); do
  for m in pv unit_mean cluster; do
    uv run python scripts/04_train_vectors.py \
      --model google/gemma-2-2b-it --concept "$c" --method "$m"
  done
done
```

Vectors land under `vectors/{model}/{concept}/{method}/response_avg.pt`.

## Stage 4: §3 grid sweep (Figs. 2, 5, 6)

Sweep every-5-layers × {1, 2, 3} coefficients with PV vectors. Used for the
alignment-vs-concept-score plots:

```bash
for c in $(ls concepts/gpt-5/extract | sed 's/.json//'); do
  uv run python scripts/05_grid_search.py \
    --model google/gemma-3-27b-it --concept "$c" --method pv
done
```

Results land under `results/steering_evaluations/{model}/judge_gemma3_12b/{concept}/`.

## Stage 5: §4 TPE search (Figs. 1, 3)

Three sweeps per (model, concept, method):

```bash
for c in $(ls concepts/gpt-5/extract | sed 's/.json//'); do
  for mode in tpe_unconstrained tpe_top15_pl tpe_union_pl_ra; do
    uv run python scripts/06_optuna_search.py \
      --config configs/search/${mode}.yaml --concept "$c"
  done
done
```

Trial histories land under `results/optuna/{model}/{concept}/{method}_{mode}/`.

## Stage 6: ANOVA appendix (Appendix Table 6)

```bash
for c in $(ls concepts/gpt-5/extract | sed 's/.json//'); do
  uv run python scripts/09_run_anova.py \
    --model google/gemma-3-27b-it --concepts "$c" --build-ablation-vectors
done
```

The `--build-ablation-vectors` flag also constructs the prompt-weighted /
drop-worst-prompt / Question-SVD vectors used in §C.4.

## Stage 7: GRACE diagnostics (Figs. 10, 11, 12)

Once stages 1–3 are done, the diagnostic suite runs without GPUs:

```bash
uv run python scripts/08_run_diagnostics.py \
  --model google/gemma-3-27b-it \
  --concepts $(ls concepts/gpt-5/extract | sed 's/.json//')
```

This emits per-concept recommendations and writes them to
`results/diagnostics/{model}/{concept}.json`.

## Stage 8: Cross-judge robustness (Tables 4, 5)

For Tables 4 and 5 we re-score the existing Gemma-3-27B responses with
GPT-4.1-mini and Nova-Lite:

```bash
for c in $(ls concepts/gpt-5/extract | sed 's/.json//'); do
  RUBRIC=$(uv run python -c "import json; print(json.load(open(f'concepts/gpt-5/eval/$c.json'))['eval_prompt'])")
  uv run python scripts/10_rejudge.py \
    --responses-dir results/steering_evaluations/gemma-3-27b-it/judge_gemma3_12b/$c \
    --backend openai --concept "$c" --rubric "$RUBRIC"
  uv run python scripts/10_rejudge.py \
    --responses-dir results/steering_evaluations/gemma-3-27b-it/judge_gemma3_12b/$c \
    --backend bedrock --concept "$c" --rubric "$RUBRIC"
done
```

## Stage 9: Notebooks → figures

Once the results directories are populated, run the notebooks:

```bash
uv run jupyter nbconvert --to notebook --execute notebooks/01_alignment_predicts_layers.ipynb
uv run jupyter nbconvert --to notebook --execute notebooks/02_search_landscape_and_geo_search.ipynb
uv run jupyter nbconvert --to notebook --execute notebooks/03_granularity_correlations.ipynb
uv run jupyter nbconvert --to notebook --execute notebooks/04_anova_decomposition.ipynb
uv run jupyter nbconvert --to notebook --execute notebooks/05_grace_diagnostics.ipynb
uv run jupyter nbconvert --to notebook --execute notebooks/06_judge_robustness.ipynb
```

PNGs land in `Images/`. The exact filenames match what the paper's LaTeX
expects (including the `per-model-gran-steerabilitty.png` filename typo).

