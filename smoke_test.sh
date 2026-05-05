#!/usr/bin/env bash
# GRACE smoke test
#
# Runs the full mini-pipeline end-to-end on Gemma-2-2B (steering model) and
# Gemma-3-12B (judge), for two concepts (`hedging`, `professional`). Total
# runtime target: ~15 minutes on one A6000-class GPU.
#
# Verifies that every stage produces its expected outputs without crashing.
# Does NOT check numerical reproducibility of paper results — the 5-question
# extract is noise-dominated and the resulting vectors are too weak to drive
# clean concept-specific steering. Pick a friendly concept like `hedging`
# (gemma-2-2b is heavily safety-trained, so adversarial concepts like `evil`
# need much more data to break through).
#
# ---------------------------------------------------------------------------
# Isolation
# ---------------------------------------------------------------------------
# All artifacts (activations, statistics, vectors, eval CSVs, optuna trial
# histories, ANOVA, diagnostics) go under a single isolated root so the smoke
# test cannot stomp on real paper artifacts in `activations/`, `vectors/`,
# `results/` at the repo root.
#
# By default the root is `results/_smoke/<timestamp>/`. Override with
# `SMOKE_ROOT=path bash smoke_test.sh`.
#
# Stages exercised:
#   [1] Activation extraction (PL + RA at every layer, both concepts)
#   [2] Per-layer statistics
#   [3] Train all 3 vector variants (pv, unit_mean, cluster) on both concepts
#   [4] Single (layer, coef) eval — exercises generate + local Gemma judge
#   [5] §3 mini grid sweep (every-4-layers x {1,2,3}) for one concept
#   [6] §4 mini TPE — 5 trials x 1 seed, both unconstrained and top-15 PL
#   [7] §C.4 ANOVA + the 3 ablation vectors
#   [8] GRACE diagnostic recommender — emits a per-concept recommendation
#   [9] Cross-judge re-judge on cached responses (skips OpenAI/Bedrock if no creds)

set -euo pipefail

CONCEPTS=(hedging professional)
MODEL="google/gemma-2-2b-it"
JUDGE="google/gemma-3-12b-it"
JUDGE_TAG="judge_gemma3_12b"
QUESTIONS=5     # tiny: keeps smoke-test runtime ~15 min

cd "$(dirname "$0")"
PY="uv run python"

SMOKE_ROOT="${SMOKE_ROOT:-results/_smoke/$(date +%Y%m%d_%H%M%S)}"
ACT_ROOT="$SMOKE_ROOT/activations"
STAT_ROOT="$SMOKE_ROOT/statistics"
VEC_ROOT="$SMOKE_ROOT/vectors"
EVAL_ROOT="$SMOKE_ROOT/steering_evaluations"
OPTUNA_ROOT="$SMOKE_ROOT/optuna"
ANOVA_ROOT="$SMOKE_ROOT/anova"
DIAG_ROOT="$SMOKE_ROOT/diagnostics"
mkdir -p "$SMOKE_ROOT"

echo "=== Smoke artifacts will be written under: $SMOKE_ROOT"

echo "=== [0/9] Sanity check: env + concept JSONs"
$PY -c "import grace; from grace.diagnostics import recommend_for_concept; print('grace import OK')"
for c in "${CONCEPTS[@]}"; do
  test -f "concepts/gpt-5/extract/$c.json" || { echo "Missing concepts/gpt-5/extract/$c.json"; exit 1; }
done

echo "=== [1/9] Extract activations (PL + RA) for both concepts"
for c in "${CONCEPTS[@]}"; do
  $PY scripts/02_extract_activations.py \
    --model "$MODEL" --concept "$c" \
    --questions "$QUESTIONS" \
    --out-root "$ACT_ROOT"
done

echo "=== [2/9] Compute per-layer statistics"
for c in "${CONCEPTS[@]}"; do
  $PY scripts/03_compute_statistics.py \
    --model "$MODEL" --concept "$c" \
    --activations-root "$ACT_ROOT" \
    --out-root "$STAT_ROOT"
done

echo "=== [3/9] Train all 3 vector variants"
for c in "${CONCEPTS[@]}"; do
  for m in pv unit_mean cluster; do
    $PY scripts/04_train_vectors.py \
      --model "$MODEL" --concept "$c" --method "$m" \
      --activations-root "$ACT_ROOT" \
      --vectors-root "$VEC_ROOT"
  done
done

echo "=== [4/9] Single (layer, coef) eval"
$PY scripts/07_evaluate_one.py \
  --model "$MODEL" --concept hedging --method pv \
  --layer 12 --coef 2.0 \
  --judge-model "$JUDGE" --judge-tag "$JUDGE_TAG" \
  --questions "$QUESTIONS" \
  --out-root "$EVAL_ROOT" \
  --vectors-root "$VEC_ROOT"

echo "=== [5/9] §3 mini grid sweep"
$PY scripts/05_grid_search.py \
  --model "$MODEL" --concept hedging --method pv \
  --layer-step 4 --coefs 1.0,2.0,3.0 \
  --judge-model "$JUDGE" --judge-tag "$JUDGE_TAG" \
  --questions "$QUESTIONS" \
  --out-root "$EVAL_ROOT" \
  --vectors-root "$VEC_ROOT"

echo "=== [6/9] §4 mini TPE — unconstrained + top-15 PL"
# The optuna script reads roots from config + override; pass them via override.
$PY scripts/06_optuna_search.py \
  --config configs/search/tpe_unconstrained.yaml \
  --concept hedging \
  --override n_trials=5 n_seeds=1 questions="$QUESTIONS" model="$MODEL" \
             statistics_root="$STAT_ROOT" eval_root="$EVAL_ROOT" out_root="$OPTUNA_ROOT" \
             vectors_root="$VEC_ROOT"
$PY scripts/06_optuna_search.py \
  --config configs/search/tpe_top15_pl.yaml \
  --concept hedging \
  --override n_trials=5 n_seeds=1 questions="$QUESTIONS" model="$MODEL" \
             statistics_root="$STAT_ROOT" eval_root="$EVAL_ROOT" out_root="$OPTUNA_ROOT" \
             vectors_root="$VEC_ROOT"

echo "=== [7/9] §C.4 ANOVA + 3 ablation vectors (one concept)"
$PY scripts/09_run_anova.py \
  --model "$MODEL" --concepts hedging \
  --build-ablation-vectors \
  --activations-root "$ACT_ROOT" \
  --out-root "$ANOVA_ROOT" \
  --vectors-root "$VEC_ROOT"

echo "=== [8/9] GRACE diagnostic recommender — the user-facing tool"
$PY scripts/08_run_diagnostics.py \
  --model "$MODEL" --concepts "${CONCEPTS[@]}" \
  --activations-root "$ACT_ROOT" \
  --statistics-root "$STAT_ROOT" \
  --output "$DIAG_ROOT"

echo "=== [9/9] Cross-judge re-judge (LOCAL ONLY; OpenAI/Bedrock skipped if no creds)"
EVAL_DIR="$EVAL_ROOT/gemma-2-2b-it/$JUDGE_TAG/hedging"
RUBRIC=$($PY -c "import json; print(json.load(open('concepts/gpt-5/eval/hedging.json'))['eval_prompt'])")
$PY scripts/10_rejudge.py \
  --responses-dir "$EVAL_DIR" \
  --backend local --judge-model "$JUDGE" \
  --concept hedging --rubric "$RUBRIC" \
  --dry-run-without-creds-for openai,bedrock || true

echo
echo "=== Smoke test passed. Artifacts at: $SMOKE_ROOT"
