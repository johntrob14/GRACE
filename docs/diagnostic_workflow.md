# GRACE Diagnostic Workflow

This walks through how to use GRACE to decide:

1. Which **vector construction** (`pv`, `unit_mean`, `cluster`) to prefer
   for a new concept.
2. Which **search space** (full layer range, top-15 PL, or top-15 PL ∪ RA)
   to constrain TPE to.
3. Whether the concept is intrinsically high-granularity (i.e. expect a
   lower steerability ceiling regardless of construction).

All diagnostics consume cached activations and per-layer statistics, so the
workflow runs in seconds on CPU once those are built.

## Inputs

For each (model, concept) pair you want to diagnose, the workflow needs:

```
activations/{model}/{concept}/activations_by_question.pt    # from scripts/02
results/statistics/{model}/{concept}_statistics.pt          # from scripts/03
```

## One-shot: `scripts/08_run_diagnostics.py`

```bash
uv run python scripts/08_run_diagnostics.py \
  --model google/gemma-3-27b-it \
  --concepts evil sycophantic golden_gate_centric
```

emits a recommendation block per concept. Sample output:

```
CONCEPT: hallucinating  (google/gemma-2-2b-it)
  Granularity G_c     = 0.886
  PL/RA correlation   = 0.61
  Magnitude CV        = 0.18
  Block-structure fr. = 90%  outlier pairs: [3, 4]
  Top-15 PL layers    = [10, 12, 15, 18, ...]
  >>> Recommendation:
        construction:  cluster
        search space:  top15_pl  (15 layers)
        search budget: 50 trials × 3 seeds
  Rationale:
    - granularity G_c = 0.886
    - PL/RA correlation 0.61 ≥ 0.20; PL alone is sufficient.
    - persistent block structure on 90% of layers — prefer cluster.
```

The same recommendations are written as JSON to
`results/diagnostics/{model}/{concept}.json` for use in notebooks /
downstream scripts.

## What each diagnostic measures

### Prompt-boundary alignment 𝒜_c(ℓ)

Average pairwise cosine similarity of the unit-normalized prompt-boundary
difference vectors at layer ℓ. High alignment indicates a broadly consistent
direction at that layer. The **top-15 layers by 𝒜_c^PL** form the default
search space (paper §4.2).

### Concept granularity G_c

The paper's `pl_cross_within` metric, equal to ``mean_ℓ γ_c(ℓ) / 𝒜_c(ℓ)``,
where γ_c(ℓ) is the within-question component of alignment. Values near 1
mean "stable concept direction across contexts"; values well above 1 mean
"locally agreed-upon direction rotates systematically across questions".

High G_c is associated with a lower best-found steerability ceiling and
slower TPE convergence (paper §5).

### Magnitude CV (`unit_mean` trigger)

Coefficient of variation of difference-vector magnitudes per layer. A high
CV indicates a heavy-tailed magnitude distribution where a few high-norm
samples can dominate the PV average.

The paper's empirical finding is that this signal **only generalizes on
Gemma-3-27B**; on Gemma-2-2B and Llama-3.3-70B the signal is weak. The
recommender encodes this with model-specific thresholds (`+∞` for the two
models where the signal doesn't hold).

### Per-pair similarity heatmap (`cluster` trigger)

For each layer, the P × P cosine-similarity matrix of the per-prompt-pair
mean directions. Persistent block structure across many layers — a subset
of pairs that mutually agree but disagree with the others — is the trigger
for using `cluster` construction.

Without a small pilot evaluation, the recommender applies `cluster`
conservatively (only when the block structure shows on ≥ 60 % of layers).
For borderline cases, run a pilot eval to confirm before switching off PV.

### PL/RA correlation (search-space trigger)

Pearson correlation between the PL alignment profile and the RA alignment
profile across layers. When this falls below 0.2, the two variants disagree
about which layers are best, and constraining to top-15 PL alone risks
missing the true peak. The remedy is to widen the search space to the
**union of top-15 PL ∪ top-15 RA** (paper §6, `tpe_union_pl_ra`).

### ANOVA decomposition (informational only)

The two-way ANOVA in `grace.diagnostics.anova` decomposes total directional
variance into η²_prompt, η²_question, η²_interaction. The Question effect
is the binding constraint on rank-1 steering: when η²_question is high, no
single direction can satisfy all input contexts. This is informational —
the recommender uses granularity G_c (which is correlated with η²_question)
as the operational signal.

## Programmatic use

You can call the recommender from Python:

```python
from grace.diagnostics import recommend_for_concept

rec = recommend_for_concept("google/gemma-3-27b-it", "sycophantic")
print(rec.construction)        # 'pv' | 'unit_mean' | 'cluster'
print(rec.search_space)        # 'top15_pl' | 'union_pl_ra'
print(rec.search_layers)       # list of layer indices
print(rec.granularity)         # G_c
print(rec.rationale)           # list of human-readable reasons
```
