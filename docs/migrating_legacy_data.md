# Packaging legacy `granular_steering` data for migration into GRACE

You are Claude, running on the server that holds the **legacy** `~/granular_steering`
repository. The owner has a new, slimmed-down public companion repo named
**GRACE** on a different machine. This document tells you exactly how to
package the existing artifacts so they can be `rsync`'d into GRACE without
further conversion. Read it end-to-end before doing anything.

The GRACE repo's pipeline is a strict subset of `granular_steering`. Most
legacy artifacts have a direct equivalent in GRACE; many do not, and **must
not** be packaged. The "Out of scope" list at the end is binding.

---

## Target layout (GRACE)

Everything you produce should land under a single staging directory
(call it `~/grace_migration/`) with this structure. After staging, the
contents of `~/grace_migration/` are meant to be merged with `rsync -av`
into the root of the GRACE repo on the other machine.

```
grace_migration/
├── concepts/
│   └── gpt-5/
│       ├── extract/{concept}.json          # extraction prompt-pairs + questions
│       └── eval/{concept}.json             # eval-time questions (held-out set)
│
├── activations/
│   └── {model_safe}/{concept}/activations_by_question.pt
│
├── vectors/
│   └── {model_safe}/{concept}/
│       ├── pv/response_avg.pt              # vanilla mean-of-diffs
│       ├── unit_mean/response_avg.pt       # unit-normalize then average then rescale
│       ├── cluster/response_avg.pt         # subset selected by cosine > 0.7
│       ├── cluster/selections.json         # which prompt-pair indices were kept
│       └── anova/                          # only if ANOVA-construction vectors exist
│           ├── prompt_weighted.pt
│           ├── question_svd.pt
│           └── drop_worst_prompt.pt
│
└── results/
    ├── statistics/{model_safe}/{concept}_statistics.pt
    ├── steering_evaluations/{model_safe}/{judge_tag}/{concept}/
    │       {concept}_{method}_layer{L}_coef{C}_summary.csv
    │       {concept}_{method}_layer{L}_coef{C}_per_sample.csv
    ├── optuna/{model_safe}/{concept}/{method}_{mode}/trial_history_seed{N}.csv
    ├── diagnostics/{model_safe}/{concept}.json    # if present in legacy
    └── anova/{model_safe}/{concept}.json          # if present in legacy
```

Allowed value enums (anything outside these is out of scope):

- `model_safe` — short name only, no org prefix. e.g. `gemma-2-2b-it`,
  `gemma-3-27b-it`, `Llama-3.3-70B-Instruct` (NOT `google/gemma-2-2b-it`).
- `method` — exactly one of: `pv`, `unit_mean`, `cluster`.
- `mode` (optuna only) — exactly one of: `unconstrained`, `top15_pl`,
  `union_pl_ra`.
- `judge_tag` — exactly one of: `judge_gemma3_12b`, `judge_gpt4_1_mini`,
  `judge_nova_2_lite`. (No `judge_gpt5`.)
- Vector activation type — for the paper, only `response_avg.pt`.
  `prompt_last.pt` / `prompt_avg.pt` exist in the schema but the paper does
  not use them; do not migrate them unless the user explicitly asks.

---

## Legacy → GRACE mapping (concrete)

### Concepts

```
granular_steering/concepts/concept_data_extract/gpt-5/{c}.json
    → grace_migration/concepts/gpt-5/extract/{c}.json

granular_steering/concepts/concept_data_eval/gpt-5/{c}.json
    → grace_migration/concepts/gpt-5/eval/{c}.json
```

The bare `granular_steering/concepts/{c}.json` files (no `gpt-5/` prefix) are
the older single-source variants — only migrate them as `extract/` if no
`concept_data_extract/gpt-5/{c}.json` exists.

### Activations

The on-disk format is identical between repos — the `activations_by_question.pt`
payload is a dict with keys `model`, `concept`, `layer_list`,
`activations_by_question`. **Do not** re-extract; just copy.

```
granular_steering/activations/{model_safe}/{c}/activations_by_question.pt
    → grace_migration/activations/{model_safe}/{c}/activations_by_question.pt
```

### Vectors (the construction rename is the important part)

The legacy repo lumps everything under `DiffMeans/` and uses long suffixed
filenames. GRACE has three constructions, each in its own subfolder, with the
filename naming the activation type:

| Legacy file (`vectors/{m}/{c}/DiffMeans/...`)           | GRACE target (`vectors/{m}/{c}/...`)         |
|---------------------------------------------------------|-----------------------------------------------|
| `response_avg_diff.pt`                                  | `pv/response_avg.pt`                          |
| `directional_diff_response_avg.pt`                      | `unit_mean/response_avg.pt`                   |
| `response_avg_diff_cluster.pt`                          | `cluster/response_avg.pt`                     |
| `cluster_selections.json`                               | `cluster/selections.json`                     |

If you see `directional_act_response_avg.pt`, that's the activation-normalized
variant — **not** what GRACE calls `unit_mean`. Skip it. GRACE's `unit_mean`
is the diff-normalized variant only (legacy `directional_diff_*`).

If the user has produced ANOVA-leverage vectors (the three constructions used
in Appendix C.4), package them under `vectors/{m}/{c}/anova/`:

| Legacy file                                        | GRACE target                              |
|----------------------------------------------------|-------------------------------------------|
| anything named `prompt_weighted*.pt`               | `anova/prompt_weighted.pt`                |
| anything named `question_svd*.pt`                  | `anova/question_svd.pt`                   |
| anything named `drop_worst_prompt*.pt`             | `anova/drop_worst_prompt.pt`              |

Find these with `find vectors -name 'prompt_weighted*.pt' -o -name
'question_svd*.pt' -o -name 'drop_worst_prompt*.pt'` and confirm with the
user before mapping if the names don't match exactly.

### Statistics

GRACE expects a single `.pt` per (model, concept) at
`results/statistics/{model_safe}/{concept}_statistics.pt`. Inside it must be
a dict with at least these top-level keys: `concept`, `model`, `layer_list`,
`prompt_paired`, `all_pairs`, `pos_alignment`, `neg_alignment` — each of the
last four with a nested `by_layer` dict keyed by integer layer.

Search the legacy repo for any precomputed statistics file for each (model,
concept). If you find one, **show the user a `torch.load(...).keys()` dump
before copying**, since legacy statistics formats vary. If a file's schema
matches, copy it to the GRACE path. If it doesn't (or none exists), do not
fabricate one — note it in your report and let the user re-run
`scripts/03_compute_statistics.py` on the GRACE side.

### Steering evaluations (CSV pairs)

The legacy repo has a doubled `steering_evaluations/steering_evaluations/...`
prefix. Strip it.

```
granular_steering/steering_evaluations/steering_evaluations/{m}/{judge_tag}/{c}/
    {c}_{method}_layer{L}_coef{C}_summary.csv
    {c}_{method}_layer{L}_coef{C}_per_sample.csv

    → grace_migration/results/steering_evaluations/{m}/{judge_tag}/{c}/<same filenames>
```

Filter rules **before copying**:

1. **`method` must be in `{pv, unit_mean, cluster}`.** Drop any file whose
   filename method group is `dir_diff_pv`, `directional`, `bipo`, `reps`,
   `canonical_pv`, `holdout_*`, `neutral`, `bestpair`, `gran_prompt`,
   `gran_context`, `1prompt`, `small`, `behavioral`, `allpairs`, etc.
2. **Filename regex** (anchored — multi-word concepts like `culinary_centric`
   collide with multi-word methods like `unit_mean` if you split naively):
   ```
   ^(?P<concept>.+)_(?P<method>pv|unit_mean|cluster)_layer-?\d+_coef-?\d+(?:\.\d+)?_(?:summary|per_sample)\.csv$
   ```
   Do not rename anything that already matches.
3. Skip any directory containing `.smoke_overwrite_backup` or similar
   ad-hoc backup suffixes.
4. Both `summary.csv` and `per_sample.csv` are useful — copy both. The
   summary CSV is the authoritative scoreboard; the per-sample CSV holds the
   raw judge outputs and is needed for re-judging.

### Optuna trial histories

The legacy layout splits results across two top-level dirs and adds a
`seed_{N}/` subdir; GRACE flattens both into a single `mode` suffix on the
method directory and renames the trial CSV.

| Legacy path                                                                                  | GRACE target                                                              |
|----------------------------------------------------------------------------------------------|---------------------------------------------------------------------------|
| `optuna_results/{m}/{c}/{method}_budget50_seeds3/seed_{N}/trial_history.csv`                 | `results/optuna/{m}/{c}/{method}_unconstrained/trial_history_seed{N}.csv` |
| `optuna_results_top15pl/{m}/{c}/{method}_budget50_seeds3/seed_{N}/trial_history.csv`         | `results/optuna/{m}/{c}/{method}_top15_pl/trial_history_seed{N}.csv`      |
| any "union of top15 PL ∪ top15 RA" search dir (legacy may not have one)                      | `results/optuna/{m}/{c}/{method}_union_pl_ra/trial_history_seed{N}.csv`   |

Drop everything in those dirs other than `trial_history.csv` (the
`search_config.yaml`, `summary.csv`, `summary.json` files do not have an
equivalent in GRACE — leave them out).

If `method` in the legacy folder is `dir_diff_pv` (or any other
non-{pv,unit_mean,cluster} value), skip the whole search.

### Diagnostics + ANOVA JSON outputs

GRACE keeps single-file-per-concept diagnostic and ANOVA reports:

```
results/diagnostics/{m}/{c}.json   # output of grace/diagnostics/recommend.py
results/anova/{m}/{c}.json         # per-layer eta-squared decomposition
```

If the legacy repo has notebook-generated equivalents (look near
`notebooks/anova_*` or in any consolidated-results JSON), inspect their schema
and only migrate if it matches the keys below. Otherwise skip — the user can
regenerate on the GRACE side.

- `diagnostics/{c}.json` minimum keys: `model_name`, `concept`, `granularity`,
  `top15_pl_layers`, `pl_ra_correlation`, `magnitude_cv`,
  `block_structure_fraction`, `construction`, `search_space`, `search_layers`.
- `anova/{c}.json` minimum keys: `per_layer` (dict keyed by layer string;
  each value has `eta_sq_prompt`, `eta_sq_question`, `eta_sq_interaction`).

---

## Out of scope (do NOT migrate)

These exist in the legacy repo and will pollute GRACE if copied. Skip on
sight, no exceptions:

- **Vector constructions other than pv / unit_mean / cluster.** That means:
  `dir_diff_pv`, `directional_act_*`, `BiPO/`, `RePS/`, `canonical_pv`,
  `holdout_*`, `neutral`, `bestpair`, `gran_prompt`, `gran_context`,
  `1prompt`, `small`, `behavioral_*`, `allpairs`, `directional_*_allpairs`,
  `*_ci_*`, `*_cosine_*`.
- **Search modes other than unconstrained / top15_pl / union_pl_ra.**
- **Patching / detection / vocab-projection experiments** — anything under
  `run_patching_experiment.py`, `test_concept_detection.py`,
  `vocab_projection.py`, etc.
- **Activation files for `_neutral` concepts** (`{c}_neutral`).
- **`prompt_last` and `prompt_avg` vector files**, unless the user explicitly
  asks for them.
- **Old eval CSVs from non-`gpt-5` concept revisions** (any concept whose
  source JSON didn't come from `concept_data_extract/gpt-5/`).
- **`steering_eval.zip`, `report-2026-03-28.md`, `requirements.txt`,
  `setup_ssd_symlinks.sh`, `refactor_utils/`, `notebooks/`,
  `core/training/bipo.py`, `core/training/reps.py`** and other source-only
  files. GRACE has its own source tree; only data is being migrated.

If unsure whether something is in scope, **stop and ask the user** rather
than copying.

---

## Recommended workflow

1. **Inventory first, copy second.** Walk the legacy tree and emit a plan
   to `~/grace_migration/MIGRATION_PLAN.txt`: every source path and every
   target path, grouped by section above. Show counts (`N concepts`,
   `N evaluations across M (layer,coef) cells`, etc.). Get the user's
   sign-off before any file is copied.
2. **Use hard links or `cp --reflink=auto` where possible** — these
   activation `.pt` files can be tens of GB each, and the staging dir lives
   on the same filesystem as the source.
3. **Don't rename in place.** Leave the legacy repo untouched; do all
   renaming during the staging copy.
4. **Verify path schemas, not contents.** For each copied file, confirm the
   target path matches one of the patterns in the "Target layout" section.
   Don't try to load `.pt` files unless investigating a schema mismatch — it
   is slow and unnecessary.
5. **Produce a final inventory** at `~/grace_migration/MIGRATION_REPORT.md`
   with: concept × model coverage matrix, anything that was found but
   skipped (and why), anything that was expected but missing.
6. The user will then `rsync -av --progress ~/grace_migration/
   user@grace-host:~/GRACE/`. Don't run rsync yourself unless asked.

## Sanity checks the user can run on the GRACE side post-migration

- Steering evals load correctly:
  ```
  python -c "from grace.eval.results import build_results_cache; \
             print(len(build_results_cache('google/gemma-2-2b-it', 'evil', 'pv')))"
  ```
- Activations load:
  ```
  python -c "from grace.activations.io import load_activations; \
             d = load_activations('google/gemma-2-2b-it', 'evil'); \
             print(d['concept'], len(d['activations_by_question']))"
  ```
- Vectors load (one per construction):
  ```
  python -c "import torch; print(sorted(torch.load( \
      'vectors/gemma-2-2b-it/evil/pv/response_avg.pt', weights_only=False).keys())[:5])"
  ```

If any of those fail with a path / schema error after migration, the source
of truth for the expected layout is `grace/paths.py` in the GRACE repo.
