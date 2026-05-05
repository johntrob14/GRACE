# Adding new concepts

GRACE works on any concept defined as a one-paragraph human-written
description. The pipeline that turns that description into a usable steering
vector + diagnostic suite has four steps.

## 1. Generate the concept JSON via GPT-5

```bash
uv run python scripts/01_generate_concept_data.py \
  --concept my_new_concept \
  --description "Frames every response in terms of marine biology, even when..."
```

This calls the OpenAI Responses API with `gpt-5` and `reasoning_effort=low`
(paper §A) and writes:

```
concepts/gpt-5/extract/my_new_concept.json   # 100 extraction questions
concepts/gpt-5/eval/my_new_concept.json      # 100 held-out eval questions
```

Both JSONs share the same 5 instruction pairs and the same per-concept
rubric. Requires `OPENAI_API_KEY` in `.env`.

## 2. Extract activations

```bash
uv run python scripts/02_extract_activations.py \
  --model google/gemma-2-2b-it --concept my_new_concept
```

Caches the activations dict to
`activations/{model}/my_new_concept/activations_by_question.pt`.

## 3. Compute statistics

```bash
uv run python scripts/03_compute_statistics.py \
  --model google/gemma-2-2b-it --concept my_new_concept
```

Writes `results/statistics/{model}/my_new_concept_statistics.pt`. This is
what every diagnostic + notebook reads.

## 4. Train vectors (optional, for steering)

```bash
for m in pv unit_mean cluster; do
  uv run python scripts/04_train_vectors.py \
    --model google/gemma-2-2b-it --concept my_new_concept --method "$m"
done
```

## 5. Run the GRACE diagnostic

```bash
uv run python scripts/08_run_diagnostics.py \
  --model google/gemma-2-2b-it --concepts my_new_concept
```

The recommendation is printed and saved to
`results/diagnostics/{model}/my_new_concept.json`. From there you can use
the recommended search-space / construction in your TPE call:

```bash
uv run python scripts/06_optuna_search.py \
  --config configs/search/tpe_top15_pl.yaml \
  --concept my_new_concept
```

## How prompts are formatted

The activation-extraction pipeline applies the model's chat template with
the contrastive instruction as a `system` message and the extraction
question as a `user` message:

```
<|system|>{instruction.pos or .neg}<|user|>{question}<|assistant|>
```

Greedy decoding produces a response, and we record three activation
variants at every layer (prompt_avg, prompt_last, response_avg). See
`grace/activations/extract.py` for the exact code path.
