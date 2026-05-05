"""LLM-as-judge backends for concept-score and coherence (paper §A.1, §A.2).

Three backends. The local Gemma judge is the default and the only one used by
the smoke test; the other two are optional and gated on env vars.

- LocalJudge:    `google/gemma-3-12b-it` served locally via HF Transformers.
                 The paper's primary judge; produces parseable structured output
                 under the integer-only schema.
- OpenAIJudge:   `gpt-4.1-mini` via the OpenAI Chat Completions API.
                 Uses logprobs over single-digit tokens for a calibrated
                 expected-score estimator (§A.2).
- BedrockJudge:  `amazon.nova-lite-v1:0` via AWS Bedrock. Requires the optional
                 `bedrock` extra (`uv sync --extra bedrock`).

All three implement the same `score(question, answer, prompt_template) -> float`
interface so the eval runner can swap backends without knowing which one it has.
"""
from __future__ import annotations

import gc
import math
import os
import re
from dataclasses import dataclass

import torch

# ---------- Local Gemma judge -----------------------------------------------

_local_judge_cache: dict[str, tuple] = {}


def _load_local_judge(model_name: str = "google/gemma-3-12b-it"):
    if model_name in _local_judge_cache:
        return _local_judge_cache[model_name]
    from grace.steering import load_model
    model, tokenizer = load_model(model_name)
    model.eval()
    _local_judge_cache[model_name] = (model, tokenizer)
    return model, tokenizer


def clear_local_judge_cache() -> None:
    """Drop all cached local judge models so the GPU memory they hold is reclaimed.

    Eval scripts run generation and judging as separate phases (steered model
    freed before the judge loads, judge freed before the next phase). Call this
    at the end of the judging phase to release the judge's GPU residency.
    """
    if not _local_judge_cache:
        return
    for key in list(_local_judge_cache.keys()):
        model, tokenizer = _local_judge_cache[key]
        if hasattr(model, "cpu"):
            try:
                model.cpu()
            except Exception:
                pass
        del model, tokenizer
    _local_judge_cache.clear()
    free_gpu_memory()


def free_gpu_memory() -> None:
    """Run gc + cuda empty_cache. Caller is responsible for `del`-ing references first."""
    gc.collect()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def _parse_int_in_range(text: str, lo: int = 0, hi: int = 100) -> float | None:
    if not text:
        return None
    if "REFUSAL" in text.upper():
        return None
    m = re.search(r"\b(\d{1,3})\b", text)
    if not m:
        return None
    val = int(m.group(1))
    return float(val) if lo <= val <= hi else None


@dataclass
class LocalJudge:
    """Gemma-3-12b-it judge served locally. Uses greedy decoding + integer parsing."""

    model_name: str = "google/gemma-3-12b-it"
    max_new_tokens: int = 8

    def __post_init__(self):
        self._model, self._tokenizer = _load_local_judge(self.model_name)

    def _generate(self, prompt: str) -> str:
        device = next(self._model.parameters()).device
        msgs = [{"role": "user", "content": prompt}]
        text = self._tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = self._tokenizer(text, return_tensors="pt", add_special_tokens=False).to(device)
        with torch.no_grad():
            out = self._model.generate(
                **inputs, do_sample=False, max_new_tokens=self.max_new_tokens,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        return self._tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    def score(self, prompt: str) -> float | None:
        return _parse_int_in_range(self._generate(prompt))


# ---------- OpenAI judge -----------------------------------------------------

@dataclass
class OpenAIJudge:
    """GPT-4.1-mini via OpenAI API, with logprobs-based expected-score aggregation."""

    model_name: str = "gpt-4.1-mini"

    def __post_init__(self):
        from openai import AsyncOpenAI
        from grace.config import require_env
        self._client = AsyncOpenAI(api_key=require_env("OPENAI_API_KEY"))

    async def _logprobs(self, prompt: str) -> dict[str, float]:
        completion = await self._client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1,
            temperature=0,
            logprobs=True,
            top_logprobs=20,
            seed=0,
        )
        try:
            top = completion.choices[0].logprobs.content[0].top_logprobs
            return {entry.token: float(entry.logprob) for entry in top}
        except (AttributeError, IndexError):
            return {}

    @staticmethod
    def _expected_value(probs: dict[str, float]) -> float | None:
        """Convert logprobs over digit tokens into a probability-weighted score."""
        total, expected = 0.0, 0.0
        for tok, lp in probs.items():
            tok = tok.strip()
            if not tok.isdigit():
                continue
            v = int(tok)
            if 0 <= v <= 100:
                p = math.exp(lp)
                total += p
                expected += p * v
        if total == 0:
            return None
        return expected / total

    async def score(self, prompt: str) -> float | None:
        return self._expected_value(await self._logprobs(prompt))


# ---------- AWS Bedrock Nova judge ------------------------------------------

@dataclass
class BedrockJudge:
    """Amazon Nova-Lite via AWS Bedrock. Requires the optional `bedrock` extra."""

    model_id: str = "amazon.nova-lite-v1:0"
    region: str = "us-east-1"

    def __post_init__(self):
        try:
            import boto3  # type: ignore
        except ImportError as e:
            raise ImportError(
                "BedrockJudge requires the optional `bedrock` extra. "
                "Install with `uv sync --extra bedrock`."
            ) from e
        from grace.config import require_env
        require_env("AWS_ACCESS_KEY_ID")
        require_env("AWS_SECRET_ACCESS_KEY")
        self._client = boto3.client("bedrock-runtime", region_name=os.environ.get("AWS_REGION", self.region))

    def score(self, prompt: str) -> float | None:
        import json
        response = self._client.invoke_model(
            modelId=self.model_id,
            body=json.dumps({
                "messages": [{"role": "user", "content": [{"text": prompt}]}],
                "inferenceConfig": {"maxTokens": 8, "temperature": 0},
            }),
        )
        body = json.loads(response["body"].read())
        text = body["output"]["message"]["content"][0]["text"]
        return _parse_int_in_range(text)


# ---------- Convenience score-many entry point ------------------------------

def score_responses(
    judge: LocalJudge | OpenAIJudge | BedrockJudge,
    prompts: list[str],
) -> list[float | None]:
    """Score a batch of prompts. Sync for Local/Bedrock; runs the async loop for OpenAI."""
    if isinstance(judge, OpenAIJudge):
        import asyncio
        return asyncio.run(_score_async(judge, prompts))
    return [judge.score(p) for p in prompts]


async def _score_async(judge: OpenAIJudge, prompts: list[str]) -> list[float | None]:
    import asyncio
    return await asyncio.gather(*(judge.score(p) for p in prompts))
