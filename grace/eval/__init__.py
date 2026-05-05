"""LLM-as-judge evaluation; concept score + coherence (paper §A.1, §A.2)."""

from grace.eval.judge import (
    BedrockJudge,
    LocalJudge,
    OpenAIJudge,
    clear_local_judge_cache,
    free_gpu_memory,
    score_responses,
)
from grace.eval.prompts import COHERENCE_PROMPT, CONCEPT_SCORE_TEMPLATE

__all__ = [
    "LocalJudge",
    "OpenAIJudge",
    "BedrockJudge",
    "score_responses",
    "clear_local_judge_cache",
    "free_gpu_memory",
    "COHERENCE_PROMPT",
    "CONCEPT_SCORE_TEMPLATE",
]
