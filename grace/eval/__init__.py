"""LLM-as-judge evaluation; concept score + coherence (paper §A.1, §A.2)."""

from grace.eval.judge import LocalJudge, OpenAIJudge, BedrockJudge, score_responses
from grace.eval.prompts import COHERENCE_PROMPT, CONCEPT_SCORE_TEMPLATE

__all__ = [
    "LocalJudge",
    "OpenAIJudge",
    "BedrockJudge",
    "score_responses",
    "COHERENCE_PROMPT",
    "CONCEPT_SCORE_TEMPLATE",
]
