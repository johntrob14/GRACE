"""Greedy generation for steered + prompting-baseline runs.

Two modes:

- Steered:        an `ActivationSteerer` is active during generation; the
                  caller passes a vector + (layer, coef).
- Prompting-only: a positive contrastive instruction is prepended as a system
                  message; no steering. This produces the prompting baseline
                  in paper §A.3.

Both modes share the same chat-templating and decoding, so the resulting
responses are directly comparable.
"""
from __future__ import annotations

from contextlib import nullcontext
from typing import Iterable

import torch
from tqdm import tqdm

from grace.steering import ActivationSteerer


def _format_chat(tokenizer, system: str | None, user: str) -> str:
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": user})
    try:
        return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    except Exception:
        prefix = f"<|system|>\n{system}\n" if system else ""
        return f"{prefix}<|user|>\n{user}\n<|assistant|>\n"


def generate_responses(
    model,
    tokenizer,
    questions: Iterable[str],
    *,
    steering_vector: torch.Tensor | None = None,
    layer_idx: int | None = None,
    coef: float = 0.0,
    prepend_system_prompt: str | None = None,
    max_new_tokens: int = 256,
    show_progress: bool = True,
) -> list[str]:
    """Greedy-generate one response per question.

    Args:
        steering_vector / layer_idx / coef:  if all three are provided and coef != 0,
            an `ActivationSteerer` wraps generation. Otherwise generation is plain.
        prepend_system_prompt: optional system message (paper's prompting baseline).
    """
    model.eval()
    device = next(model.parameters()).device

    use_steer = steering_vector is not None and layer_idx is not None and coef != 0
    if use_steer:
        steerer_ctx = ActivationSteerer(
            model, steering_vector, coeff=coef, layer_idx=layer_idx, positions="response",
        )
    else:
        steerer_ctx = nullcontext()

    out: list[str] = []
    with steerer_ctx:
        iterator = tqdm(list(questions), disable=not show_progress, desc="generating")
        for q in iterator:
            prompt = _format_chat(tokenizer, prepend_system_prompt, q)
            inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
            with torch.no_grad():
                gen = model.generate(
                    **inputs, do_sample=False, max_new_tokens=max_new_tokens,
                    pad_token_id=tokenizer.eos_token_id,
                )
            text = tokenizer.decode(gen[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            out.append(text)
    return out
