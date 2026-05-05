"""HF model + tokenizer loader."""
from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(
    model_name: str,
    dtype: torch.dtype = torch.bfloat16,
    device_map: str | dict = "auto",
):
    """Load `(model, tokenizer)` from HF Hub.

    Args:
        model_name: HF Hub id (e.g. ``google/gemma-2-2b-it``).
        dtype: Torch dtype. ``bfloat16`` is what the paper used everywhere.
        device_map: ``"auto"`` shards the model across visible GPUs.
    """
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype, device_map=device_map)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return model, tokenizer
