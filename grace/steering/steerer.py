"""Inference-time activation steering via a forward hook.

Adds ``coeff * vector`` to the residual stream output of a chosen transformer
block. Used as a context manager:

    steerer = ActivationSteerer(model, vector, coeff=2.0, layer_idx=18, positions="response")
    with steerer:
        out = model.generate(...)
"""
from __future__ import annotations

from typing import Iterable, Sequence, Union

import torch

# Layer-list attribute paths in priority order. Add more as needed.
_LAYER_PATHS: Iterable[str] = (
    "model.layers",                 # Llama / Mistral / Gemma 1, 2 / Qwen
    "model.language_model.layers",  # Gemma 3 (multimodal)
    "language_model.model.layers",  # some VLMs
    "transformer.h",                # GPT-2 / NeoX
    "encoder.layer",                # BERT-style
)


def locate_layers(model: torch.nn.Module):
    """Return the indexable list of transformer blocks for `model`."""
    for path in _LAYER_PATHS:
        cur = model
        for part in path.split("."):
            if not hasattr(cur, part):
                cur = None
                break
            cur = getattr(cur, part)
        if cur is not None and hasattr(cur, "__getitem__"):
            return cur
    raise ValueError(
        f"Could not find a transformer-layer list on this model. "
        f"Add the right attribute path to grace/steering/steerer.py::_LAYER_PATHS."
    )


class ActivationSteerer:
    """Add ``coeff * vector`` to the output of one transformer block."""

    def __init__(
        self,
        model: torch.nn.Module,
        steering_vector: Union[torch.Tensor, Sequence[float]],
        *,
        coeff: float = 1.0,
        layer_idx: int,
        positions: str = "response",
    ):
        self.model = model
        self.coeff = float(coeff)
        self.layer_idx = int(layer_idx)
        if positions not in {"all", "prompt", "response"}:
            raise ValueError("positions must be one of {'all', 'prompt', 'response'}.")
        self.positions = positions

        p = next(model.parameters())
        self.vector = torch.as_tensor(steering_vector, dtype=p.dtype, device="cpu")
        if self.vector.ndim != 1:
            raise ValueError("steering vector must be 1-D")

        self._handle = None
        self._device_cache: dict[torch.device, torch.Tensor] = {}

    def _scaled(self, device: torch.device) -> torch.Tensor:
        if device not in self._device_cache:
            self._device_cache[device] = (self.coeff * self.vector).to(device)
        return self._device_cache[device]

    def _hook(self, module, ins, out):
        input_seq_len = ins[0].shape[1] if (isinstance(ins, (tuple, list)) and torch.is_tensor(ins[0])) else None

        def add(t: torch.Tensor) -> torch.Tensor:
            steer = self._scaled(t.device)
            seq_len = t.shape[1]
            is_prompt_encoding = input_seq_len is not None and input_seq_len == seq_len and seq_len > 1
            if self.positions == "all":
                return t + steer
            if self.positions == "prompt":
                if seq_len == 1:
                    return t
                return t + steer
            # "response": skip the prompt-encoding forward pass; only add to newly generated tokens.
            if is_prompt_encoding:
                return t
            t2 = t.clone()
            t2[:, -1, :] += steer
            return t2

        if torch.is_tensor(out):
            return add(out)
        if isinstance(out, (tuple, list)) and out and torch.is_tensor(out[0]):
            return (add(out[0]), *out[1:])
        return out

    def __enter__(self):
        layers = locate_layers(self.model)
        if not -len(layers) <= self.layer_idx < len(layers):
            raise IndexError(f"layer_idx={self.layer_idx} out of range for {len(layers)} layers.")
        self._handle = layers[self.layer_idx].register_forward_hook(self._hook)
        return self

    def __exit__(self, *exc):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
        self._device_cache.clear()
