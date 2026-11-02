"""Experimental Qwen3 single-token decode over TT blocks."""

from __future__ import annotations

from types import MethodType
from typing import Any

import numpy as np
import torch
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3Attention,
    apply_rotary_pos_emb,
)

from experiments.chunked_attention import chunked_tt_attention
from experiments.tt_cache import TTCompressedCache


class DenseTailBlock:
    def __init__(self, states: torch.Tensor) -> None:
        self.states = states

    def full(self) -> np.ndarray:
        return self.states.float().contiguous().numpy()


def install_chunked_tt_decode(model: Any) -> None:
    """Patch only single-token TT decode; retain normal Qwen3 prefill."""
    for module in model.modules():
        if not isinstance(module, Qwen3Attention):
            continue
        original_forward = module.forward

        def forward(
            self: Qwen3Attention,
            hidden_states: torch.Tensor,
            position_embeddings: tuple[torch.Tensor, torch.Tensor],
            attention_mask: torch.Tensor | None,
            past_key_values: TTCompressedCache | None = None,
            *,
            _original=original_forward,
            **kwargs: Any,
        ) -> tuple[torch.Tensor, None]:
            if not isinstance(past_key_values, TTCompressedCache) or hidden_states.shape[1] != 1:
                return _original(
                    hidden_states,
                    position_embeddings,
                    attention_mask,
                    past_key_values=past_key_values,
                    **kwargs,
                )
            input_shape = hidden_states.shape[:-1]
            hidden_shape = (*input_shape, -1, self.head_dim)
            query = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
            keys = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
            values = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            cos, sin = position_embeddings
            query, keys = apply_rotary_pos_emb(query, keys, cos, sin)
            layer = past_key_values.layers[self.layer_idx]
            layer.update(keys, values, materialize=False)
            key_blocks = list(layer.key_blocks)
            value_blocks = list(layer.value_blocks)
            if layer.tail_keys is not None:
                key_blocks.append(DenseTailBlock(layer.tail_keys))
                value_blocks.append(DenseTailBlock(layer.tail_values))
            result = chunked_tt_attention(
                query,
                key_blocks,
                value_blocks,
                key_value_heads=keys.shape[1],
            )
            return self.o_proj(result.reshape(*input_shape, -1).contiguous()), None

        module.forward = MethodType(forward, module)
