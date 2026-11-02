"""Single-query attention over CPU TT blocks with bounded GPU staging."""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

import numpy as np
import torch


def chunked_tt_attention(
    query: torch.Tensor,
    key_blocks: Iterable[Any],
    value_blocks: Iterable[Any],
    *,
    key_value_heads: int,
    staging_tokens: int = 1024,
) -> torch.Tensor:
    """Evaluate one decode query without materializing the full KV history.

    Query has shape [batch=1, query_heads, query_tokens=1, head_dim]. Blocks
    expose `.full()` and contain [1, kv_heads, block_tokens, head_dim]. This
    prototype is exact for its reconstructed K/V and has no causal masking for
    multi-token queries; it is intentionally limited to single-token decode.
    """
    if query.ndim != 4 or query.shape[0] != 1 or query.shape[2] != 1:
        raise ValueError("Expected one decode query [1, heads, 1, dim]")
    if query.shape[1] % key_value_heads:
        raise ValueError("Query heads must be divisible by KV heads")
    if staging_tokens <= 0:
        raise ValueError("staging_tokens must be positive")
    repeat = query.shape[1] // key_value_heads
    query_float = query.float()
    scale = 1 / math.sqrt(query.shape[-1])
    maximum = None
    denominator = None
    numerator = None
    count = 0
    key_group: list[np.ndarray] = []
    value_group: list[np.ndarray] = []
    group_tokens = 0

    def groups():
        nonlocal group_tokens
        for key_block, value_block in zip(key_blocks, value_blocks, strict=True):
            key_array = key_block.full()
            value_array = value_block.full()
            key_group.append(key_array)
            value_group.append(value_array)
            group_tokens += key_array.shape[-2]
            if group_tokens >= staging_tokens:
                yield np.concatenate(key_group, axis=-2), np.concatenate(value_group, axis=-2)
                key_group.clear()
                value_group.clear()
                group_tokens = 0
        if key_group:
            yield np.concatenate(key_group, axis=-2), np.concatenate(value_group, axis=-2)

    for key_array, value_array in groups():
        keys = torch.from_numpy(key_array).to(query.device, dtype=query.dtype)
        values = torch.from_numpy(value_array).to(query.device, dtype=query.dtype)
        if keys.shape != values.shape or keys.shape[1] != key_value_heads:
            raise ValueError("Incompatible K/V block shapes")
        keys = keys.repeat_interleave(repeat, dim=1).float()
        values = values.repeat_interleave(repeat, dim=1).float()
        scores = torch.matmul(query_float, keys.transpose(-2, -1)) * scale
        block_maximum = scores.amax(dim=-1, keepdim=True)
        if maximum is None:
            maximum = block_maximum
            weights = (scores - maximum).exp()
            denominator = weights.sum(dim=-1, keepdim=True)
            numerator = torch.matmul(weights, values)
        else:
            new_maximum = torch.maximum(maximum, block_maximum)
            previous_weight = (maximum - new_maximum).exp()
            weights = (scores - new_maximum).exp()
            denominator = denominator * previous_weight + weights.sum(
                dim=-1, keepdim=True
            )
            numerator = numerator * previous_weight + torch.matmul(weights, values)
            maximum = new_maximum
        count += 1
    if not count:
        raise ValueError("No KV blocks")
    return (numerator / denominator).to(query.dtype)
