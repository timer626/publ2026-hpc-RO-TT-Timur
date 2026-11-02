"""Bounded GPU staging of CPU TT blocks with fused attention per group."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch

from experiments.triton_attention_pilot import fused_attention


def streaming_tt_attention(
    query: torch.Tensor,
    key_blocks: Sequence[Any],
    value_blocks: Sequence[Any],
    *,
    staging_tokens: int = 1024,
) -> torch.Tensor:
    """Return one-query GQA attention without full-history GPU materialization.

    Query is [query_heads, head_dim]. CPU block `.full()` arrays have shape
    [1, kv_heads, block_tokens, head_dim]. Each group is staged on GPU, reduced
    by Triton, and merged with a stable log-sum-exp accumulator.
    """
    if len(key_blocks) != len(value_blocks) or not key_blocks or staging_tokens < 1:
        raise ValueError("Expected matched nonempty blocks and positive staging_tokens")
    pending_k: list[np.ndarray] = []
    pending_v: list[np.ndarray] = []
    pending_tokens = 0
    result = None
    total_lse = None

    def consume() -> None:
        nonlocal result, total_lse, pending_tokens
        k = torch.from_numpy(np.concatenate(pending_k, axis=-2).squeeze(0)).to(query.device, query.dtype)
        v = torch.from_numpy(np.concatenate(pending_v, axis=-2).squeeze(0)).to(query.device, query.dtype)
        part, lse = fused_attention(query, k, v, return_lse=True)
        part = part.float()
        if result is None:
            result, total_lse = part, lse
        else:
            combined = torch.logaddexp(total_lse, lse)
            result = result * (total_lse - combined).exp()[:, None] + part * (lse - combined).exp()[:, None]
            total_lse = combined
        pending_k.clear()
        pending_v.clear()
        pending_tokens = 0

    for key_block, value_block in zip(key_blocks, value_blocks, strict=True):
        key = key_block.full()
        value = value_block.full()
        if key.shape != value.shape or key.ndim != 4 or key.shape[0] != 1:
            raise ValueError("Invalid K/V block shape")
        pending_k.append(key)
        pending_v.append(value)
        pending_tokens += key.shape[-2]
        if pending_tokens >= staging_tokens:
            consume()
    if pending_k:
        consume()
    return result.to(query.dtype)
