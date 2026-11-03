"""Single-query attention by contracting resident TT cores without dense KV blocks."""

from __future__ import annotations

from collections.abc import Sequence
import math

import torch

from experiments.gpu_resident_tt import ResidentTTBlock


def _factors(block: ResidentTTBlock) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    cores = [q.float() * scale.float() for q, scale in zip(block.cores, block.scales, strict=True)]
    if len(cores) != 4 or block.shape[0] != 1:
        raise ValueError("Expected four-core TT block with batch size one")
    head = torch.einsum("iar,rhb->hb", cores[0], cores[1])
    return head, cores[2], cores[3].squeeze(-1)


def direct_tt_attention(
    query: torch.Tensor,
    key_blocks: Sequence[ResidentTTBlock],
    value_blocks: Sequence[ResidentTTBlock],
) -> torch.Tensor:
    """Compute online softmax over TT blocks, contracting K and V with query."""
    if len(key_blocks) != len(value_blocks) or not key_blocks:
        raise ValueError("Expected matched nonempty blocks")
    heads, dim = query.shape
    kv_heads = key_blocks[0].shape[1]
    if heads % kv_heads or key_blocks[0].shape[-1] != dim:
        raise ValueError("Incompatible GQA shapes")
    head_index = torch.arange(heads, device=query.device) // (heads // kv_heads)
    q = query.float()
    maximum = torch.full((heads,), -torch.inf, device=query.device)
    denominator = torch.zeros((heads,), device=query.device)
    numerator = torch.zeros((heads, dim), device=query.device)
    for key, value in zip(key_blocks, value_blocks, strict=True):
        ka, kb, kc = _factors(key)
        va, vb, vc = _factors(value)
        projected_query = kc @ q.T
        middle = torch.einsum("atb,bq->atq", kb, projected_query)
        scores = torch.einsum("qa,atq->qt", ka.index_select(0, head_index), middle)
        scores *= 1 / math.sqrt(dim)
        block_max = scores.max(dim=-1).values
        next_max = torch.maximum(maximum, block_max)
        old_factor = torch.exp(maximum - next_max)
        weights = torch.exp(scores - next_max[:, None])
        weighted_middle = torch.einsum("qt,atb->qab", weights, vb)
        part = torch.einsum("qa,qab,bd->qd", va.index_select(0, head_index), weighted_middle, vc)
        numerator = numerator * old_factor[:, None] + part
        denominator = denominator * old_factor + weights.sum(dim=-1)
        maximum = next_max
    return (numerator / denominator[:, None]).to(query.dtype)
