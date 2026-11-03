"""GPU-resident quantized TT cores with bounded staged decode attention."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch

from experiments.triton_attention_pilot import fused_attention
from experiments.tt_cache import Int8TTBlock


@dataclass
class ResidentTTBlock:
    cores: list[torch.Tensor]
    scales: list[torch.Tensor]
    shape: tuple[int, ...]

    @classmethod
    def from_cpu(cls, block: Int8TTBlock, device: torch.device) -> "ResidentTTBlock":
        return cls(
            [torch.from_numpy(core).to(device) for core in block.cores],
            [torch.from_numpy(np.asarray(scale)).to(device) for scale in block.scales],
            tuple(int(core.shape[1]) for core in block.cores),
        )

    def full(self, dtype: torch.dtype = torch.float16) -> torch.Tensor:
        current: torch.Tensor | None = None
        for quantized, scale in zip(self.cores, self.scales, strict=True):
            core = quantized.float() * scale.float()
            current = core if current is None else torch.einsum("...a,anb->...nb", current, core)
            if current.ndim > 3:
                current = current.reshape(-1, current.shape[-1])
        assert current is not None
        return current.reshape(self.shape).to(dtype)

    @property
    def payload_bytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in self.cores + self.scales)


@dataclass
class ResidentDenseBlock:
    states: torch.Tensor
    device: torch.device

    def full(self, dtype: torch.dtype = torch.float16) -> torch.Tensor:
        return self.states.to(device=self.device, dtype=dtype)


def resident_tt_attention(
    query: torch.Tensor,
    key_blocks: Sequence[ResidentTTBlock | ResidentDenseBlock],
    value_blocks: Sequence[ResidentTTBlock | ResidentDenseBlock],
    *,
    staging_tokens: int = 1024,
) -> torch.Tensor:
    """Reconstruct only one staging group and merge fused attention outputs."""
    if len(key_blocks) != len(value_blocks) or not key_blocks or staging_tokens < 1:
        raise ValueError("Expected matched nonempty blocks and positive staging_tokens")
    output: torch.Tensor | None = None
    total_lse: torch.Tensor | None = None
    pending_k: list[torch.Tensor] = []
    pending_v: list[torch.Tensor] = []
    pending_tokens = 0

    def consume() -> None:
        nonlocal output, total_lse, pending_tokens
        k = torch.cat(pending_k, dim=-2).squeeze(0)
        v = torch.cat(pending_v, dim=-2).squeeze(0)
        part, lse = fused_attention(query, k, v, return_lse=True)
        part = part.float()
        if output is None:
            output, total_lse = part, lse
        else:
            assert total_lse is not None
            combined = torch.logaddexp(total_lse, lse)
            output = output * (total_lse - combined).exp()[:, None] + part * (lse - combined).exp()[:, None]
            total_lse = combined
        pending_k.clear()
        pending_v.clear()
        pending_tokens = 0

    for key_block, value_block in zip(key_blocks, value_blocks, strict=True):
        key = key_block.full(query.dtype)
        value = value_block.full(query.dtype)
        if key.shape != value.shape or key.ndim != 4 or key.shape[0] != 1:
            raise ValueError("Invalid K/V block shape")
        pending_k.append(key)
        pending_v.append(value)
        pending_tokens += key.shape[-2]
        if pending_tokens >= staging_tokens:
            consume()
    if pending_k:
        consume()
    assert output is not None
    return output.to(query.dtype)
