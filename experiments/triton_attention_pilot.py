"""Triton fused online-softmax attention for one decode query.

This is a kernel benchmark only: K/V are already GPU-resident and contiguous.
It measures the compute path after TT block reconstruction, isolating kernel
cost from compression and host/device transfers.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import triton
import triton.language as tl


@triton.jit
def _decode_kernel(q, k, v, out, lse, length, dim, kv_heads, query_heads, stride_qh, stride_kl, stride_kh, stride_vl, stride_vh, stride_oh, scale: tl.constexpr, BLOCK_D: tl.constexpr, BLOCK_N: tl.constexpr):
    head = tl.program_id(0)
    offs_d = tl.arange(0, BLOCK_D)
    qv = tl.load(q + head * stride_qh + offs_d, mask=offs_d < dim, other=0.0).to(tl.float32)
    kv_head = head // (query_heads // kv_heads)
    maxv = tl.full((), -float("inf"), tl.float32)
    den = tl.zeros((), tl.float32)
    acc = tl.zeros((BLOCK_D,), tl.float32)
    for start in tl.range(0, length, BLOCK_N):
        offs_n = start + tl.arange(0, BLOCK_N)
        kptr = k + offs_n[:, None] * stride_kl + kv_head * stride_kh + offs_d[None, :]
        kval = tl.load(kptr, mask=(offs_n[:, None] < length) & (offs_d[None, :] < dim), other=0.0).to(tl.float32)
        score = tl.sum(kval * qv[None, :], axis=1) * scale
        block_max = tl.max(tl.where(offs_n < length, score, -float("inf")), axis=0)
        new_max = tl.maximum(maxv, block_max)
        old_factor = tl.exp(maxv - new_max)
        weight = tl.exp(score - new_max)
        weight = tl.where(offs_n < length, weight, 0.0)
        vptr = v + offs_n[:, None] * stride_vl + kv_head * stride_vh + offs_d[None, :]
        vval = tl.load(vptr, mask=(offs_n[:, None] < length) & (offs_d[None, :] < dim), other=0.0).to(tl.float32)
        acc = acc * old_factor + tl.sum(weight[:, None] * vval, axis=0)
        den = den * old_factor + tl.sum(weight, axis=0)
        maxv = new_max
    tl.store(out + head * stride_oh + offs_d, (acc / den).to(tl.float16), mask=offs_d < dim)
    tl.store(lse + head, maxv + tl.log(den))


def fused_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *, return_lse: bool = False) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """q [heads,d], k/v [kv_heads,length,d], GQA repeat by head modulo."""
    heads, dim = q.shape
    kv_heads, length, kdim = k.shape
    if kdim != dim or heads % kv_heads:
        raise ValueError("incompatible GQA shapes")
    out = torch.empty_like(q)
    lse = torch.empty(heads, device=q.device, dtype=torch.float32)
    grid = (heads,)
    _decode_kernel[grid](q, k, v, out, lse, length, dim, kv_heads, heads, q.stride(0), k.stride(1), k.stride(0), v.stride(1), v.stride(0), out.stride(0), 1 / math.sqrt(dim), BLOCK_D=triton.next_power_of_2(dim), BLOCK_N=128, num_warps=4)
    return (out, lse) if return_lse else out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lengths", type=int, nargs="+", default=[2048, 8192, 16384])
    args = parser.parse_args()
    torch.manual_seed(42)
    rows = []
    for length in args.lengths:
        q = torch.randn(16, 128, device="cuda", dtype=torch.float16)
        k = torch.randn(4, length, 128, device="cuda", dtype=torch.float16)
        v = torch.randn_like(k)
        reference = torch.nn.functional.scaled_dot_product_attention(q[None, :, None, :], k[None].repeat_interleave(4, 1), v[None].repeat_interleave(4, 1)).squeeze(0).squeeze(1)
        fused_attention(q, k, v)
        torch.cuda.synchronize()
        started = time.perf_counter()
        result = fused_attention(q, k, v)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        rows.append({"length": length, "fused_seconds": elapsed, "max_absolute_error": float((result.float() - reference.float()).abs().max())})
        print(json.dumps(rows[-1]), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
