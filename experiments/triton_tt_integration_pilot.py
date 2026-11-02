"""Integrate TT block reconstruction with the Triton GPU attention kernel."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import tt

from experiments.chunked_attention import chunked_tt_attention
from experiments.tt_cache import Int8TTBlock
from experiments.triton_attention_pilot import fused_attention


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lengths", type=int, nargs="+", default=[2048, 8192, 16384])
    parser.add_argument("--block-size", type=int, default=256)
    args = parser.parse_args()
    torch.manual_seed(42)
    rows = []
    for length in args.lengths:
        q = torch.randn(16, 128, device="cuda", dtype=torch.float16)
        keys = torch.randn(4, length, 128, device="cuda", dtype=torch.float16)
        values = torch.randn_like(keys)
        key_blocks, value_blocks = [], []
        started = time.perf_counter()
        for start in range(0, length, args.block_size):
            for source, destination in ((keys, key_blocks), (values, value_blocks)):
                array = source[..., start:start + args.block_size, :].float().cpu().numpy()[None]
                destination.append(Int8TTBlock(tt.vector(array, eps=0.1).cores, per_row=True))
        reconstructed_k = torch.from_numpy(np.concatenate([b.full() for b in key_blocks], axis=-2)).to("cuda", dtype=q.dtype).squeeze(0)
        reconstructed_v = torch.from_numpy(np.concatenate([b.full() for b in value_blocks], axis=-2)).to("cuda", dtype=q.dtype).squeeze(0)
        torch.cuda.synchronize()
        reconstruction_seconds = time.perf_counter() - started
        dense_k = reconstructed_k.repeat_interleave(4, dim=0)
        dense_v = reconstructed_v.repeat_interleave(4, dim=0)
        fused_attention(q, reconstructed_k, reconstructed_v)
        torch.nn.functional.scaled_dot_product_attention(q[None, :, None, :], dense_k[None], dense_v[None])
        chunked_tt_attention(q[None, :, None, :], key_blocks, value_blocks, key_value_heads=4)
        torch.cuda.synchronize()
        started = time.perf_counter()
        dense = torch.nn.functional.scaled_dot_product_attention(q[None, :, None, :], dense_k[None], dense_v[None]).squeeze(0).squeeze(1)
        torch.cuda.synchronize()
        dense_seconds = time.perf_counter() - started
        started = time.perf_counter()
        fused = fused_attention(q, reconstructed_k, reconstructed_v)
        torch.cuda.synchronize()
        fused_seconds = time.perf_counter() - started
        started = time.perf_counter()
        chunked = chunked_tt_attention(q[None, :, None, :], key_blocks, value_blocks, key_value_heads=4).squeeze(0).squeeze(1)
        torch.cuda.synchronize()
        chunked_seconds = time.perf_counter() - started
        row = {
            "length": length,
            "block_size": args.block_size,
            "tt_reconstruction_seconds": reconstruction_seconds,
            "dense_attention_seconds": dense_seconds,
            "triton_attention_seconds": fused_seconds,
            "python_chunked_seconds": chunked_seconds,
            "triton_max_error": float((fused.float() - dense.float()).abs().max()),
            "python_chunked_max_error": float((chunked.float() - dense.float()).abs().max()),
            "tt_payload_bytes": sum(b.storage_bytes for b in key_blocks + value_blocks),
            "dense_equivalent_bytes": int(keys.numel() * 2 + values.numel() * 2),
        }
        rows.append(row)
        print(json.dumps(row), flush=True)
        del keys, values, reconstructed_k, reconstructed_v, dense_k, dense_v
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
