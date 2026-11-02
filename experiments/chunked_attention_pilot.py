"""Compare blockwise TT attention with dense attention for one decode query."""

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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lengths", type=int, nargs="+", default=[512, 2048, 4096, 8192])
    parser.add_argument("--block-size", type=int, default=128)
    args = parser.parse_args()
    device = torch.device("cuda")
    torch.manual_seed(42)
    rows = []
    for length in args.lengths:
        query = torch.randn(1, 8, 1, 32, device=device, dtype=torch.float16)
        keys = torch.randn(1, 4, length, 32, device=device, dtype=torch.float16)
        values = torch.randn_like(keys)
        key_blocks = []
        value_blocks = []
        for start in range(0, length, args.block_size):
            for source, destination in ((keys, key_blocks), (values, value_blocks)):
                array = source[..., start : start + args.block_size, :].float().cpu().numpy()
                destination.append(Int8TTBlock(tt.vector(array, eps=0.1).cores, per_row=True))
        reconstructed_keys = torch.from_numpy(
            np.concatenate([block.full() for block in key_blocks], axis=-2)
        ).to(device, dtype=query.dtype)
        reconstructed_values = torch.from_numpy(
            np.concatenate([block.full() for block in value_blocks], axis=-2)
        ).to(device, dtype=query.dtype)
        dense_keys = reconstructed_keys.repeat_interleave(2, dim=1)
        dense_values = reconstructed_values.repeat_interleave(2, dim=1)
        torch.cuda.synchronize()
        dense_started = time.perf_counter()
        dense_scores = torch.matmul(query.float(), dense_keys.float().transpose(-2, -1)) / math.sqrt(32)
        reference = torch.matmul(dense_scores.softmax(dim=-1), dense_values.float()).to(query.dtype)
        torch.cuda.synchronize()
        dense_seconds = time.perf_counter() - dense_started
        del dense_keys, dense_values, reconstructed_keys, reconstructed_values, dense_scores, keys, values
        memory_before = torch.cuda.memory_allocated() / (1024**2)
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        chunked = chunked_tt_attention(query, key_blocks, value_blocks, key_value_heads=4)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        peak = torch.cuda.max_memory_allocated() / (1024**2)
        error = float((chunked.float() - reference.float()).abs().max())
        row = {
            "length": length,
            "block_size": args.block_size,
            "blocks": len(key_blocks),
            "max_absolute_attention_error": error,
            "chunked_seconds": elapsed,
            "dense_attention_seconds": dense_seconds,
            "chunked_peak_cuda_memory_mib": peak,
            "chunked_incremental_peak_mib": peak - memory_before,
        }
        print(json.dumps(row), flush=True)
        rows.append(row)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
