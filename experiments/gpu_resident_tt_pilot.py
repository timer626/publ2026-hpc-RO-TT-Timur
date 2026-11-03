"""Measure GPU-resident TT reconstruction integrated with staged Triton attention."""

from __future__ import annotations

import os

for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import tt

from experiments.gpu_resident_tt import ResidentTTBlock, resident_tt_attention
from experiments.direct_tt_attention import direct_tt_attention
from experiments.triton_streaming_tt import streaming_tt_attention
from experiments.tt_cache import Int8TTBlock


def make_blocks(length: int, block_size: int, heads: int, dim: int) -> tuple[list[Int8TTBlock], list[Int8TTBlock]]:
    rng = np.random.default_rng(42)
    keys, values = [], []
    for start in range(0, length, block_size):
        size = min(block_size, length - start)
        for destination in (keys, values):
            array = rng.standard_normal((1, heads, size, dim), dtype=np.float32)
            destination.append(Int8TTBlock(tt.vector(array, eps=0.1).cores, per_row=True))
    return keys, values


def timed_attention(fn, scans: int) -> tuple[torch.Tensor, float]:
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(scans):
        output = fn()
    torch.cuda.synchronize()
    return output, (time.perf_counter() - start) / scans


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lengths", type=int, nargs="+", default=[2048, 8192])
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--staging-tokens", type=int, default=1024)
    parser.add_argument("--scans", type=int, default=16)
    parser.add_argument("--direct-scans", type=int, default=3)
    args = parser.parse_args()
    torch.manual_seed(42)
    rows = []
    for length in args.lengths:
        keys, values = make_blocks(length, args.block_size, 8, 128)
        query = torch.randn(32, 128, device="cuda", dtype=torch.float16)
        torch.cuda.synchronize()
        base_alloc = torch.cuda.memory_allocated()
        transfer_start = time.perf_counter()
        rk = [ResidentTTBlock.from_cpu(block, query.device) for block in keys]
        rv = [ResidentTTBlock.from_cpu(block, query.device) for block in values]
        torch.cuda.synchronize()
        transfer_seconds = time.perf_counter() - transfer_start
        resident_alloc = torch.cuda.memory_allocated() - base_alloc
        gpu_fn = lambda: resident_tt_attention(query, rk, rv, staging_tokens=args.staging_tokens)
        cpu_fn = lambda: streaming_tt_attention(query, keys, values, staging_tokens=args.staging_tokens)
        gpu_fn()
        cpu_fn()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        gpu_output, gpu_seconds = timed_attention(gpu_fn, args.scans)
        gpu_incremental_peak = torch.cuda.max_memory_allocated() - base_alloc
        cpu_output, cpu_seconds = timed_attention(cpu_fn, args.scans)
        dense_k = torch.from_numpy(np.concatenate([block.full() for block in keys], axis=-2)).to(query.device, query.dtype).squeeze(0)
        dense_v = torch.from_numpy(np.concatenate([block.full() for block in values], axis=-2)).to(query.device, query.dtype).squeeze(0)
        reference_output = torch.nn.functional.scaled_dot_product_attention(
            query[None, :, None, :],
            dense_k.repeat_interleave(query.shape[0] // dense_k.shape[0], dim=0)[None],
            dense_v.repeat_interleave(query.shape[0] // dense_v.shape[0], dim=0)[None],
        ).squeeze(0).squeeze(1)
        direct_fn = lambda: direct_tt_attention(query, rk, rv)
        direct_fn()
        direct_output, direct_seconds = timed_attention(direct_fn, args.direct_scans)
        # Check reconstruction and attention agreement separately.
        reference = keys[0].full()
        reconstructed = rk[0].full(query.dtype).float().cpu().numpy()
        row = {
            "length": length, "block_size": args.block_size,
            "staging_tokens": args.staging_tokens, "scans": args.scans,
            "core_transfer_seconds": transfer_seconds,
            "resident_core_mib": resident_alloc / 1024**2,
            "gpu_attention_seconds_per_scan": gpu_seconds,
            "cpu_streaming_seconds_per_scan": cpu_seconds,
            "decode_speedup": cpu_seconds / gpu_seconds,
            "gpu_incremental_peak_mib": gpu_incremental_peak / 1024**2,
            "max_attention_difference": float((gpu_output.float() - cpu_output.float()).abs().max()),
            "max_attention_reference_error": float((gpu_output.float() - reference_output.float()).abs().max()),
            "direct_attention_seconds_per_scan": direct_seconds,
            "direct_vs_staged_speedup": gpu_seconds / direct_seconds,
            "max_direct_reference_error": float((direct_output.float() - reference_output.float()).abs().max()),
            "max_reconstruction_difference_first_block": float(np.abs(reconstructed - reference).max()),
            "max_tt_rank": max(int(block.r.max()) for block in keys + values),
        }
        rows.append(row)
        print(json.dumps(row), flush=True)
        del rk, rv, keys, values, query
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
