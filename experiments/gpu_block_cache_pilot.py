"""Measure GPU caching of reconstructed TT blocks during sequential decode scans."""

from __future__ import annotations

import os

for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
import json
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import tt

from experiments.tt_cache import Int8TTBlock


def build_blocks(length: int, heads: int, dim: int, block_size: int) -> list[tuple[Int8TTBlock, Int8TTBlock]]:
    generator = np.random.default_rng(42)
    pairs = []
    for start in range(0, length, block_size):
        size = min(block_size, length - start)
        key = generator.standard_normal((1, heads, size, dim), dtype=np.float32)
        value = generator.standard_normal((1, heads, size, dim), dtype=np.float32)
        pairs.append((
            Int8TTBlock(tt.vector(key, eps=0.1).cores, per_row=True),
            Int8TTBlock(tt.vector(value, eps=0.1).cores, per_row=True),
        ))
    return pairs


def run_case(blocks, *, scans: int, budget_mib: int, policy: str) -> dict:
    block_bytes = (blocks[0][0].full().nbytes + blocks[0][1].full().nbytes) // 2
    capacity = min(len(blocks), budget_mib * 1024**2 // block_bytes)
    cache: OrderedDict[int, tuple[torch.Tensor, torch.Tensor]] = OrderedDict()
    recent = set(range(len(blocks) - capacity, len(blocks))) if capacity else set()
    hits = misses = 0
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(scans):
        for index, (key_block, value_block) in enumerate(blocks):
            if index in cache:
                hits += 1
                if policy == "lru":
                    cache.move_to_end(index)
                continue
            misses += 1
            key = torch.from_numpy(key_block.full()).to("cuda", dtype=torch.float16)
            value = torch.from_numpy(value_block.full()).to("cuda", dtype=torch.float16)
            if policy == "lru" and capacity:
                cache[index] = (key, value)
                if len(cache) > capacity:
                    cache.popitem(last=False)
            elif policy == "recent" and index in recent:
                cache[index] = (key, value)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    return {
        "policy": policy,
        "budget_mib": budget_mib,
        "capacity_blocks": capacity,
        "block_bytes_gpu_float16": block_bytes,
        "scans": scans,
        "accesses": scans * len(blocks),
        "hits": hits,
        "misses": misses,
        "hit_rate": hits / (hits + misses),
        "elapsed_seconds": elapsed,
        "peak_cuda_memory_mib": torch.cuda.max_memory_allocated() / 1024**2,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--length", type=int, default=8192)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--scans", type=int, default=16)
    parser.add_argument("--budgets-mib", type=int, nargs="+", default=[0, 8, 16, 32])
    args = parser.parse_args()
    started = time.perf_counter()
    blocks = build_blocks(args.length, args.heads, args.dim, args.block_size)
    build_seconds = time.perf_counter() - started
    rows = []
    for policy, budgets in (("none", [0]), ("lru", args.budgets_mib[1:]), ("recent", args.budgets_mib[1:])):
        for budget in budgets:
            row = run_case(blocks, scans=args.scans, budget_mib=budget, policy=policy)
            row.update({"length": args.length, "heads": args.heads, "dim": args.dim, "block_size": args.block_size, "tt_build_seconds": build_seconds})
            rows.append(row)
            print(json.dumps(row), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
