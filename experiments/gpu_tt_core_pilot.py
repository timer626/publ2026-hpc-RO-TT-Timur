"""Compare CPU TT reconstruction with GPU-resident quantized-core reconstruction."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import tt

from experiments.tt_cache import Int8TTBlock


def gpu_full(block: Int8TTBlock) -> torch.Tensor:
    """Reconstruct one quantized TT tensor without returning to CPU."""
    current: torch.Tensor | None = None
    for core, scale in zip(block.cores, block.scales):
        q = torch.from_numpy(core).to("cuda", dtype=torch.float32)
        s = torch.from_numpy(np.asarray(scale)).to("cuda", dtype=torch.float32)
        dequant = q * s
        if current is None:
            current = dequant
        else:
            current = torch.einsum("...a,anb->...nb", current, dequant)
        # Merge all completed physical dimensions while retaining TT rank.
        if current.ndim > 3:
            current = current.reshape(-1, current.shape[-1])
    assert current is not None
    shape = tuple(int(core.shape[1]) for core in block.cores)
    return current.reshape(shape)


def make_block(shape: tuple[int, int, int, int], eps: float) -> Int8TTBlock:
    array = np.random.default_rng(42).standard_normal(shape, dtype=np.float32)
    return Int8TTBlock(tt.vector(array, eps=eps).cores, per_row=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lengths", type=int, nargs="+", default=[2048, 8192])
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--eps", type=float, default=0.1)
    args = parser.parse_args()
    rows = []
    for length in args.lengths:
        blocks = [make_block((1, 8, min(args.block_size, length - start), 128), args.eps)
                  for start in range(0, length, args.block_size)]
        # Warm up CUDA allocations and kernels.
        gpu_full(blocks[0]); torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        cpu_started = time.perf_counter()
        cpu_payload = [torch.from_numpy(b.full()).to("cuda", dtype=torch.float16) for b in blocks]
        torch.cuda.synchronize()
        cpu_seconds = time.perf_counter() - cpu_started
        cpu_peak = torch.cuda.max_memory_allocated() / 1024**2
        del cpu_payload
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        gpu_started = time.perf_counter()
        gpu_payload = [gpu_full(b).to(dtype=torch.float16) for b in blocks]
        torch.cuda.synchronize()
        gpu_seconds = time.perf_counter() - gpu_started
        gpu_peak = torch.cuda.max_memory_allocated() / 1024**2
        reference = blocks[0].full()
        error = float((gpu_payload[0].float().cpu().numpy() - reference).max())
        row = {"length": length, "block_size": args.block_size, "blocks": len(blocks),
               "eps": args.eps, "cpu_reconstruct_transfer_seconds": cpu_seconds,
               "gpu_core_reconstruct_seconds": gpu_seconds,
               "speedup": cpu_seconds / gpu_seconds, "gpu_peak_mib": gpu_peak,
               "max_abs_error_first_block": abs(error),
               "max_tt_rank": max(int(b.r.max()) for b in blocks),
               "tt_core_bytes": sum(b.storage_bytes for b in blocks)}
        rows.append(row); print(json.dumps(row), flush=True)
        del gpu_payload
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
