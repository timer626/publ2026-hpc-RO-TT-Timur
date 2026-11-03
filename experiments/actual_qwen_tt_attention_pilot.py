"""Compare TT-aware attention paths on real Qwen3-8B prefill cache blocks."""

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
from transformers import AutoModelForCausalLM, AutoTokenizer

from experiments.direct_tt_attention import direct_tt_attention
from experiments.full_model_resident_pilot import prompt_near_length
from experiments.gpu_resident_tt import ResidentTTBlock, resident_tt_attention
from experiments.tt_cache import TTCompressedCache


def timed(fn, scans: int) -> tuple[torch.Tensor, float]:
    fn()
    torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(scans):
        result = fn()
    torch.cuda.synchronize()
    return result, (time.perf_counter() - started) / scans


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--targets", type=int, nargs="+", default=[2048, 8192])
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--scans", type=int, default=8)
    args = parser.parse_args()
    row = json.loads(args.source.read_text().splitlines()[0])
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, local_files_only=True, dtype="auto").cuda().eval()
    rows = []
    for target in args.targets:
        prompt, prompt_tokens = prompt_near_length(tokenizer, row, target)
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        cache = TTCompressedCache(model.config, 0.1, 8, "int8_row", 128)
        with torch.inference_mode():
            output = model(**inputs, past_key_values=cache, use_cache=True, logits_to_keep=1)
        del output
        layer = cache.layers[args.layer]
        keys = layer.key_blocks
        values = layer.value_blocks
        resident_k = [ResidentTTBlock.from_cpu(block, inputs.input_ids.device) for block in keys]
        resident_v = [ResidentTTBlock.from_cpu(block, inputs.input_ids.device) for block in values]
        torch.manual_seed(42)
        query = torch.randn(model.config.num_attention_heads, model.config.head_dim, device="cuda", dtype=torch.float16)
        staged, staged_seconds = timed(lambda: resident_tt_attention(query, resident_k, resident_v), args.scans)
        direct, direct_seconds = timed(lambda: direct_tt_attention(query, resident_k, resident_v), args.scans)
        dense_k = torch.from_numpy(np.concatenate([block.full() for block in keys], axis=-2)).to("cuda", query.dtype).squeeze(0)
        dense_v = torch.from_numpy(np.concatenate([block.full() for block in values], axis=-2)).to("cuda", query.dtype).squeeze(0)
        repeat = query.shape[0] // dense_k.shape[0]
        reference = torch.nn.functional.scaled_dot_product_attention(
            query[None, :, None, :],
            dense_k.repeat_interleave(repeat, dim=0)[None],
            dense_v.repeat_interleave(repeat, dim=0)[None],
        ).squeeze(0).squeeze(1)
        ranks = [int(block.r.max()) for block in keys + values]
        result = {
            "prompt_target": target, "prompt_tokens": prompt_tokens,
            "layer": args.layer, "blocks": len(keys), "scans": args.scans,
            "min_tt_rank": min(ranks), "median_tt_rank": float(np.median(ranks)),
            "max_tt_rank": max(ranks),
            "staged_seconds_per_scan": staged_seconds,
            "direct_seconds_per_scan": direct_seconds,
            "direct_vs_staged_speedup": staged_seconds / direct_seconds,
            "staged_max_error": float((staged.float() - reference.float()).abs().max()),
            "direct_max_error": float((direct.float() - reference.float()).abs().max()),
            "stored_core_bytes": sum(block.storage_bytes for block in keys + values),
        }
        rows.append(result)
        print(json.dumps(result), flush=True)
        del cache, resident_k, resident_v, dense_k, dense_v, reference, inputs
        torch.cuda.empty_cache()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
