"""Separate prefill and decode timings for a small synthetic Qwen3 cache pilot."""

from __future__ import annotations

import os

for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import DynamicCache, Qwen3Config, Qwen3ForCausalLM

from experiments.chunked_qwen3 import install_chunked_tt_decode
from experiments.tt_cache import TTCompressedCache


def timed_forward(model, tokens, cache):
    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.inference_mode():
        output = model(input_ids=tokens, past_key_values=cache, use_cache=True)
    torch.cuda.synchronize()
    return output, time.perf_counter() - start


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt-tokens", type=int, nargs="+", default=[2048, 8192])
    parser.add_argument("--decode-tokens", type=int, default=64)
    args = parser.parse_args()
    torch.manual_seed(42)
    config = Qwen3Config(
        vocab_size=256,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=4,
        num_attention_heads=8,
        num_key_value_heads=4,
        head_dim=32,
        max_position_embeddings=max(args.prompt_tokens) + args.decode_tokens + 8,
    )
    rows = []
    for length in args.prompt_tokens:
        tokens = torch.randint(0, config.vocab_size, (1, length), device="cuda")
        for kind in ("dense", "tt", "chunked_tt"):
            torch.manual_seed(42)
            model = Qwen3ForCausalLM(config).cuda().eval()
            cache = (
                DynamicCache()
                if kind == "dense"
                else TTCompressedCache(config, 0.1, 8, "int8_row", 128)
            )
            if kind == "chunked_tt":
                install_chunked_tt_decode(model)
            torch.cuda.reset_peak_memory_stats()
            output, prefill = timed_forward(model, tokens, cache)
            decode = 0.0
            next_token = output.logits[:, -1:].argmax(dim=-1)
            for _ in range(args.decode_tokens):
                output, elapsed = timed_forward(model, next_token, cache)
                decode += elapsed
                next_token = output.logits[:, -1:].argmax(dim=-1)
            row = {
                "cache": kind,
                "prompt_tokens": length,
                "decode_tokens": args.decode_tokens,
                "prefill_seconds": prefill,
                "decode_seconds": decode,
                "decode_tokens_per_second": args.decode_tokens / decode,
                "peak_cuda_memory_gib": torch.cuda.max_memory_allocated() / 1024**3,
            }
            if kind != "dense":
                row["tt_cache"] = cache.get_statistics()
            rows.append(row)
            print(json.dumps(row), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
