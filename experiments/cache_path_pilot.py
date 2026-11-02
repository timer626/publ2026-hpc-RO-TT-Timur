"""Profile dense versus TT cache on synthetic long-context generation.

This is an engineering pilot: the random Qwen3 model measures cache-path
costs, peak memory, and decode throughput without retrieval or answer quality.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from experiments.tt_cache import TTCompressedCache


def run_case(
    model: Qwen3ForCausalLM,
    prompt_tokens: int,
    new_tokens: int,
    cache_kind: str,
    epsilon: float,
    prefill_block_size: int,
    block_size: int,
) -> dict[str, int | float | str]:
    device = next(model.parameters()).device
    input_ids = torch.randint(0, model.config.vocab_size, (1, prompt_tokens), device=device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    cache = None
    if cache_kind == "tt":
        cache = TTCompressedCache(
            model.config,
            epsilon=epsilon,
            block_size=block_size,
            prefill_block_size=prefill_block_size,
            core_quantization="int8_row",
        )
    started = time.perf_counter()
    with torch.inference_mode():
        generated = model.generate(
            input_ids,
            max_new_tokens=new_tokens,
            min_new_tokens=new_tokens,
            do_sample=False,
            use_cache=True,
            past_key_values=cache,
        )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak = torch.cuda.max_memory_allocated(device) / (1024**3)
    else:
        peak = 0.0
    elapsed = time.perf_counter() - started
    actual_new_tokens = generated.shape[-1] - prompt_tokens
    result: dict[str, int | float | str] = {
        "cache": cache_kind,
        "prompt_tokens": prompt_tokens,
        "new_tokens": new_tokens,
        "actual_new_tokens": actual_new_tokens,
        "elapsed_seconds": elapsed,
        "decode_tokens_per_second": actual_new_tokens / max(elapsed, 1e-9),
        "peak_cuda_memory_gib": peak,
    }
    if cache is not None:
        stats = cache.get_statistics()
        result.update(
            {
                "stored_bytes": stats["stored_bytes"],
                "dense_equivalent_bytes": stats["dense_equivalent_bytes"],
                "compression_ratio": stats["compression_ratio"],
                "mean_reconstruction_error": stats["mean_relative_reconstruction_error"],
                "tt_decomposition_seconds": sum(
                    layer.tt_decomposition_seconds for layer in cache.layers
                ),
                "tt_reconstruction_seconds": sum(
                    layer.tt_reconstruction_seconds for layer in cache.layers
                ),
            }
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt-tokens", type=int, nargs="+", default=[512, 2048, 4096])
    parser.add_argument("--new-tokens", type=int, nargs="+", default=[64, 256])
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this pilot")
    config = Qwen3Config(
        vocab_size=256,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=4,
        num_attention_heads=8,
        num_key_value_heads=4,
        head_dim=32,
        max_position_embeddings=max(args.prompt_tokens) + max(args.new_tokens) + 8,
    )
    model = Qwen3ForCausalLM(config).cuda().eval()
    rows = []
    for prompt_tokens in args.prompt_tokens:
        for new_tokens in args.new_tokens:
            for cache_kind in ("dense", "tt"):
                rows.append(
                    run_case(
                        model,
                        prompt_tokens,
                        new_tokens,
                        cache_kind,
                        epsilon=0.1,
                        prefill_block_size=128,
                        block_size=8,
                    )
                )
                print(json.dumps(rows[-1], ensure_ascii=False), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
