"""End-to-end smoke comparison of dense and chunked TT decode."""

from __future__ import annotations

import json
import time

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from experiments.chunked_qwen3 import install_chunked_tt_decode
from experiments.tt_cache import TTCompressedCache


def main() -> None:
    torch.manual_seed(42)
    config = Qwen3Config(
        vocab_size=256,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=4,
        num_attention_heads=8,
        num_key_value_heads=4,
        head_dim=16,
        max_position_embeddings=512,
    )
    model = Qwen3ForCausalLM(config).cuda().eval()
    tokens = torch.randint(0, config.vocab_size, (1, 128), device="cuda")
    results = []
    outputs = []
    for chunked in (False, True):
        if chunked:
            install_chunked_tt_decode(model)
        cache = TTCompressedCache(config, 0.1, 8, "int8_row", 32)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            output = model.generate(
                tokens, max_new_tokens=32, min_new_tokens=32,
                do_sample=False, past_key_values=cache,
            )
        torch.cuda.synchronize()
        stats = cache.get_statistics()
        row = {
            "chunked_decode": chunked,
            "seconds": time.perf_counter() - started,
            "peak_cuda_memory_mib": torch.cuda.max_memory_allocated() / 1024**2,
            "materialization_calls": stats["materialization_calls"],
            "reconstruction_seconds": stats["tt_reconstruction_seconds"],
            "compression_ratio": stats["compression_ratio"],
        }
        results.append(row)
        outputs.append(output)
        print(json.dumps(row), flush=True)
    print(json.dumps({"same_generated_tokens": bool(torch.equal(outputs[0], outputs[1]))}))


if __name__ == "__main__":
    main()
