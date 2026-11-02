"""Matched long-prompt pilot with pinned Qwen3-8B and fixed RAG documents."""

from __future__ import annotations

import os

for _variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_variable] = "1"

import argparse
import json
import time
from itertools import product
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from config import get_settings
from experiments.baseline import build_generation_prompt
from experiments.chunked_qwen3 import install_chunked_tt_decode
from experiments.tt_cache import TTCompressedCache
from metrics import evaluate_answer


def prompt_near_length(tokenizer, row: dict, target: int, distinct_fill: bool = False) -> tuple[str, int]:
    documents = row["retrieved_documents"]
    repetitions = 1
    best = build_generation_prompt(tokenizer, row["question"], documents)
    best_length = len(tokenizer(best)["input_ids"])
    if best_length > target:
        raise ValueError(f"Base prompt {best_length} exceeds target {target}")
    if distinct_fill:
        for filler in row.get("filler_documents", []):
            candidate = build_generation_prompt(tokenizer, row["question"], documents + [filler])
            length = len(tokenizer(candidate)["input_ids"])
            if length > target:
                break
            documents = documents + [filler]
            best, best_length = candidate, length
        return best, best_length
    while repetitions < 32:
        repetitions += 1
        candidate = build_generation_prompt(tokenizer, row["question"], documents * repetitions)
        length = len(tokenizer(candidate)["input_ids"])
        if length > target:
            break
        best, best_length = candidate, length
    return best, best_length


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt-targets", type=int, nargs="+", default=[2048, 4096])
    parser.add_argument("--new-tokens", type=int, nargs="+", default=[64, 256])
    parser.add_argument("--cache", choices=["dense", "tt"], required=True)
    parser.add_argument("--chunked-decode", action="store_true")
    parser.add_argument("--natural-stop", action="store_true")
    parser.add_argument("--questions", type=int, default=1)
    parser.add_argument("--distinct-fill", action="store_true")
    args = parser.parse_args()
    settings = get_settings(Path(__file__).resolve().parents[1])
    tokenizer = AutoTokenizer.from_pretrained(settings.generator.local_dir, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        settings.generator.local_dir, local_files_only=True, dtype="auto"
    ).cuda().eval()
    if args.chunked_decode:
        if args.cache != "tt":
            raise ValueError("Chunked decode requires --cache tt")
        install_chunked_tt_decode(model)
    if args.questions < 1:
        raise ValueError("--questions must be positive")
    rows = [json.loads(line) for line in args.source.read_text().splitlines()[:args.questions]]
    results = []
    for row, target in product(rows, args.prompt_targets):
        prompt, prompt_length = prompt_near_length(tokenizer, row, target, args.distinct_fill)
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        for new_tokens in args.new_tokens:
            set_seed(42)
            cache = (
                TTCompressedCache(model.config, 0.1, 8, "int8_row", 128)
                if args.cache == "tt"
                else None
            )
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            started = time.perf_counter()
            generation = dict(
                max_new_tokens=new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.8,
                top_k=20,
                min_p=0.0,
                use_cache=True,
                past_key_values=cache,
            )
            if not args.natural_stop:
                generation["min_new_tokens"] = new_tokens
            with torch.inference_mode():
                generated = model.generate(
                    **inputs,
                    **generation,
                )
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            answer = tokenizer.decode(generated[0, prompt_length:], skip_special_tokens=True).strip()
            result = {
                "cache": args.cache,
                "chunked_decode": args.chunked_decode,
                "question_id": row["id"],
                "prompt_target": target,
                "prompt_tokens": prompt_length,
                "new_tokens": new_tokens,
                "actual_new_tokens": generated.shape[-1] - prompt_length,
                "elapsed_seconds": elapsed,
                "tokens_per_second": (generated.shape[-1] - prompt_length) / elapsed,
                "natural_stop": args.natural_stop,
                "distinct_fill": args.distinct_fill,
                "peak_cuda_memory_gib": torch.cuda.max_memory_allocated() / 1024**3,
                "prediction": answer,
                **evaluate_answer(answer, row["reference_answer"]),
            }
            if cache is not None:
                result["tt_cache"] = cache.get_statistics()
            results.append(result)
            print(json.dumps(result, ensure_ascii=False), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
