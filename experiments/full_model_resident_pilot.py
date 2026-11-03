"""Fixed-question Qwen3-8B latency comparison for TT decode paths."""

from __future__ import annotations

import os

for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from experiments.chunked_qwen3 import install_chunked_tt_decode
from experiments.tt_cache import TTCompressedCache


def build_prompt(tokenizer, question: str, documents: list[dict]) -> str:
    context = "\n\n".join(document["text"] for document in documents)
    return tokenizer.apply_chat_template(
        [
            {"role": "system", "content": "Answer the question using only the provided passages. Return a concise answer without explanation."},
            {"role": "user", "content": f"Passages:\n{context}\n\nQuestion: {question}"},
        ],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def prompt_near_length(tokenizer, row: dict, target: int) -> tuple[str, int]:
    documents = row["retrieved_documents"]
    best = build_prompt(tokenizer, row["question"], documents)
    best_length = len(tokenizer(best)["input_ids"])
    if best_length > target:
        raise ValueError(f"Base prompt {best_length} exceeds target {target}")
    for repetitions in range(2, 33):
        candidate = build_prompt(tokenizer, row["question"], documents * repetitions)
        length = len(tokenizer(candidate)["input_ids"])
        if length > target:
            break
        best, best_length = candidate, length
    return best, best_length


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=["dense", "triton", "resident"], required=True)
    parser.add_argument("--prompt-targets", type=int, nargs="+", default=[2048, 8192])
    parser.add_argument("--new-tokens", type=int, default=64)
    args = parser.parse_args()
    row = json.loads(args.source.read_text().splitlines()[0])
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, local_files_only=True, dtype="auto").cuda().eval()
    if args.mode != "dense":
        install_chunked_tt_decode(model, triton_streaming=args.mode == "triton", gpu_resident=args.mode == "resident")
    results = []
    for target in args.prompt_targets:
        prompt, prompt_length = prompt_near_length(tokenizer, row, target)
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        set_seed(42)
        cache = None if args.mode == "dense" else TTCompressedCache(model.config, 0.1, 8, "int8_row", 128)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(
                **inputs, max_new_tokens=args.new_tokens, min_new_tokens=args.new_tokens,
                do_sample=True, temperature=0.7, top_p=0.8, top_k=20, min_p=0.0,
                use_cache=True, past_key_values=cache,
            )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        continuation = generated[0, prompt_length:].tolist()
        result = {
            "mode": args.mode, "question_id": row["id"], "prompt_target": target,
            "prompt_tokens": prompt_length, "new_tokens": args.new_tokens,
            "actual_new_tokens": len(continuation), "elapsed_seconds": elapsed,
            "peak_cuda_memory_gib": torch.cuda.max_memory_allocated() / 1024**3,
            "continuation_sha256": hashlib.sha256(json.dumps(continuation).encode()).hexdigest(),
            "prediction": tokenizer.decode(continuation, skip_special_tokens=True).strip(),
        }
        if cache is not None:
            result["tt_cache"] = cache.get_statistics()
        results.append(result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
