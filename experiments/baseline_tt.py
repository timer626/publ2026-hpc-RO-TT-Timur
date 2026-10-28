"""Evaluate the per-example RAG baseline with a TT-compressed KV cache."""

from __future__ import annotations

import os

# TT-SVD repeatedly decomposes small matrices. Large BLAS thread pools add
# substantial overhead; callers can override these defaults in their shell.
for _thread_variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_thread_variable, "1")

import argparse
import json
import logging
import platform
import time
from datetime import datetime, timezone
from importlib.metadata import version as package_version
from pathlib import Path
from statistics import fmean
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from config import get_settings
from experiments.baseline import (
    RETRIEVAL_INSTRUCTION,
    build_generation_prompt,
    encode_queries,
    load_retrieval_data,
    retrieve_for_examples,
)
from experiments.plots import save_experiment_plots
from experiments.tt_cache import TTCompressedCache
from metrics import evaluate_answer


logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run HotpotQA RAG baseline with a TT-compressed KV cache."
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of passages retrieved within each example (default: 5).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Optionally evaluate only the first N examples.",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Start at this zero-based dataset index (default: 0).",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=64,
        help="Maximum number of generated answer tokens (default: 64).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for generation (default: 42).",
    )
    parser.add_argument(
        "--seed-per-example",
        action="store_true",
        help="Seed each example with seed plus its dataset index for shard-safe runs.",
    )
    parser.add_argument(
        "--tt-epsilon",
        type=float,
        default=1e-2,
        help="Relative Frobenius tolerance passed to ttpy TT-SVD (default: 0.01).",
    )
    parser.add_argument(
        "--tt-block-size",
        type=int,
        default=16,
        help="Generated KV tokens per TT block (default: 16).",
    )
    parser.add_argument(
        "--tt-prefill-block-size",
        type=int,
        help="Optional separate TT block size for prompt prefill.",
    )
    parser.add_argument(
        "--tt-core-quantization",
        choices=("none", "int8", "int8_row"),
        default="none",
        help="Optional TT-core storage quantization (default: none).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory in which this run's results will be saved.",
    )
    return parser.parse_args()


def generate_answer_with_tt_cache(
    model: Any,
    tokenizer: Any,
    question: str,
    documents: list[dict[str, Any]],
    device: torch.device,
    max_new_tokens: int,
    tt_epsilon: float,
    tt_block_size: int,
    tt_core_quantization: str,
    tt_prefill_block_size: int | None,
) -> tuple[str, int, int, float, dict[str, int | float]]:
    prompt = build_generation_prompt(tokenizer, question, documents)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_token_count = inputs["input_ids"].shape[1]
    cache = TTCompressedCache(
        config=model.config,
        epsilon=tt_epsilon,
        block_size=tt_block_size,
        core_quantization=tt_core_quantization,
        prefill_block_size=tt_prefill_block_size,
    )

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started_at = time.perf_counter()

    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.8,
            top_k=20,
            min_p=0.0,
            use_cache=True,
            past_key_values=cache,
        )

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed_seconds = time.perf_counter() - started_at

    answer_ids = generated[0, prompt_token_count:]
    answer = tokenizer.decode(answer_ids, skip_special_tokens=True).strip()
    return (
        answer,
        prompt_token_count,
        len(answer_ids),
        elapsed_seconds,
        cache.get_statistics(),
    )


def validate_args(args: argparse.Namespace) -> None:
    if args.top_k <= 0:
        raise ValueError("--top-k должен быть больше нуля")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit должен быть больше нуля")
    if args.offset < 0:
        raise ValueError("--offset не может быть отрицательным")
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens должен быть больше нуля")
    if not 0.0 < args.tt_epsilon < 1.0:
        raise ValueError("--tt-epsilon должен быть между 0 и 1")
    if args.tt_block_size <= 0:
        raise ValueError("--tt-block-size должен быть больше нуля")
    if args.tt_prefill_block_size is not None and args.tt_prefill_block_size <= 0:
        raise ValueError("--tt-prefill-block-size должен быть больше нуля")


def main() -> None:
    args = parse_args()
    validate_args(args)

    project_root = Path(__file__).resolve().parents[1]
    settings = get_settings(project_root)
    if settings.constants.is_first_run:
        raise RuntimeError(
            "Сначала завершите первичную подготовку, проверьте файлы и вручную "
            "установите constants.is_first_run = false в configs/config.toml"
        )

    dataset, document_index, document_embeddings, documents = load_retrieval_data(
        settings
    )
    end_index = min(
        args.offset + args.limit if args.limit is not None else len(dataset),
        len(dataset),
    )
    dataset = dataset.select(range(args.offset, end_index))
    examples = list(dataset)
    if not examples:
        raise ValueError("В выбранном наборе данных нет примеров для оценки")

    device = torch.device(
        "cuda"
        if settings.constants.device_cuda and torch.cuda.is_available()
        else "cpu"
    )
    logger.info("Устройство для вычислений: %s", device)

    questions = [example["question"] for example in examples]
    query_embeddings, query_embedding_seconds, query_peak_memory_gb = encode_queries(
        questions,
        settings,
        device,
    )
    if query_embeddings.shape[1] != document_embeddings.shape[1]:
        raise ValueError(
            "Размерность эмбеддингов запросов не совпадает с размерностью "
            "эмбеддингов документов"
        )

    retrieved_examples, search_seconds = retrieve_for_examples(
        examples,
        query_embeddings,
        document_index,
        document_embeddings,
        documents,
        args.top_k,
    )
    del query_embeddings, document_embeddings

    generator_tokenizer = AutoTokenizer.from_pretrained(
        settings.generator.local_dir,
        local_files_only=True,
    )
    generator_model = AutoModelForCausalLM.from_pretrained(
        settings.generator.local_dir,
        local_files_only=True,
        dtype="auto",
    ).to(device)
    generator_model.eval()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    set_seed(args.seed)
    output_root = args.output_dir or project_root / "experiments" / "results"
    run_id = datetime.now(timezone.utc).strftime("baseline_tt_%Y%m%dT%H%M%S%fZ")
    run_directory = output_root / run_id
    run_directory.mkdir(parents=True, exist_ok=False)
    predictions_path = run_directory / "predictions.jsonl"

    answer_em_scores = []
    answer_f1_scores = []
    retrieval_recall_scores = []
    retrieval_precision_scores = []
    all_supporting_documents_retrieved = []
    generation_seconds = []
    prompt_token_counts = []
    generated_token_counts = []
    cache_statistics = []

    with predictions_path.open("w", encoding="utf-8") as predictions_file:
        for row_number, retrieved_example in enumerate(retrieved_examples, start=1):
            example = retrieved_example["example"]
            example_index = args.offset + row_number - 1
            if args.seed_per_example:
                set_seed(args.seed + example_index)
            (
                prediction,
                prompt_tokens,
                generated_tokens,
                elapsed_seconds,
                cache_stats,
            ) = generate_answer_with_tt_cache(
                generator_model,
                generator_tokenizer,
                example["question"],
                retrieved_example["documents"],
                device,
                args.max_new_tokens,
                args.tt_epsilon,
                args.tt_block_size,
                args.tt_core_quantization,
                args.tt_prefill_block_size,
            )
            answer_metrics = evaluate_answer(prediction, example["answer"])
            retrieval_metrics = retrieved_example["retrieval_metrics"]

            result = {
                "id": example["id"],
                "example_index": example_index,
                "question": example["question"],
                "reference_answer": example["answer"],
                "prediction": prediction,
                **answer_metrics,
                **retrieval_metrics,
                "prompt_tokens": prompt_tokens,
                "generated_tokens": generated_tokens,
                "generation_seconds": elapsed_seconds,
                "retrieved_documents": retrieved_example["documents"],
                "tt_cache": cache_stats,
            }
            predictions_file.write(json.dumps(result, ensure_ascii=False) + "\n")
            predictions_file.flush()

            answer_em_scores.append(answer_metrics["answer_em"])
            answer_f1_scores.append(answer_metrics["answer_f1"])
            retrieval_recall_scores.append(
                retrieval_metrics["retrieval_recall_at_k"]
            )
            retrieval_precision_scores.append(
                retrieval_metrics["retrieval_precision_at_k"]
            )
            all_supporting_documents_retrieved.append(
                float(retrieval_metrics["all_supporting_documents_retrieved"])
            )
            generation_seconds.append(elapsed_seconds)
            prompt_token_counts.append(prompt_tokens)
            generated_token_counts.append(generated_tokens)
            cache_statistics.append(cache_stats)

            if row_number % 25 == 0 or row_number == len(retrieved_examples):
                logger.info(
                    "Обработано примеров: %s/%s",
                    row_number,
                    len(retrieved_examples),
                )

    peak_generation_memory_gb = None
    if device.type == "cuda":
        peak_generation_memory_gb = (
            torch.cuda.max_memory_allocated(device) / (1024 ** 3)
        )

    cache_byte_metrics = (
        "tt_core_bytes",
        "dense_tail_bytes",
        "dense_equivalent_bytes",
        "stored_bytes",
    )
    cache_summary = {
        f"mean_{name}": fmean(float(stats[name]) for stats in cache_statistics)
        for name in cache_byte_metrics
    }
    cache_summary.update(
        {
            "mean_compression_ratio": fmean(
                float(stats["compression_ratio"])
                for stats in cache_statistics
            ),
            "mean_relative_reconstruction_error": fmean(
                float(stats["mean_relative_reconstruction_error"])
                for stats in cache_statistics
            ),
            "max_relative_reconstruction_error": max(
                float(stats["max_relative_reconstruction_error"])
                for stats in cache_statistics
            ),
            "max_tt_rank": max(
                int(stats["max_tt_rank"]) for stats in cache_statistics
            ),
            "mean_tt_blocks": fmean(
                float(stats["tt_blocks"]) for stats in cache_statistics
            ),
            "total_tt_decomposition_seconds": sum(
                float(stats["tt_decomposition_seconds"])
                for stats in cache_statistics
            ),
            "total_tt_reconstruction_seconds": sum(
                float(stats["tt_reconstruction_seconds"])
                for stats in cache_statistics
            ),
        }
    )

    total_generation_seconds = sum(generation_seconds)
    total_generated_tokens = sum(generated_token_counts)
    summary = {
        "experiment": "per_example_rag_baseline_with_tt_kv_cache",
        "run_id": run_id,
        "dataset": {
            "repo_id": settings.dataset_raw.repo_id,
            "revision": settings.dataset_raw.revision,
            "config": settings.dataset_raw.name,
            "split": settings.dataset_raw.split,
            "start_index": args.offset,
            "end_index_exclusive": end_index,
            "examples_evaluated": len(examples),
        },
        "retrieval": {
            "scope": "documents indexed under the current example_id",
            "top_k": args.top_k,
            "query_instruction": RETRIEVAL_INSTRUCTION,
            "embedding_model_repo_id": settings.embedding.repo_id,
            "embedding_model_revision": settings.embedding.revision,
            "query_embedding_seconds": query_embedding_seconds,
            "search_seconds": search_seconds,
            "recall_at_k": fmean(retrieval_recall_scores),
            "precision_at_k": fmean(retrieval_precision_scores),
            "all_supporting_documents_retrieved_rate": fmean(
                all_supporting_documents_retrieved
            ),
        },
        "generation": {
            "model_repo_id": settings.generator.repo_id,
            "revision": settings.generator.revision,
            "kv_cache_compression": "TT-SVD via ttpy",
            "thinking_enabled": False,
            "max_new_tokens": args.max_new_tokens,
            "seed": args.seed,
            "seed_per_example": args.seed_per_example,
            "sampling": {
                "do_sample": True,
                "temperature": 0.7,
                "top_p": 0.8,
                "top_k": 20,
                "min_p": 0.0,
            },
            "latency_scope": (
                "model.generate including TT cache updates and reconstruction"
            ),
            "answer_em": fmean(answer_em_scores),
            "answer_f1": fmean(answer_f1_scores),
            "mean_generation_seconds": fmean(generation_seconds),
            "generated_tokens_per_second": (
                total_generated_tokens / total_generation_seconds
                if total_generation_seconds
                else 0.0
            ),
            "mean_prompt_tokens": fmean(prompt_token_counts),
            "mean_generated_tokens": fmean(generated_token_counts),
        },
        "kv_cache": {
            "format": (
                "ttpy TT-vector blocks, one decomposition for each key and "
                "value block"
            ),
            "ttpy_version": package_version("ttpy"),
            "tensor_shape": "batch x key_value_heads x sequence x head_dim",
            "storage_device": (
                "CPU TT cores; reconstructed dense K/V moved to model device"
            ),
            "byte_accounting": (
                "TT core and dense tail payload; Python overhead excluded"
            ),
            "compression_scope": (
                "persistent cache payload; temporary dense attention tensors "
                "are excluded"
            ),
            "epsilon": args.tt_epsilon,
            "generated_token_block_size": args.tt_block_size,
            "prefill_block_size": args.tt_prefill_block_size or args.tt_block_size,
            "core_quantization": args.tt_core_quantization,
            "storage_policy": (
                "compress prompt prefill and complete generated-token blocks; "
                "keep the newest incomplete block dense"
            ),
            "attention_input": "dense K/V reconstructed from TT blocks",
            **cache_summary,
        },
        "performance": {
            "device": str(device),
            "query_peak_cuda_memory_gb": query_peak_memory_gb,
            "generation_peak_cuda_memory_gb": peak_generation_memory_gb,
        },
        "software": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
        },
        "artifacts": {
            "predictions": str(predictions_path),
        },
    }

    summary_path = run_directory / "summary.json"
    summary["artifacts"]["plots"] = save_experiment_plots(
        predictions_path,
        summary,
        run_directory,
    )
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("Итоги TT baseline: %s", summary_path)
    logger.info(
        "EM=%.4f, F1=%.4f, cache ratio=%.3f, max rank=%s",
        summary["generation"]["answer_em"],
        summary["generation"]["answer_f1"],
        summary["kv_cache"]["mean_compression_ratio"],
        summary["kv_cache"]["max_tt_rank"],
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    main()
