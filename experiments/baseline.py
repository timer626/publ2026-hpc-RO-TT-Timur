"""Evaluate per-example RAG with the original, uncompressed model."""

from __future__ import annotations

import argparse
import json
import logging
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean
from typing import Any

import numpy as np
import torch
import torch.nn.functional as functional
from datasets import load_from_disk
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, set_seed

from config import Settings, get_settings
from experiments.plots import save_experiment_plots
from metrics import evaluate_answer


logger = logging.getLogger(__name__)

RETRIEVAL_INSTRUCTION = (
    "Given a question, retrieve the passages from this example that contain "
    "evidence needed to answer it."
)
GENERATION_SYSTEM_PROMPT = (
    "Answer the question using only the provided passages. "
    "Return a concise answer without explanation."
)


def last_token_pool(
    last_hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Pool embeddings in the same way as the document embedding pipeline."""
    is_left_padded = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if bool(is_left_padded):
        return last_hidden_states[:, -1]

    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_indices = torch.arange(
        last_hidden_states.shape[0],
        device=last_hidden_states.device,
    )
    return last_hidden_states[batch_indices, sequence_lengths]


def build_query_text(question: str) -> str:
    return f"Instruct: {RETRIEVAL_INSTRUCTION}\nQuery: {question}"


def validate_document_index(
    document_index: dict[str, list[int]],
    documents: list[dict[str, Any]],
) -> None:
    """Ensure every index entry points to its own document exactly once."""
    document_owners: list[str | None] = [None] * len(documents)

    for example_id, document_indices in document_index.items():
        if not isinstance(document_indices, list):
            raise ValueError(
                f"Индекс для example_id={example_id} должен быть списком"
            )

        for document_index_value in document_indices:
            if (
                not isinstance(document_index_value, int)
                or isinstance(document_index_value, bool)
                or not 0 <= document_index_value < len(documents)
            ):
                raise ValueError(
                    f"Некорректный номер документа в индексе "
                    f"example_id={example_id}: {document_index_value}"
                )

            document = documents[document_index_value]
            if not {"example_id", "title", "text"} <= document.keys():
                raise ValueError(
                    f"В документе с индексом {document_index_value} "
                    "отсутствуют обязательные поля"
                )

            if str(document["example_id"]) != example_id:
                raise ValueError(
                    f"Индекс example_id={example_id} указывает на документ "
                    f"другого примера: {document['example_id']}"
                )

            if document_owners[document_index_value] is not None:
                raise ValueError(
                    f"Документ с индексом {document_index_value} "
                    "встречается в индексе более одного раза"
                )
            document_owners[document_index_value] = example_id

    unindexed_documents = [
        index
        for index, owner in enumerate(document_owners)
        if owner is None
    ]
    if unindexed_documents:
        raise ValueError(
            f"В индексе отсутствуют документы, например индекс "
            f"{unindexed_documents[0]}"
        )


def load_retrieval_data(
    settings: Settings,
) -> tuple[Any, dict[str, Any], np.ndarray, list[dict[str, Any]]]:
    dataset = load_from_disk(settings.dataset_raw.local_dir)

    with settings.dataset_prepared.local_index_file_documents.open(
        "r",
        encoding="utf-8",
    ) as index_file:
        document_index = json.load(index_file)

    document_embeddings = np.load(settings.dataset_prepared.local_file_embeddings)
    if document_embeddings.ndim != 2:
        raise ValueError("documents.npy должен содержать двумерную матрицу эмбеддингов")

    with settings.dataset_prepared.local_file_documents.open(
        "r",
        encoding="utf-8",
    ) as documents_file:
        documents = [
            json.loads(line)
            for line in documents_file
            if line.strip()
        ]

    if len(documents) != len(document_embeddings):
        raise ValueError(
            "Количество документов не совпадает с количеством строк в documents.npy"
        )
    validate_document_index(document_index, documents)

    return dataset, document_index, document_embeddings, documents


def encode_queries(
    questions: list[str],
    settings: Settings,
    device: torch.device,
) -> tuple[np.ndarray, float, float | None]:
    tokenizer = AutoTokenizer.from_pretrained(
        settings.embedding.local_dir,
        local_files_only=True,
        padding_side=settings.constants.padding_side,
    )
    model = AutoModel.from_pretrained(
        settings.embedding.local_dir,
        local_files_only=True,
        dtype="auto",
    ).to(device)
    model.eval()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    query_texts = [build_query_text(question) for question in questions]
    query_embeddings = []
    started_at = time.perf_counter()

    for start in range(0, len(query_texts), settings.constants.batch_size):
        batch_texts = query_texts[start:start + settings.constants.batch_size]
        inputs = tokenizer(
            batch_texts,
            padding=settings.constants.padding,
            truncation=settings.constants.truncation,
            max_length=settings.constants.max_length,
            return_tensors="pt",
        ).to(device)

        with torch.inference_mode():
            outputs = model(**inputs)
            embeddings = last_token_pool(
                outputs.last_hidden_state,
                inputs["attention_mask"],
            )
            embeddings = functional.normalize(embeddings.float(), p=2, dim=1)

        query_embeddings.append(embeddings.cpu().numpy())

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    elapsed_seconds = time.perf_counter() - started_at
    peak_memory_gb = None
    if device.type == "cuda":
        peak_memory_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)

    result = np.concatenate(query_embeddings, axis=0)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return result, elapsed_seconds, peak_memory_gb


def retrieve_for_examples(
    examples: list[dict[str, Any]],
    queries: np.ndarray,
    document_index: dict[str, list[int]],
    document_embeddings: np.ndarray,
    documents: list[dict[str, Any]],
    top_k: int,
) -> tuple[list[dict[str, Any]], float]:
    retrieved_examples = []
    search_seconds = 0.0

    for example, query_embedding in zip(examples, queries, strict=True):
        example_id = str(example["id"])
        candidate_indices = document_index.get(example_id)
        if not candidate_indices:
            raise KeyError(f"Для example_id={example_id} нет документов в индексе")

        candidate_embeddings = document_embeddings[candidate_indices]
        started_at = time.perf_counter()
        scores = candidate_embeddings @ query_embedding
        ranked_positions = np.argsort(-scores, kind="stable")[:top_k]
        search_seconds += time.perf_counter() - started_at

        retrieved_documents = []
        for position in ranked_positions:
            document_index_value = candidate_indices[int(position)]
            document = documents[document_index_value]
            retrieved_documents.append(
                {
                    "title": document["title"],
                    "text": document["text"],
                    "score": float(scores[position]),
                }
            )

        gold_titles = set(example["supporting_facts"]["title"])
        retrieved_titles = {document["title"] for document in retrieved_documents}
        relevant_count = len(gold_titles & retrieved_titles)
        retrieval_metrics = {
            "retrieval_recall_at_k": relevant_count / len(gold_titles),
            "retrieval_precision_at_k": relevant_count / len(retrieved_documents),
            "all_supporting_documents_retrieved": gold_titles <= retrieved_titles,
        }

        retrieved_examples.append(
            {
                "example": example,
                "documents": retrieved_documents,
                "retrieval_metrics": retrieval_metrics,
            }
        )

    return retrieved_examples, search_seconds


def build_generation_prompt(
    tokenizer: Any,
    question: str,
    documents: list[dict[str, Any]],
) -> str:
    context = "\n\n".join(
        document["text"]
        for document in documents
    )
    messages = [
        {"role": "system", "content": GENERATION_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Passages:\n{context}\n\nQuestion: {question}",
        },
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def generate_answer(
    model: Any,
    tokenizer: Any,
    question: str,
    documents: list[dict[str, Any]],
    device: torch.device,
    max_new_tokens: int,
) -> tuple[str, int, int, float]:
    prompt = build_generation_prompt(tokenizer, question, documents)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_token_count = inputs["input_ids"].shape[1]

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
        )

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed_seconds = time.perf_counter() - started_at

    answer_ids = generated[0, prompt_token_count:]
    answer = tokenizer.decode(answer_ids, skip_special_tokens=True).strip()
    return answer, prompt_token_count, len(answer_ids), elapsed_seconds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run an uncompressed per-example RAG baseline on HotpotQA."
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
        "--output-dir",
        type=Path,
        help="Directory in which this run's results will be saved.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.top_k <= 0:
        raise ValueError("--top-k должен быть больше нуля")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit должен быть больше нуля")
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens должен быть больше нуля")

    project_root = Path(__file__).resolve().parents[1]
    settings = get_settings(project_root)
    if settings.constants.is_first_run:
        raise RuntimeError(
            "Сначала завершите первичную подготовку, проверьте файлы и вручную "
            "установите constants.is_first_run = false в configs/config.toml"
        )

    dataset, document_index, document_embeddings, documents = load_retrieval_data(settings)
    if args.limit is not None:
        dataset = dataset.select(range(min(args.limit, len(dataset))))
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
    run_id = datetime.now(timezone.utc).strftime("baseline_%Y%m%dT%H%M%S%fZ")
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

    with predictions_path.open("w", encoding="utf-8") as predictions_file:
        for row_number, retrieved_example in enumerate(retrieved_examples, start=1):
            example = retrieved_example["example"]
            prediction, prompt_tokens, generated_tokens, elapsed_seconds = generate_answer(
                generator_model,
                generator_tokenizer,
                example["question"],
                retrieved_example["documents"],
                device,
                args.max_new_tokens,
            )
            answer_metrics = evaluate_answer(prediction, example["answer"])
            retrieval_metrics = retrieved_example["retrieval_metrics"]

            result = {
                "id": example["id"],
                "question": example["question"],
                "reference_answer": example["answer"],
                "prediction": prediction,
                **answer_metrics,
                **retrieval_metrics,
                "prompt_tokens": prompt_tokens,
                "generated_tokens": generated_tokens,
                "generation_seconds": elapsed_seconds,
                "retrieved_documents": retrieved_example["documents"],
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

            if row_number % 25 == 0 or row_number == len(retrieved_examples):
                logger.info("Обработано примеров: %s/%s", row_number, len(retrieved_examples))

    peak_generation_memory_gb = None
    if device.type == "cuda":
        peak_generation_memory_gb = (
            torch.cuda.max_memory_allocated(device) / (1024 ** 3)
        )

    total_generation_seconds = sum(generation_seconds)
    total_generated_tokens = sum(generated_token_counts)
    summary = {
        "experiment": "per_example_rag_baseline_without_kv_compression",
        "run_id": run_id,
        "dataset": {
            "repo_id": settings.dataset_raw.repo_id,
            "revision": settings.dataset_raw.revision,
            "config": settings.dataset_raw.name,
            "split": settings.dataset_raw.split,
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
            "kv_cache_compression": "none",
            "thinking_enabled": False,
            "max_new_tokens": args.max_new_tokens,
            "seed": args.seed,
            "sampling": {
                "do_sample": True,
                "temperature": 0.7,
                "top_p": 0.8,
                "top_k": 20,
                "min_p": 0.0,
            },
            "latency_scope": "model.generate only; prompt tokenization excluded",
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
    logger.info("Итоги baseline: %s", summary_path)
    logger.info(
        "EM=%.4f, F1=%.4f, Recall@%s=%.4f",
        summary["generation"]["answer_em"],
        summary["generation"]["answer_f1"],
        args.top_k,
        summary["retrieval"]["recall_at_k"],
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    main()
