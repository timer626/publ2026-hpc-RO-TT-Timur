"""Validate and combine independently seeded RAG evaluation shards."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from statistics import fmean
from typing import Any


def load_run(summary_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    predictions_path = summary_path.parent / "predictions.jsonl"
    with predictions_path.open(encoding="utf-8") as predictions_file:
        predictions = [json.loads(line) for line in predictions_file]
    if len(predictions) != summary["dataset"]["examples_evaluated"]:
        raise ValueError(f"Неполный запуск: {summary_path}")
    start = summary["dataset"]["start_index"]
    end = summary["dataset"]["end_index_exclusive"]
    if [row["example_index"] for row in predictions] != list(range(start, end)):
        raise ValueError(f"Неверный порядок примеров: {summary_path}")
    if not summary["generation"]["seed_per_example"]:
        raise ValueError(f"Для объединения нужен --seed-per-example: {summary_path}")
    return summary, predictions


def average(rows: list[dict[str, Any]], key: str) -> float:
    return fmean(float(row[key]) for row in rows)


def paired_bootstrap_interval(
    baseline_rows: list[dict[str, Any]],
    tt_rows: list[dict[str, Any]],
    key: str,
    replicates: int = 5000,
) -> list[float]:
    deltas = [
        float(tt[key]) - float(baseline[key])
        for baseline, tt in zip(baseline_rows, tt_rows)
    ]
    random_generator = random.Random(42)
    sample_means = sorted(
        sum(deltas[random_generator.randrange(len(deltas))] for _ in deltas)
        / len(deltas)
        for _ in range(replicates)
    )
    return [
        sample_means[int(0.025 * replicates)],
        sample_means[int(0.975 * replicates) - 1],
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, nargs="+", required=True)
    parser.add_argument("--tt", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    baseline_runs = [load_run(path) for path in args.baseline]
    baseline_summary = baseline_runs[0][0]
    baseline_rows = [row for _, rows in baseline_runs for row in rows]
    tt_runs = [load_run(path) for path in args.tt]
    baseline_by_index = {row["example_index"]: row for row in baseline_rows}
    if len(baseline_by_index) != len(baseline_rows):
        raise ValueError("В baseline повторяются индексы примеров")

    tt_rows = [row for _, rows in tt_runs for row in rows]
    tt_by_index = {row["example_index"]: row for row in tt_rows}
    if len(tt_by_index) != len(tt_rows):
        raise ValueError("TT shards пересекаются")
    if set(tt_by_index) != set(baseline_by_index):
        raise ValueError("TT shards не покрывают те же примеры, что baseline")

    reference_tt = tt_runs[0][0]
    for summary, _ in baseline_runs:
        for section, keys in (
            ("dataset", ("repo_id", "revision", "config", "split")),
            ("retrieval", ("top_k", "embedding_model_revision")),
            ("generation", ("revision", "max_new_tokens", "seed", "sampling")),
        ):
            if any(summary[section][key] != baseline_summary[section][key] for key in keys):
                raise ValueError(f"Несовместимые baseline shards: {summary['run_id']}")
    for section, keys in (
        ("dataset", ("repo_id", "revision", "config", "split")),
        ("retrieval", ("top_k", "embedding_model_revision")),
        ("generation", ("revision", "max_new_tokens", "seed", "sampling")),
    ):
        if any(
            baseline_summary[section][key] != reference_tt[section][key]
            for key in keys
        ):
            raise ValueError(f"Несовместимые baseline и TT: {section}")
    for summary, _ in tt_runs:
        for section, keys in (
            ("dataset", ("repo_id", "revision", "config", "split")),
            ("retrieval", ("top_k", "embedding_model_revision")),
            ("generation", ("revision", "max_new_tokens", "seed", "sampling")),
            ("kv_cache", ("epsilon", "generated_token_block_size", "core_quantization")),
        ):
            if any(summary[section][key] != reference_tt[section][key] for key in keys):
                raise ValueError(f"Несовместимые TT shards: {summary['run_id']}")

    retrieval_mismatches = 0
    for index, tt_row in tt_by_index.items():
        baseline_row = baseline_by_index[index]
        if (tt_row["id"], tt_row["question"], tt_row["reference_answer"]) != (
            baseline_row["id"],
            baseline_row["question"],
            baseline_row["reference_answer"],
        ):
            raise ValueError(f"Пример {index} различается между baseline и TT")
        tt_documents = [
            (doc["title"], doc["text"]) for doc in tt_row["retrieved_documents"]
        ]
        baseline_documents = [
            (doc["title"], doc["text"])
            for doc in baseline_row["retrieved_documents"]
        ]
        if tt_documents != baseline_documents:
            retrieval_mismatches += 1

    tt_rows.sort(key=lambda row: row["example_index"])
    baseline_rows.sort(key=lambda row: row["example_index"])
    dense_bytes = sum(row["tt_cache"]["dense_equivalent_bytes"] for row in tt_rows)
    stored_bytes = sum(row["tt_cache"]["stored_bytes"] for row in tt_rows)
    output = {
        "examples_evaluated": len(tt_rows),
        "baseline_runs": [summary["run_id"] for summary, _ in baseline_runs],
        "tt_runs": [summary["run_id"] for summary, _ in tt_runs],
        "tt_settings": {
            key: reference_tt["kv_cache"][key]
            for key in ("epsilon", "generated_token_block_size", "core_quantization")
        },
        "retrieval_mismatches": retrieval_mismatches,
        "baseline": {
            "answer_em": average(baseline_rows, "answer_em"),
            "answer_f1": average(baseline_rows, "answer_f1"),
            "mean_generation_seconds": average(baseline_rows, "generation_seconds"),
            "peak_cuda_memory_gb": max(
                summary["performance"]["generation_peak_cuda_memory_gb"]
                for summary, _ in baseline_runs
            ),
        },
        "tt": {
            "answer_em": average(tt_rows, "answer_em"),
            "answer_f1": average(tt_rows, "answer_f1"),
            "mean_generation_seconds": average(tt_rows, "generation_seconds"),
            "mean_compression_ratio": fmean(
                row["tt_cache"]["compression_ratio"] for row in tt_rows
            ),
            "total_payload_compression_ratio": dense_bytes / stored_bytes,
            "dense_equivalent_bytes": dense_bytes,
            "stored_bytes": stored_bytes,
            "mean_reconstruction_error": fmean(
                row["tt_cache"]["mean_relative_reconstruction_error"]
                for row in tt_rows
            ),
            "peak_cuda_memory_gb": max(
                summary["performance"]["generation_peak_cuda_memory_gb"]
                for summary, _ in tt_runs
            ),
        },
        "paired_difference": {
            "answer_em": average(tt_rows, "answer_em")
            - average(baseline_rows, "answer_em"),
            "answer_em_bootstrap_95pct": paired_bootstrap_interval(
                baseline_rows, tt_rows, "answer_em"
            ),
            "answer_f1": average(tt_rows, "answer_f1")
            - average(baseline_rows, "answer_f1"),
            "answer_f1_bootstrap_95pct": paired_bootstrap_interval(
                baseline_rows, tt_rows, "answer_f1"
            ),
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "comparison.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (args.output_dir / "tt_predictions.jsonl").open("w", encoding="utf-8") as output_file:
        for row in tt_rows:
            output_file.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
