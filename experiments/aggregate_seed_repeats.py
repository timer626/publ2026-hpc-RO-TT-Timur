"""Summarize matched baseline/TT comparisons across generation seeds."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from statistics import fmean, stdev
from typing import Any


def run_rows(root: Path, run_id: str) -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    paths = list(root.glob(f"hpc0n*/{run_id}/summary.json"))
    if len(paths) != 1:
        raise ValueError(f"Expected one summary for {run_id}; found {len(paths)}")
    summary = json.loads(paths[0].read_text(encoding="utf-8"))
    with (paths[0].parent / "predictions.jsonl").open(encoding="utf-8") as source:
        rows = [json.loads(line) for line in source]
    indexed = {row["example_index"]: row for row in rows}
    if len(indexed) != len(rows) or len(rows) != summary["dataset"]["examples_evaluated"]:
        raise ValueError(f"Incomplete or overlapping rows in {run_id}")
    if not summary["generation"]["seed_per_example"]:
        raise ValueError(f"Run {run_id} lacks per-example seeding")
    return summary, indexed


def documents(row: dict[str, Any]) -> list[tuple[str, str]]:
    return [(doc["title"], doc["text"]) for doc in row["retrieved_documents"]]


def cluster_interval(values: list[float], replicates: int = 5000) -> list[float]:
    generator = random.Random(42)
    count = len(values)
    samples = sorted(
        sum(values[generator.randrange(count)] for _ in range(count)) / count
        for _ in range(replicates)
    )
    return [samples[int(0.025 * replicates)], samples[int(0.975 * replicates) - 1]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparisons", nargs="+", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    by_seed: dict[int, dict[str, Any]] = {}
    reference_ids: dict[int, str] | None = None
    reference_documents: dict[int, list[tuple[str, str]]] | None = None
    reference_settings: dict[str, Any] | None = None
    indices: set[int] | None = None
    for comparison_path in args.comparisons:
        comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
        baseline_runs = [run_rows(args.runs_root, run_id) for run_id in comparison["baseline_runs"]]
        tt_runs = [run_rows(args.runs_root, run_id) for run_id in comparison["tt_runs"]]
        all_summaries = [summary for summary, _ in baseline_runs + tt_runs]
        seeds = {summary["generation"]["seed"] for summary in all_summaries}
        if len(seeds) != 1:
            raise ValueError(f"Mixed seeds in {comparison_path}")
        seed = seeds.pop()
        if seed in by_seed:
            raise ValueError(f"Repeated base seed {seed}")
        if reference_settings is None:
            reference_settings = comparison["tt_settings"]
        elif comparison["tt_settings"] != reference_settings:
            raise ValueError("TT settings differ between seeds")

        baseline = {index: row for _, rows in baseline_runs for index, row in rows.items()}
        tt = {index: row for _, rows in tt_runs for index, row in rows.items()}
        if len(baseline) != sum(len(rows) for _, rows in baseline_runs):
            raise ValueError(f"Overlapping baseline shards for seed {seed}")
        if len(tt) != sum(len(rows) for _, rows in tt_runs):
            raise ValueError(f"Overlapping TT shards for seed {seed}")
        if set(baseline) != set(tt) or len(tt) != comparison["examples_evaluated"]:
            raise ValueError(f"Different or incomplete question sets for seed {seed}")
        if indices is None:
            indices = set(tt)
            reference_ids = {index: row["id"] for index, row in baseline.items()}
            reference_documents = {index: documents(row) for index, row in baseline.items()}
        elif set(tt) != indices:
            raise ValueError("Question indices differ between seeds")
        for index in sorted(tt):
            if baseline[index]["id"] != tt[index]["id"]:
                raise ValueError(f"Different question at index {index}, seed {seed}")
            if baseline[index]["id"] != reference_ids[index]:
                raise ValueError(f"Question IDs differ between seeds at index {index}")
            if documents(baseline[index]) != documents(tt[index]):
                raise ValueError(f"Retrieval mismatch at index {index}, seed {seed}")
            if documents(baseline[index]) != reference_documents[index]:
                raise ValueError(f"Retrieval differs between seeds at index {index}")

        by_seed[seed] = {
            "comparison": str(comparison_path),
            "baseline_runs": comparison["baseline_runs"],
            "tt_runs": comparison["tt_runs"],
            "baseline_answer_em": fmean(row["answer_em"] for row in baseline.values()),
            "tt_answer_em": fmean(row["answer_em"] for row in tt.values()),
            "baseline_answer_f1": fmean(row["answer_f1"] for row in baseline.values()),
            "tt_answer_f1": fmean(row["answer_f1"] for row in tt.values()),
            "delta_em": fmean(tt[index]["answer_em"] - baseline[index]["answer_em"] for index in tt),
            "delta_f1": fmean(tt[index]["answer_f1"] - baseline[index]["answer_f1"] for index in tt),
            "rows": {index: (baseline[index], tt[index]) for index in tt},
        }

    ordered_seeds = sorted(by_seed)
    ordered_indices = sorted(indices)
    output: dict[str, Any] = {
        "question_count": len(ordered_indices),
        "base_seeds": ordered_seeds,
        "seed_per_example": True,
        "tt_settings": reference_settings,
        "retrieval_mismatches": 0,
        "per_seed": [],
    }
    for seed in ordered_seeds:
        entry = dict(by_seed[seed])
        del entry["rows"]
        output["per_seed"].append({"seed": seed, **entry})
    output["across_seed"] = {}
    for metric in ("em", "f1"):
        seed_deltas = [by_seed[seed][f"delta_{metric}"] for seed in ordered_seeds]
        question_deltas = [
            fmean(
                by_seed[seed]["rows"][index][1][f"answer_{metric}"]
                - by_seed[seed]["rows"][index][0][f"answer_{metric}"]
                for seed in ordered_seeds
            )
            for index in ordered_indices
        ]
        output["across_seed"][metric] = {
            "mean_paired_delta": fmean(seed_deltas),
            "seed_delta_range": [min(seed_deltas), max(seed_deltas)],
            "seed_delta_standard_deviation": stdev(seed_deltas) if len(seed_deltas) > 1 else None,
            "question_cluster_bootstrap_95pct": cluster_interval(question_deltas),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
