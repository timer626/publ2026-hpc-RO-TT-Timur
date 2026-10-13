"""Create and save plots for experiment result files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


def save_experiment_plots(
    predictions_path: Path,
    summary: dict[str, Any],
    run_directory: Path,
) -> dict[str, str]:
    """Save quality, generation, and optional TT-cache figures."""
    predictions = _load_predictions(predictions_path)
    plots_directory = run_directory / "plots"
    plots_directory.mkdir(parents=True, exist_ok=True)

    plot_paths = {
        "quality_metrics": plots_directory / "quality_metrics.png",
        "generation_metrics": plots_directory / "generation_metrics.png",
    }
    _save_quality_plot(summary, plot_paths["quality_metrics"])
    _save_generation_plot(
        predictions,
        summary,
        plot_paths["generation_metrics"],
    )

    if predictions and "tt_cache" in predictions[0]:
        plot_paths["tt_cache_metrics"] = plots_directory / "tt_cache_metrics.png"
        _save_tt_cache_plot(predictions, plot_paths["tt_cache_metrics"])

    return {name: str(path) for name, path in plot_paths.items()}


def _load_predictions(predictions_path: Path) -> list[dict[str, Any]]:
    with predictions_path.open("r", encoding="utf-8") as predictions_file:
        predictions = [
            json.loads(line)
            for line in predictions_file
            if line.strip()
        ]
    if not predictions:
        raise ValueError(
            f"Нет результатов для построения графиков: {predictions_path}"
        )
    return predictions


def _save_quality_plot(summary: dict[str, Any], output_path: Path) -> None:
    generation = summary["generation"]
    retrieval = summary["retrieval"]
    labels = [
        "Answer EM",
        "Answer F1",
        "Retrieval recall@k",
        "Retrieval precision@k",
        "All supporting docs",
    ]
    values = [
        generation["answer_em"],
        generation["answer_f1"],
        retrieval["recall_at_k"],
        retrieval["precision_at_k"],
        retrieval["all_supporting_documents_retrieved_rate"],
    ]

    figure, axis = plt.subplots(figsize=(10, 5.5))
    bars = axis.bar(labels, values, color="#3B82A0")
    axis.set_title("Answer and retrieval quality")
    axis.set_ylabel("Score")
    axis.set_ylim(0, 1.08)
    axis.grid(axis="y", alpha=0.25)
    axis.set_axisbelow(True)
    axis.tick_params(axis="x", rotation=18)
    for bar, value in zip(bars, values, strict=True):
        axis.annotate(
            f"{value:.3f}",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            fontsize=9,
        )
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _save_generation_plot(
    predictions: list[dict[str, Any]],
    summary: dict[str, Any],
    output_path: Path,
) -> None:
    example_numbers = list(range(1, len(predictions) + 1))
    generation_seconds = [
        float(row["generation_seconds"])
        for row in predictions
    ]
    tokens_per_second = [
        float(row["generated_tokens"]) / float(row["generation_seconds"])
        if float(row["generation_seconds"]) > 0
        else 0.0
        for row in predictions
    ]

    figure, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    axes[0].plot(example_numbers, generation_seconds, color="#3B82A0")
    axes[0].axhline(
        summary["generation"]["mean_generation_seconds"],
        color="#D97706",
        linestyle="--",
        label="Mean",
    )
    axes[0].set_title("Generation latency")
    axes[0].set_ylabel("Seconds per example")
    axes[0].legend()

    axes[1].plot(example_numbers, tokens_per_second, color="#508C62")
    axes[1].axhline(
        summary["generation"]["generated_tokens_per_second"],
        color="#D97706",
        linestyle="--",
        label="Mean",
    )
    axes[1].set_title("Generation throughput")
    axes[1].set_ylabel("Generated tokens / second")
    axes[1].legend()

    peak_memory = summary["performance"]["generation_peak_cuda_memory_gb"]
    if peak_memory is None:
        axes[2].text(
            0.5,
            0.5,
            "CUDA memory unavailable",
            ha="center",
            va="center",
            transform=axes[2].transAxes,
        )
        axes[2].set_title("Peak generation memory")
        axes[2].set_axis_off()
    else:
        axes[2].bar(["Peak"], [peak_memory], color="#8A6EAF")
        axes[2].set_title("Peak generation memory")
        axes[2].set_ylabel("Allocated CUDA memory (GiB)")
        axes[2].grid(axis="y", alpha=0.25)
        axes[2].set_axisbelow(True)
        axes[2].annotate(
            f"{peak_memory:.2f} GiB",
            (0, peak_memory),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            fontsize=9,
        )

    for axis in axes[:2]:
        axis.set_xlabel("Example")
        axis.grid(alpha=0.25)
        axis.set_axisbelow(True)
    figure.suptitle("Generation performance")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _save_tt_cache_plot(
    predictions: list[dict[str, Any]],
    output_path: Path,
) -> None:
    example_numbers = list(range(1, len(predictions) + 1))
    cache_rows = [row["tt_cache"] for row in predictions]
    dense_mib = [
        float(row["dense_equivalent_bytes"]) / (1024 ** 2)
        for row in cache_rows
    ]
    stored_mib = [
        float(row["stored_bytes"]) / (1024 ** 2)
        for row in cache_rows
    ]
    compression_ratios = [
        float(row["compression_ratio"])
        for row in cache_rows
    ]
    reconstruction_errors = [
        float(row["mean_relative_reconstruction_error"])
        for row in cache_rows
    ]
    max_ranks = [int(row["max_tt_rank"]) for row in cache_rows]
    decomposition_seconds = [
        float(row["tt_decomposition_seconds"])
        for row in cache_rows
    ]
    reconstruction_seconds = [
        float(row["tt_reconstruction_seconds"])
        for row in cache_rows
    ]

    figure, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes[0, 0].plot(example_numbers, dense_mib, label="Dense equivalent")
    axes[0, 0].plot(example_numbers, stored_mib, label="Stored TT + tail")
    axes[0, 0].set_title("Cache payload size")
    axes[0, 0].set_ylabel("MiB")
    axes[0, 0].legend()

    axes[0, 1].plot(example_numbers, compression_ratios, color="#508C62")
    axes[0, 1].axhline(1.0, color="#555555", linestyle="--", linewidth=1)
    axes[0, 1].set_title("Cache compression ratio")
    axes[0, 1].set_ylabel("Dense equivalent / stored")

    axes[0, 2].plot(example_numbers, reconstruction_errors, color="#C26B55")
    axes[0, 2].set_title("Relative reconstruction error")
    axes[0, 2].set_ylabel("Relative Frobenius error")

    axes[1, 0].plot(example_numbers, max_ranks, color="#8A6EAF")
    axes[1, 0].set_title("Maximum TT rank")
    axes[1, 0].set_ylabel("Rank")

    axes[1, 1].plot(
        example_numbers,
        decomposition_seconds,
        label="Decomposition",
    )
    axes[1, 1].plot(
        example_numbers,
        reconstruction_seconds,
        label="Reconstruction",
    )
    axes[1, 1].set_title("TT processing time")
    axes[1, 1].set_ylabel("Seconds per example")
    axes[1, 1].legend()

    axes[1, 2].axis("off")
    for axis in axes.flat[:5]:
        axis.set_xlabel("Example")
        axis.grid(alpha=0.25)
        axis.set_axisbelow(True)

    figure.suptitle("TT KV-cache metrics")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
