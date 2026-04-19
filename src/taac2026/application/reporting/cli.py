from __future__ import annotations

import argparse
from pathlib import Path

from ...reporting.model_performance_plot import plot_model_performance


ROOT = Path(__file__).resolve().parents[4]
DEFAULT_SUMMARY_ROOT = ROOT / "outputs" / "smoke"
DEFAULT_SEARCH_ROOT = ROOT / "outputs" / "config"
DEFAULT_EXPERIMENTS_DOC_PATH = ROOT / "docs" / "experiments.md"
DEFAULT_OUTPUT_PATHS = {
    "size": ROOT / "figures" / "model_performance_vs_size.png",
    "compute": ROOT / "figures" / "model_performance_vs_compute.png",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot TAAC model performance against size or total training compute")
    parser.add_argument(
        "--summary-root",
        default=str(DEFAULT_SUMMARY_ROOT),
        help="Directory containing per-model summary.json files",
    )
    parser.add_argument(
        "--search-root",
        default=str(DEFAULT_SEARCH_ROOT),
        help="Directory containing per-model optuna study folders such as baseline_optuna",
    )
    parser.add_argument(
        "--experiments-doc",
        default=str(DEFAULT_EXPERIMENTS_DOC_PATH),
        help="Fallback markdown file used when summary.json artifacts are unavailable",
    )
    parser.add_argument(
        "--x-metric",
        choices=("size", "compute"),
        default="size",
        help="Horizontal axis metric: model size in million parameters or estimated end-to-end training TFLOPs",
    )
    parser.add_argument(
        "--output-path",
        help="PNG output path; defaults to a metric-specific file name under figures/",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_path = Path(args.output_path) if args.output_path else DEFAULT_OUTPUT_PATHS[args.x_metric]
    plot_model_performance(
        summary_root=Path(args.summary_root),
        search_root=Path(args.search_root),
        experiments_doc_path=Path(args.experiments_doc),
        output_path=output_path,
        x_metric=args.x_metric,
    )
    return 0


__all__ = ["main", "parse_args"]
