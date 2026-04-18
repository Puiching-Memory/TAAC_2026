from __future__ import annotations

import argparse
from pathlib import Path

from taac2026.infrastructure.io.console import configure_logging, logger
from taac2026.reporting.benchmark_charts import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_PERFORMANCE_DIR,
    DEFAULT_SUMMARY_PATH,
    load_benchmark_records,
    write_benchmark_charts,
    write_benchmark_summary,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate ECharts benchmark reports")
    parser.add_argument(
        "--input",
        nargs="*",
        default=[],
        help="Optional pytest-benchmark JSON file(s)",
    )
    parser.add_argument(
        "--performance-dir",
        default=str(DEFAULT_PERFORMANCE_DIR),
        help="Directory containing custom benchmark JSON payloads",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Output directory for benchmark .echarts.json files",
    )
    parser.add_argument(
        "--summary-path",
        default=str(DEFAULT_SUMMARY_PATH),
        help="Output path for the benchmark acceptance summary JSON",
    )
    parser.add_argument(
        "--label",
        default="",
        help="Fallback run label used when the benchmark JSON has no explicit phase metadata",
    )
    parser.add_argument(
        "--baseline-phase",
        default="baseline",
        help="Phase name treated as the benchmark baseline in the acceptance summary",
    )
    parser.add_argument(
        "--candidate-phase",
        default=None,
        help="Optional phase name compared against the baseline in the acceptance summary",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    configure_logging()
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    records = load_benchmark_records(
        args.input,
        performance_dir=args.performance_dir,
        default_label=args.label,
    )
    written = write_benchmark_charts(output_dir=output_dir, records=records)
    summary_path = write_benchmark_summary(
        args.summary_path,
        records=records,
        baseline_phase=args.baseline_phase,
        candidate_phase=args.candidate_phase,
    )
    logger.info("Written {} benchmark charts → {}", len(written), output_dir)
    logger.info("Written benchmark acceptance summary → {}", summary_path)
    return 0


__all__ = ["main", "parse_args"]