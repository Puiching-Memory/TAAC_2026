from __future__ import annotations

from taac2026.application.reporting.cli import _benchmark_override_args, compute_pareto_frontier, parse_args


def test_compute_pareto_frontier_uses_smaller_x_and_higher_y() -> None:
    rows = [
        {"label": "large_low", "size": 4.0, "auc": 0.61},
        {"label": "tiny", "size": 1.0, "auc": 0.58},
        {"label": "mid_best", "size": 2.0, "auc": 0.64},
        {"label": "dominated", "size": 3.0, "auc": 0.60},
        {"label": "largest_best", "size": 5.0, "auc": 0.66},
    ]

    frontier = compute_pareto_frontier(rows, x_key="size", y_key="auc")

    assert [row["label"] for row in frontier] == ["tiny", "mid_best", "largest_best"]


def test_reporting_cli_forwards_max_steps_override() -> None:
    args = parse_args([
        "--dataset-path",
        "demo.parquet",
        "--max-steps",
        "17",
        "--num-workers",
        "3",
        "--device",
        "cpu",
    ])

    assert _benchmark_override_args(args) == (
        "--max_steps",
        "17",
        "--num_workers",
        "3",
        "--device",
        "cpu",
    )