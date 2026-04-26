from __future__ import annotations

from pathlib import Path

from taac2026.application.reporting.cli import compute_pareto_frontier, discover_experiment_paths


def test_discover_experiment_paths_filters_to_valid_packages(tmp_path: Path) -> None:
    config_root = tmp_path / "config"
    config_root.mkdir()

    valid = config_root / "valid_exp"
    valid.mkdir()
    for name in ("__init__.py", "model.py", "ns_groups.json"):
        (valid / name).write_text("", encoding="utf-8")

    missing_model = config_root / "missing_model"
    missing_model.mkdir()
    (missing_model / "__init__.py").write_text("", encoding="utf-8")
    (missing_model / "ns_groups.json").write_text("{}", encoding="utf-8")

    hidden = config_root / "__pycache__"
    hidden.mkdir()
    for name in ("__init__.py", "model.py", "ns_groups.json"):
        (hidden / name).write_text("", encoding="utf-8")

    assert discover_experiment_paths(config_root) == ["config/valid_exp"]


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