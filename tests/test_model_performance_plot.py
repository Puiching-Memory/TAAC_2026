from __future__ import annotations

import json
from pathlib import Path

import pytest

from taac2026.application.reporting import cli as reporting_cli
from taac2026.reporting.model_performance_plot import (
    DISPLAY_NAMES,
    load_base_points,
    load_search_series,
    merge_best_search_points,
    plot_model_performance,
)

pytestmark = pytest.mark.unit


def _summary_payload(*, auc: float, parameter_size_mb: float, total_parameters: int, total_tflops: float) -> dict:
    return {
        "metrics": {"auc": auc},
        "model_profile": {
            "parameter_size_mb": parameter_size_mb,
            "total_parameters": total_parameters,
        },
        "compute_profile": {
            "estimated_end_to_end_tflops_total": total_tflops,
        },
    }


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_load_base_points_falls_back_to_experiments_doc(tmp_path: Path) -> None:
    summary_root = tmp_path / "outputs" / "smoke"
    for index, slug in enumerate(DISPLAY_NAMES):
        if slug == "baseline":
            continue
        _write_json(
            summary_root / slug / "summary.json",
            _summary_payload(
                auc=0.700 + index * 0.001,
                parameter_size_mb=16.0 + index,
                total_parameters=4_000_000 + index * 100_000,
                total_tflops=1.0 + index * 0.1,
            ),
        )

    experiments_doc = tmp_path / "docs" / "experiments.md"
    experiments_doc.parent.mkdir(parents=True, exist_ok=True)
    experiments_doc.write_text(
        "\n".join(
            [
                "# Experiments",
                "",
                "## 当前可复核 smoke 结果",
                "",
                "| 实验包 | 目录 | 模型 | AUC | PR AUC | Brier | Logloss | 延迟 | 约束 | TFLOPs | 模型大小(MB) |",
                "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
                "| Baseline | config/gen/baseline | baseline | 0.812 | 0.000 | 0.000 | 0.000 | 0.000 | yes | 2.5 | 24.0 |",
            ]
        ),
        encoding="utf-8",
    )

    points = load_base_points(summary_root, experiments_doc)

    baseline = next(point for point in points if point.slug == "baseline")
    assert baseline.source == "docs"
    assert baseline.auc == 0.812
    assert baseline.total_tflops == 2.5


def test_plot_model_performance_merges_search_trials_and_writes_outputs(tmp_path: Path) -> None:
    summary_root = tmp_path / "outputs" / "smoke"
    for index, slug in enumerate(DISPLAY_NAMES):
        _write_json(
            summary_root / slug / "summary.json",
            _summary_payload(
                auc=0.710 + index * 0.005,
                parameter_size_mb=12.0 + index,
                total_parameters=3_000_000 + index * 250_000,
                total_tflops=1.0 + index * 0.2,
            ),
        )

    search_root = tmp_path / "outputs" / "gen"
    _write_json(
        search_root / "baseline_optuna" / "trial_0000" / "summary.json",
        _summary_payload(
            auc=0.905,
            parameter_size_mb=20.0,
            total_parameters=5_000_000,
            total_tflops=3.5,
        ),
    )

    experiments_doc = tmp_path / "docs" / "experiments.md"
    base_points = load_base_points(summary_root, experiments_doc)
    search_series = load_search_series(search_root)
    merged_points = merge_best_search_points(base_points, search_series)

    baseline = next(point for point in merged_points if point.slug == "baseline")
    assert baseline.source == "search"
    assert baseline.auc == 0.905

    output_path = tmp_path / "figures" / "model_performance_vs_compute.png"
    plot_model_performance(
        summary_root=summary_root,
        search_root=search_root,
        experiments_doc_path=experiments_doc,
        output_path=output_path,
        x_metric="compute",
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0
    assert output_path.with_suffix(".svg").exists()


def test_reporting_cli_main_uses_metric_specific_defaults(tmp_path: Path, monkeypatch) -> None:
    summary_root = tmp_path / "outputs" / "smoke"
    search_root = tmp_path / "outputs" / "gen"
    experiments_doc = tmp_path / "docs" / "experiments.md"
    output_path = tmp_path / "figures" / "model_performance_vs_compute.png"
    captured: dict[str, object] = {}

    def fake_plot_model_performance(*, summary_root, search_root, experiments_doc_path, output_path, x_metric) -> None:
        captured["summary_root"] = summary_root
        captured["search_root"] = search_root
        captured["experiments_doc_path"] = experiments_doc_path
        captured["output_path"] = output_path
        captured["x_metric"] = x_metric

    monkeypatch.setattr(reporting_cli, "plot_model_performance", fake_plot_model_performance)
    monkeypatch.setattr(reporting_cli, "DEFAULT_SUMMARY_ROOT", summary_root)
    monkeypatch.setattr(reporting_cli, "DEFAULT_SEARCH_ROOT", search_root)
    monkeypatch.setattr(reporting_cli, "DEFAULT_EXPERIMENTS_DOC_PATH", experiments_doc)
    monkeypatch.setitem(reporting_cli.DEFAULT_OUTPUT_PATHS, "compute", output_path)

    exit_code = reporting_cli.main(["--x-metric", "compute"])

    assert exit_code == 0
    assert captured == {
        "summary_root": summary_root,
        "search_root": search_root,
        "experiments_doc_path": experiments_doc,
        "output_path": output_path,
        "x_metric": "compute",
    }
