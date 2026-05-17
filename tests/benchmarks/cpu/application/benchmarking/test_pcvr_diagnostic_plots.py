from __future__ import annotations

from pathlib import Path

import pytest

import taac2026.application.benchmarking.pcvr_diagnostic_plots as diagnostic_plots
from taac2026.application.benchmarking.pcvr_diagnostic_plots import (
    DiagnosticInputError,
    FIGURE_NAMES,
    format_summary,
    parse_args,
    run_diagnostics,
)
from taac2026.infrastructure.io.json import dumps, loads


def _write_run(run_dir: Path, *, experiment: str, scores: list[float], seed: int) -> None:
    run_dir.mkdir(parents=True)
    predictions_path = run_dir / "validation_predictions.jsonl"
    lines = []
    for index, score in enumerate(scores):
        lines.append(
            dumps(
                {
                    "sample_index": index,
                    "user_id": f"u{index}",
                    "score": score,
                    "target": float(index % 2),
                    "timestamp": None,
                }
            )
        )
    predictions_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (run_dir / "evaluation.json").write_text(
        dumps(
            {
                "experiment_name": experiment,
                "validation_predictions_path": str(predictions_path),
                "metrics": {
                    "auc": 0.6 + seed * 0.01,
                    "logloss": 0.7 - seed * 0.01,
                    "score_diagnostics": {"score_std": 0.1 + seed * 0.01},
                },
                "telemetry": {"elapsed_sec": 0.4 + seed * 0.1, "rows": len(scores), "rows_per_sec": 10.0},
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "training_telemetry.json").write_text(
        dumps({"elapsed_sec": 1.0 + seed, "cpu_peak_rss_mb": 100.0 + seed, "cuda_peak_allocated_mb": 0.0}),
        encoding="utf-8",
    )
    (run_dir / "evaluation_telemetry.json").write_text(
        dumps({"elapsed_sec": 0.4 + seed * 0.1, "rows": len(scores), "rows_per_sec": 10.0, "cpu_peak_rss_mb": 110.0}),
        encoding="utf-8",
    )
    (run_dir / "inference_telemetry.json").write_text(
        dumps({"elapsed_sec": 0.3 + seed * 0.1, "rows": len(scores), "rows_per_sec": 12.0, "cpu_peak_rss_mb": 108.0}),
        encoding="utf-8",
    )


def test_run_diagnostics_generates_all_svg_figures(tmp_path: Path) -> None:
    run_a = tmp_path / "baseline_seed1"
    run_b = tmp_path / "tokenformer_seed1"
    _write_run(run_a, experiment="pcvr_baseline", scores=[0.1, 0.8, 0.2, 0.7], seed=1)
    _write_run(run_b, experiment="pcvr_tokenformer", scores=[0.2, 0.7, 0.35, 0.9], seed=2)
    output_dir = tmp_path / "figures"

    summary = run_diagnostics(
        parse_args(
            [
                "--run",
                f"baseline={run_a}",
                "--run",
                f"tokenformer={run_b}",
                "--output-dir",
                str(output_dir),
                "--top-disagreement",
                "2",
            ]
        )
    )

    assert set(summary["figures"]) == set(FIGURE_NAMES)
    for path_value in summary["figures"].values():
        path = Path(path_value)
        assert path.exists()
        assert path.read_text(encoding="utf-8").lstrip().startswith("<?xml")
    summary_path = Path(summary["summary_path"])
    assert summary_path.exists()
    saved_summary = loads(summary_path.read_bytes())
    assert saved_summary["runs"][0]["prediction_count"] == 4
    runtime_svg = Path(summary["figures"]["runtime_resources"]).read_text(encoding="utf-8")
    stability_svg = Path(summary["figures"]["stability"]).read_text(encoding="utf-8")
    disagreement_svg = Path(summary["figures"]["sample_disagreement"]).read_text(encoding="utf-8")
    assert "Runtime / Resource Scorecard" in runtime_svg
    assert "Training Cost Tradeoff" in runtime_svg
    assert "Inference Efficiency Tradeoff" in runtime_svg
    assert "brighter is better" in runtime_svg
    assert "Single-Run Smoke Metrics" in stability_svg
    assert "Top 2 Samples: Prediction By Model" in disagreement_svg


def test_run_diagnostics_accepts_cwd_relative_prediction_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_dir = tmp_path / "outputs" / "smoke" / "baseline_seed1"
    _write_run(run_dir, experiment="pcvr_baseline", scores=[0.1, 0.8, 0.2, 0.7], seed=1)
    evaluation_path = run_dir / "evaluation.json"
    evaluation = loads(evaluation_path.read_bytes())
    evaluation["validation_predictions_path"] = str((run_dir / "validation_predictions.jsonl").relative_to(tmp_path))
    evaluation_path.write_text(dumps(evaluation), encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    summary = run_diagnostics(parse_args(["--run", f"baseline={run_dir}", "--output-dir", str(tmp_path / "figures")]))

    assert summary["runs"][0]["prediction_count"] == 4


def test_run_diagnostics_rejects_missing_validation_outputs(tmp_path: Path) -> None:
    run_dir = tmp_path / "baseline_seed42"
    run_dir.mkdir()

    with pytest.raises(DiagnosticInputError, match=r"validation_predictions\.jsonl"):
        run_diagnostics(parse_args(["--run", f"baseline={run_dir}", "--output-dir", str(tmp_path / "figures")]))

    assert not (tmp_path / "figures").exists()


def test_run_diagnostics_allow_partial_writes_placeholder_figures(tmp_path: Path) -> None:
    run_dir = tmp_path / "baseline_seed42"
    run_dir.mkdir()
    output_dir = tmp_path / "figures"

    summary = run_diagnostics(
        parse_args(
            [
                "--run",
                f"baseline={run_dir}",
                "--output-dir",
                str(output_dir),
                "--allow-partial",
            ]
        )
    )

    assert summary["runs"][0]["missing_required_inputs"]
    assert all(Path(path).exists() for path in summary["figures"].values())


def test_format_summary_is_human_readable(tmp_path: Path) -> None:
    run_dir = tmp_path / "baseline_seed1"
    _write_run(run_dir, experiment="pcvr_baseline", scores=[0.1, 0.8, 0.2, 0.7], seed=1)
    summary = run_diagnostics(parse_args(["--run", f"baseline={run_dir}", "--output-dir", str(tmp_path / "figures")]))

    report = format_summary(summary)

    assert report.startswith("PCVR diagnostics written to:")
    assert "Figures:" in report
    assert "baseline [ok]" in report
    assert not report.lstrip().startswith("{")


def test_main_prints_report_and_returns_error_for_missing_inputs(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    missing_dir = tmp_path / "missing"
    missing_dir.mkdir()

    exit_code = diagnostic_plots.main(["--run", f"baseline={missing_dir}", "--output-dir", str(tmp_path / "figures")])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "PCVR diagnostic inputs are incomplete" in captured.err
    assert captured.out == ""
