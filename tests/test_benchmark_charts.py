from __future__ import annotations

import json
from pathlib import Path

import pytest

from taac2026.application.reporting.bench_cli import main, parse_args
from taac2026.reporting.benchmark_charts import (
    build_benchmark_acceptance_summary,
    load_pytest_benchmark_file,
    write_benchmark_charts,
)


def _write_benchmark_payload(path: Path) -> None:
    payload = {
        "commit_info": {"id": "phase0-baseline"},
        "benchmarks": [
            {
                "name": "embedding_lookup",
                "stats": {"median": 0.0015, "mean": 0.0017, "iqr": 0.0002, "ops": 2048.0},
                "extra_info": {
                    "component": "embedding",
                    "phase": "baseline",
                    "label": "phase-0",
                    "throughput": 4096.0,
                },
            },
        ],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def test_parse_args_accepts_benchmark_inputs() -> None:
    args = parse_args([
        "--input",
        "benchmark-result.json",
        "other.json",
        "--performance-dir",
        "outputs/performance",
        "--output-dir",
        "docs/assets/figures/benchmarks",
        "--summary-path",
        "docs/assets/figures/benchmarks/benchmark_acceptance.json",
        "--label",
        "phase-0",
        "--baseline-phase",
        "baseline",
        "--candidate-phase",
        "phase-1",
    ])

    assert args.input == ["benchmark-result.json", "other.json"]
    assert args.performance_dir == "outputs/performance"
    assert args.output_dir == "docs/assets/figures/benchmarks"
    assert args.summary_path == "docs/assets/figures/benchmarks/benchmark_acceptance.json"
    assert args.label == "phase-0"
    assert args.baseline_phase == "baseline"
    assert args.candidate_phase == "phase-1"


def test_load_pytest_benchmark_file_parses_stats(tmp_path: Path) -> None:
    benchmark_json = tmp_path / "benchmark-result.json"
    _write_benchmark_payload(benchmark_json)

    records = load_pytest_benchmark_file(benchmark_json)

    assert len(records) == 1
    assert records[0]["component"] == "embedding"
    assert records[0]["phase"] == "baseline"
    assert records[0]["label"] == "phase-0"
    assert records[0]["median_ms"] == pytest.approx(1.5)
    assert records[0]["throughput"] == pytest.approx(4096.0)


def test_write_benchmark_charts_creates_expected_files(tmp_path: Path) -> None:
    records = [
        {"name": "embedding_lookup", "component": "embedding", "phase": "baseline", "label": "phase-0", "median_ms": 1.5, "throughput": 4096.0},
        {"name": "attention_forward", "component": "attention", "phase": "baseline", "label": "phase-0", "median_ms": 2.5},
        {"name": "ffn_forward", "component": "ffn", "phase": "baseline", "label": "phase-0", "median_ms": 1.9},
        {"name": "rmsnorm", "component": "rmsnorm", "phase": "baseline", "label": "phase-0", "median_ms": 0.4},
        {"name": "collate", "component": "collate", "phase": "baseline", "label": "phase-0", "median_ms": 3.1},
        {"name": "e2e_train_step", "component": "e2e_train_step", "phase": "baseline", "label": "phase-0", "median_ms": 12.5},
        {"name": "inference_latency", "component": "inference", "phase": "baseline", "label": "phase-0", "model": "tiny_baseline", "median_ms": 4.2, "times_ms": [3.8, 4.0, 4.2, 4.5, 4.9]},
        {"name": "quantized_inference", "component": "quantization", "phase": "baseline", "label": "phase-0", "model": "tiny_baseline/int8", "median_ms": 2.1, "memory_mb": 128.0},
    ]

    written = write_benchmark_charts(output_dir=tmp_path, records=records)
    names = sorted(path.name for path in written)

    assert names == [
        "component_latency.echarts.json",
        "e2e_train_step.echarts.json",
        "inference_boxplot.echarts.json",
        "quantization_comparison.echarts.json",
        "throughput_trend.echarts.json",
    ]
    component_chart = json.loads((tmp_path / "component_latency.echarts.json").read_text(encoding="utf-8"))
    assert component_chart["series"]
    assert component_chart["xAxis"]["data"] == ["collate", "embedding", "attention", "ffn", "rmsnorm"]


def test_build_benchmark_acceptance_summary_flags_missing_candidate_data() -> None:
    summary = build_benchmark_acceptance_summary([
        {"name": "embedding_lookup", "component": "embedding", "phase": "baseline", "label": "phase-0", "median_ms": 1.5, "throughput": 4096.0},
        {"name": "attention_forward", "component": "attention", "phase": "baseline", "label": "phase-0", "median_ms": 2.5},
    ])

    assert summary["baseline_phase"] == "baseline"
    assert summary["candidate_phase"] is None
    assert summary["acceptance"]["embedding_throughput_vs_baseline"]["status"] == "not_enough_data"
    assert summary["acceptance"]["attention_latency_vs_baseline"]["status"] == "not_enough_data"


def test_build_benchmark_acceptance_summary_compares_candidate_phase() -> None:
    summary = build_benchmark_acceptance_summary(
        [
            {"name": "embedding_lookup", "component": "embedding", "phase": "baseline", "label": "phase-0", "median_ms": 1.5, "throughput": 4096.0},
            {"name": "embedding_lookup", "component": "embedding", "phase": "optimized", "label": "phase-1", "median_ms": 0.7, "throughput": 9216.0},
            {"name": "attention_forward", "component": "attention", "phase": "baseline", "label": "phase-0", "median_ms": 2.5},
            {"name": "attention_forward", "component": "attention", "phase": "optimized", "label": "phase-1", "median_ms": 1.4},
            {"name": "quantized_inference", "component": "quantization", "phase": "optimized", "label": "phase-1", "model": "tiny/int8", "median_ms": 2.1, "memory_mb": 128.0},
        ],
        baseline_phase="baseline",
        candidate_phase="optimized",
    )

    assert summary["candidate_phase"] == "optimized"
    assert summary["candidate_phases"] == {"embedding": "optimized", "attention": "optimized"}
    assert summary["acceptance"]["embedding_throughput_vs_baseline"]["status"] == "pass"
    assert summary["acceptance"]["attention_latency_vs_baseline"]["status"] == "pass"
    assert summary["acceptance"]["int8_quantization_record_present"]["status"] == "pass"


def test_build_benchmark_acceptance_summary_resolves_candidate_phase_per_component() -> None:
    summary = build_benchmark_acceptance_summary(
        [
            {"name": "embedding_lookup", "component": "embedding", "phase": "baseline", "label": "phase-0", "median_ms": 1.5, "throughput": 4096.0},
            {"name": "embedding_lookup_torchrec", "component": "embedding", "phase": "phase-2", "label": "phase-2", "median_ms": 0.6, "throughput": 10240.0},
            {"name": "attention_forward", "component": "attention", "phase": "baseline", "label": "phase-0", "median_ms": 2.5},
            {"name": "attention_forward_triton", "component": "attention", "phase": "phase-3", "label": "phase-3", "median_ms": 1.4},
            {"name": "quantized_inference", "component": "quantization", "phase": "phase-6", "label": "phase-6", "model": "tiny/int8", "median_ms": 2.1, "memory_mb": 128.0},
        ],
        baseline_phase="baseline",
    )

    assert summary["candidate_phase"] is None
    assert summary["candidate_phases"] == {"embedding": "phase-2", "attention": "phase-3"}
    assert summary["acceptance"]["embedding_throughput_vs_baseline"]["candidate_phase"] == "phase-2"
    assert summary["acceptance"]["embedding_throughput_vs_baseline"]["status"] == "pass"
    assert summary["acceptance"]["attention_latency_vs_baseline"]["candidate_phase"] == "phase-3"
    assert summary["acceptance"]["attention_latency_vs_baseline"]["status"] == "pass"
    assert summary["acceptance"]["int8_quantization_record_present"]["status"] == "pass"


def test_bench_cli_writes_placeholder_charts_without_input(tmp_path: Path) -> None:
    performance_dir = tmp_path / "performance"
    output_dir = tmp_path / "charts"
    summary_path = tmp_path / "charts" / "benchmark_acceptance.json"
    performance_dir.mkdir(parents=True)

    exit_code = main([
        "--performance-dir",
        str(performance_dir),
        "--output-dir",
        str(output_dir),
        "--summary-path",
        str(summary_path),
    ])

    assert exit_code == 0
    assert (output_dir / "component_latency.echarts.json").exists()
    assert summary_path.exists()
    placeholder = json.loads((output_dir / "component_latency.echarts.json").read_text(encoding="utf-8"))
    assert placeholder["series"] == []
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["acceptance"]["int8_quantization_record_present"]["status"] == "missing"
