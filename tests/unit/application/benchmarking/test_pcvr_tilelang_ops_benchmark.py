from __future__ import annotations

from taac2026.application.benchmarking.pcvr_tilelang_ops_benchmark import (
    OperatorBenchmarkResult,
    _summarize_runs,
    parse_args,
    run_benchmark,
)


def test_parse_pcvr_tilelang_ops_benchmark_args_accepts_backend_subset() -> None:
    args = parse_args(
        [
            "--rows",
            "128",
            "--cols",
            "64",
            "--steps",
            "2",
            "--warmup-steps",
            "1",
            "--repeats",
            "3",
            "--backends",
            "torch,tilelang",
            "--dtype",
            "float32",
        ]
    )

    assert args.rows == 128
    assert args.cols == 64
    assert args.steps == 2
    assert args.warmup_steps == 1
    assert args.repeats == 3
    assert args.backends == ("torch", "tilelang")


def test_run_pcvr_tilelang_ops_benchmark_reports_cpu_tilelang_as_unsupported() -> None:
    args = parse_args(
        [
            "--device",
            "cpu",
            "--rows",
            "64",
            "--cols",
            "32",
            "--steps",
            "2",
            "--warmup-steps",
            "0",
            "--repeats",
            "2",
            "--backends",
            "torch,tilelang",
            "--dtype",
            "float32",
        ]
    )

    summary = run_benchmark(args)

    assert summary["operator"] == "rms_norm"
    assert summary["device"] == "cpu"
    assert summary["backends"] == ["torch", "tilelang"]
    assert len(summary["results"]) == 2

    rows = {row["backend"]: row for row in summary["results"]}
    assert rows["torch"]["status"] == "ok"
    assert rows["torch"]["resolved_backend"] == "torch"
    assert rows["torch"]["ops_per_sec"] > 0.0
    assert rows["tilelang"]["status"] == "unsupported"
    assert rows["tilelang"]["successful_repeats"] == 0
    assert rows["tilelang"]["error"]


def test_summarize_runs_keeps_first_successful_compile_time() -> None:
    summary = _summarize_runs(
        backend="tilelang",
        repeats=3,
        runs=[
            OperatorBenchmarkResult(
                backend="tilelang",
                status="ok",
                resolved_backend="tilelang",
                elapsed_sec=0.3,
                step_time_ms=1.5,
                ops_per_sec=10.0,
                measured_steps=10,
                warmup_steps=2,
                compile_sec=0.9,
                max_abs_error=0.0,
            ),
            OperatorBenchmarkResult(
                backend="tilelang",
                status="ok",
                resolved_backend="tilelang",
                elapsed_sec=0.2,
                step_time_ms=1.0,
                ops_per_sec=12.0,
                measured_steps=10,
                warmup_steps=2,
                compile_sec=0.1,
                max_abs_error=0.0,
            ),
            OperatorBenchmarkResult(
                backend="tilelang",
                status="ok",
                resolved_backend="tilelang",
                elapsed_sec=0.1,
                step_time_ms=0.5,
                ops_per_sec=14.0,
                measured_steps=10,
                warmup_steps=2,
                compile_sec=0.05,
                max_abs_error=0.0,
            ),
        ],
    )

    assert summary.compile_sec == 0.9