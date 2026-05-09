from __future__ import annotations

import importlib

import torch

from taac2026.application.benchmarking.pcvr_tilelang_ops_benchmark import (
    OperatorBenchmarkResult,
    _benchmark_embedding_bag_mean_backend,
    _benchmark_flash_attention_backend,
    _benchmark_rms_norm_backend,
    _summarize_runs,
    parse_args,
    run_benchmark,
)

tilelang_benchmark = importlib.import_module("taac2026.application.benchmarking.pcvr_tilelang_ops_benchmark")


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
            "torch,tilelang,triton",
            "--dtype",
            "float32",
        ]
    )

    assert args.rows == 128
    assert args.cols == 64
    assert args.operator == "rms_norm"
    assert args.steps == 2
    assert args.warmup_steps == 1
    assert args.repeats == 3
    assert args.backends == ("torch", "tilelang", "triton")
    assert args.block_rows is None
    assert args.atol == 1e-5
    assert args.rtol == 1e-5


def test_parse_pcvr_tilelang_ops_benchmark_accepts_flash_attention_operator() -> None:
    args = parse_args(
        [
            "--operator",
            "flash_attention",
            "--batch",
            "2",
            "--heads",
            "4",
            "--query-len",
            "32",
            "--kv-len",
            "64",
            "--head-dim",
            "16",
            "--dtype",
            "float16",
        ]
    )

    assert args.operator == "flash_attention"
    assert args.backends == ("torch", "tilelang")
    assert args.batch == 2
    assert args.heads == 4
    assert args.query_len == 32
    assert args.kv_len == 64
    assert args.head_dim == 16
    assert args.atol == 1e-2
    assert args.rtol == 1e-2


def test_parse_pcvr_tilelang_ops_benchmark_accepts_embedding_bag_mean_operator() -> None:
    args = parse_args(
        [
            "--operator",
            "embedding_bag_mean",
            "--batch",
            "16",
            "--embedding-vocab-size",
            "1024",
            "--embedding-dim",
            "32",
            "--embedding-bag-size",
            "3",
            "--embedding-padding-prob",
            "0.5",
            "--block-cols",
            "32",
            "--dtype",
            "float16",
        ]
    )

    assert args.operator == "embedding_bag_mean"
    assert args.batch == 16
    assert args.embedding_vocab_size == 1024
    assert args.embedding_dim == 32
    assert args.embedding_bag_size == 3
    assert args.embedding_padding_prob == 0.5
    assert args.block_cols == 32
    assert args.atol == 1e-2
    assert args.rtol == 1e-2


def test_parse_pcvr_tilelang_ops_benchmark_accepts_cuembed_backend() -> None:
    args = parse_args(
        [
            "--operator",
            "embedding_bag_mean",
            "--backends",
            "torch,cuembed",
            "--dtype",
            "float32",
        ]
    )

    assert args.backends == ("torch", "cuembed")


def test_parse_pcvr_tilelang_ops_benchmark_rejects_explicit_triton_flash_attention_backend() -> None:
    try:
        parse_args(
            [
                "--operator",
                "flash_attention",
                "--backends",
                "torch,triton",
            ]
        )
    except ValueError as error:
        assert "rms_norm" in str(error)
        assert "embedding_bag_mean" in str(error)
    else:  # pragma: no cover - defensive branch
        raise AssertionError("parse_args accepted triton for flash_attention")


def test_run_pcvr_tilelang_ops_benchmark_rejects_cuembed_for_non_embedding_operator() -> None:
    args = parse_args(
        [
            "--operator",
            "rms_norm",
            "--device",
            "cpu",
            "--backends",
            "cuembed",
            "--dtype",
            "float32",
        ]
    )

    try:
        run_benchmark(args)
    except ValueError as error:
        assert "embedding_bag_mean" in str(error)
    else:  # pragma: no cover - defensive branch
        raise AssertionError("run_benchmark accepted cuembed for rms_norm")


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
            "torch,tilelang,triton",
            "--dtype",
            "float32",
        ]
    )

    summary = run_benchmark(args)

    assert summary["operator"] == "rms_norm"
    assert summary["device"] == "cpu"
    assert summary["backends"] == ["torch", "tilelang", "triton"]
    assert len(summary["results"]) == 3

    rows = {row["backend"]: row for row in summary["results"]}
    assert rows["torch"]["status"] == "ok"
    assert rows["torch"]["resolved_backend"] == "torch"
    assert rows["torch"]["ops_per_sec"] > 0.0
    assert rows["torch"]["max_abs_error"] == 0.0
    assert rows["torch"]["max_rel_error"] == 0.0
    assert rows["tilelang"]["status"] == "unsupported"
    assert rows["tilelang"]["successful_repeats"] == 0
    assert rows["tilelang"]["error"]
    assert rows["triton"]["status"] == "unsupported"
    assert rows["triton"]["successful_repeats"] == 0
    assert rows["triton"]["error"]
    assert summary["atol"] == 1e-5
    assert summary["rtol"] == 1e-5


def test_run_flash_attention_benchmark_reports_cpu_tilelang_as_unsupported() -> None:
    args = parse_args(
        [
            "--operator",
            "flash_attention",
            "--device",
            "cpu",
            "--batch",
            "2",
            "--heads",
            "2",
            "--query-len",
            "8",
            "--kv-len",
            "8",
            "--head-dim",
            "16",
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

    assert summary["operator"] == "flash_attention"
    assert summary["device"] == "cpu"
    assert summary["backends"] == ["torch", "tilelang"]
    rows = {row["backend"]: row for row in summary["results"]}
    assert rows["torch"]["status"] == "ok"
    assert rows["torch"]["resolved_backend"] == "torch"
    assert rows["torch"]["max_abs_error"] == 0.0
    assert rows["torch"]["max_rel_error"] == 0.0
    assert rows["tilelang"]["status"] == "unsupported"
    assert rows["tilelang"]["successful_repeats"] == 0
    assert rows["tilelang"]["error"]


def test_run_embedding_bag_mean_benchmark_reports_cpu_tilelang_as_unsupported() -> None:
    args = parse_args(
        [
            "--operator",
            "embedding_bag_mean",
            "--device",
            "cpu",
            "--batch",
            "4",
            "--embedding-vocab-size",
            "32",
            "--embedding-dim",
            "8",
            "--embedding-bag-size",
            "3",
            "--steps",
            "2",
            "--warmup-steps",
            "0",
            "--repeats",
            "2",
            "--backends",
            "torch,tilelang,triton",
            "--dtype",
            "float32",
        ]
    )

    summary = run_benchmark(args)

    assert summary["operator"] == "embedding_bag_mean"
    assert summary["device"] == "cpu"
    assert summary["backends"] == ["torch", "tilelang", "triton"]
    assert summary["embedding_vocab_size"] == 32
    assert summary["embedding_dim"] == 8
    assert summary["embedding_bag_size"] == 3
    rows = {row["backend"]: row for row in summary["results"]}
    assert rows["torch"]["status"] == "ok"
    assert rows["torch"]["resolved_backend"] == "torch"
    assert rows["torch"]["max_abs_error"] == 0.0
    assert rows["torch"]["max_rel_error"] == 0.0
    assert rows["tilelang"]["status"] == "unsupported"
    assert rows["tilelang"]["successful_repeats"] == 0
    assert rows["tilelang"]["error"]
    assert rows["triton"]["status"] == "unsupported"
    assert rows["triton"]["successful_repeats"] == 0
    assert rows["triton"]["error"]


def test_run_embedding_bag_mean_benchmark_reports_cpu_cuembed_as_unsupported() -> None:
    args = parse_args(
        [
            "--operator",
            "embedding_bag_mean",
            "--device",
            "cpu",
            "--batch",
            "4",
            "--embedding-vocab-size",
            "32",
            "--embedding-dim",
            "8",
            "--embedding-bag-size",
            "3",
            "--steps",
            "2",
            "--warmup-steps",
            "0",
            "--repeats",
            "2",
            "--backends",
            "torch,cuembed",
            "--dtype",
            "float32",
        ]
    )

    summary = run_benchmark(args)

    assert summary["operator"] == "embedding_bag_mean"
    assert summary["device"] == "cpu"
    assert summary["backends"] == ["torch", "cuembed"]
    rows = {row["backend"]: row for row in summary["results"]}
    assert rows["torch"]["status"] == "ok"
    assert rows["cuembed"]["status"] == "unsupported"
    assert rows["cuembed"]["successful_repeats"] == 0
    assert rows["cuembed"]["error"]
    assert isinstance(summary["cuembed_available"], bool)


def test_benchmark_marks_tilelang_accuracy_failure_as_error(monkeypatch) -> None:
    args = parse_args(
        [
            "--device",
            "cpu",
            "--rows",
            "4",
            "--cols",
            "8",
            "--steps",
            "1",
            "--warmup-steps",
            "0",
            "--repeats",
            "1",
            "--backends",
            "tilelang",
            "--dtype",
            "float32",
            "--atol",
            "1e-6",
            "--rtol",
            "1e-6",
        ]
    )
    x = torch.ones((args.rows, args.cols), dtype=args.dtype)
    weight = torch.ones((args.cols,), dtype=args.dtype)

    monkeypatch.setattr(tilelang_benchmark, "_benchmark_callable", lambda *args, **kwargs: (0.1, 100.0))
    monkeypatch.setattr(
        tilelang_benchmark,
        "_prepare_rms_norm_callable",
        lambda **kwargs: (lambda: torch.full_like(x, 2.0), "tilelang", 0.25),
    )

    result = _benchmark_rms_norm_backend(
        backend="tilelang",
        args=args,
        device=torch.device("cpu"),
        x=x,
        weight=weight,
    )

    assert result.status == "error"
    assert result.error is not None
    assert "close" in result.error.lower()
    assert result.max_abs_error is None
    assert result.max_rel_error is None


def test_benchmark_marks_triton_accuracy_failure_as_error(monkeypatch) -> None:
    args = parse_args(
        [
            "--device",
            "cpu",
            "--rows",
            "4",
            "--cols",
            "8",
            "--steps",
            "1",
            "--warmup-steps",
            "0",
            "--repeats",
            "1",
            "--backends",
            "triton",
            "--dtype",
            "float32",
            "--atol",
            "1e-6",
            "--rtol",
            "1e-6",
        ]
    )
    x = torch.ones((args.rows, args.cols), dtype=args.dtype)
    weight = torch.ones((args.cols,), dtype=args.dtype)

    monkeypatch.setattr(tilelang_benchmark, "_benchmark_callable", lambda *args, **kwargs: (0.1, 100.0))
    monkeypatch.setattr(
        tilelang_benchmark,
        "_prepare_rms_norm_callable",
        lambda **kwargs: (lambda: torch.full_like(x, 2.0), "triton", 0.25),
    )

    result = _benchmark_rms_norm_backend(
        backend="triton",
        args=args,
        device=torch.device("cpu"),
        x=x,
        weight=weight,
    )

    assert result.status == "error"
    assert result.error is not None
    assert "close" in result.error.lower()


def test_flash_attention_benchmark_marks_tilelang_accuracy_failure_as_error(monkeypatch) -> None:
    args = parse_args(
        [
            "--operator",
            "flash_attention",
            "--device",
            "cpu",
            "--batch",
            "1",
            "--heads",
            "2",
            "--query-len",
            "4",
            "--kv-len",
            "4",
            "--head-dim",
            "8",
            "--steps",
            "1",
            "--warmup-steps",
            "0",
            "--repeats",
            "1",
            "--backends",
            "tilelang",
            "--dtype",
            "float32",
            "--atol",
            "1e-6",
            "--rtol",
            "1e-6",
        ]
    )
    q = torch.ones((args.batch, args.heads, args.query_len, args.head_dim), dtype=args.dtype)
    k = torch.ones((args.batch, args.heads, args.kv_len, args.head_dim), dtype=args.dtype)
    v = torch.ones((args.batch, args.heads, args.kv_len, args.head_dim), dtype=args.dtype)

    monkeypatch.setattr(tilelang_benchmark, "_benchmark_callable", lambda *args, **kwargs: (0.1, 100.0))
    monkeypatch.setattr(
        tilelang_benchmark,
        "_prepare_flash_attention_callable",
        lambda **kwargs: (lambda: torch.full_like(q, 2.0), "tilelang", 0.25),
    )

    result = _benchmark_flash_attention_backend(
        backend="tilelang",
        args=args,
        device=torch.device("cpu"),
        q=q,
        k=k,
        v=v,
    )

    assert result.status == "error"
    assert result.error is not None
    assert "close" in result.error.lower()
    assert result.max_abs_error is None
    assert result.max_rel_error is None


def test_embedding_bag_mean_benchmark_marks_tilelang_accuracy_failure_as_error(monkeypatch) -> None:
    args = parse_args(
        [
            "--operator",
            "embedding_bag_mean",
            "--device",
            "cpu",
            "--batch",
            "2",
            "--embedding-vocab-size",
            "8",
            "--embedding-dim",
            "4",
            "--embedding-bag-size",
            "3",
            "--steps",
            "1",
            "--warmup-steps",
            "0",
            "--repeats",
            "1",
            "--backends",
            "tilelang",
            "--dtype",
            "float32",
            "--atol",
            "1e-6",
            "--rtol",
            "1e-6",
        ]
    )
    embedding_weight = torch.ones((args.embedding_vocab_size + 1, args.embedding_dim), dtype=args.dtype)
    values = torch.ones((args.batch, args.embedding_bag_size), dtype=torch.long)

    monkeypatch.setattr(tilelang_benchmark, "_benchmark_callable", lambda *args, **kwargs: (0.1, 100.0))
    monkeypatch.setattr(
        tilelang_benchmark,
        "_prepare_embedding_bag_mean_callable",
        lambda **kwargs: (lambda: torch.full((args.batch, args.embedding_dim), 2.0), "tilelang", 0.25),
    )

    result = _benchmark_embedding_bag_mean_backend(
        backend="tilelang",
        args=args,
        device=torch.device("cpu"),
        embedding_weight=embedding_weight,
        values=values,
    )

    assert result.status == "error"
    assert result.error is not None
    assert "close" in result.error.lower()
    assert result.max_abs_error is None
    assert result.max_rel_error is None


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
    assert summary.max_rel_error is None
