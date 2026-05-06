"""Benchmark TileLang-backed PCVR operators on synthetic tensors."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import os
import statistics
import time
from collections.abc import Callable, Sequence
from typing import Any, Literal

import torch
import torch.nn.functional as F

from taac2026.infrastructure.io.json import dumps
from taac2026.infrastructure.io.streams import write_stdout_line
from taac2026.infrastructure.accelerators import (
    clear_embedding_bag_mean_kernel_cache,
    clear_flash_attention_kernel_cache,
    clear_tilelang_kernel_cache,
    compile_embedding_bag_mean_kernel,
    compile_flash_attention_kernel,
    compile_rms_norm_kernel,
    resolved_embedding_bag_mean_backend,
    resolved_flash_attention_backend,
    resolved_rms_norm_backend,
    tilelang_available,
)


BenchmarkBackend = Literal["torch", "tilelang"]
DEFAULT_BACKENDS: tuple[BenchmarkBackend, ...] = ("torch", "tilelang")


@dataclass(slots=True)
class OperatorBenchmarkResult:
    backend: str
    status: str
    resolved_backend: str | None
    elapsed_sec: float
    step_time_ms: float
    ops_per_sec: float
    measured_steps: int
    warmup_steps: int
    compile_sec: float | None
    max_abs_error: float | None
    max_rel_error: float | None = None
    error: str | None = None


@dataclass(slots=True)
class OperatorBenchmarkSummary:
    backend: str
    status: str
    resolved_backend: str | None
    elapsed_sec: float
    elapsed_sec_mean: float
    elapsed_sec_median: float
    elapsed_sec_p95: float
    step_time_ms: float
    step_time_ms_mean: float
    step_time_ms_median: float
    step_time_ms_p95: float
    ops_per_sec: float
    ops_per_sec_mean: float
    ops_per_sec_median: float
    ops_per_sec_p95: float
    measured_steps: int
    warmup_steps: int
    repeats: int
    compile_sec: float | None
    max_abs_error: float | None
    successful_repeats: int
    runs: list[dict[str, Any]]
    max_rel_error: float | None = None
    error: str | None = None


def _parse_backend_names(value: str) -> tuple[BenchmarkBackend, ...]:
    names = tuple(name.strip().lower() for name in value.split(",") if name.strip())
    if not names:
        raise ValueError("at least one backend must be specified")
    invalid = [name for name in names if name not in DEFAULT_BACKENDS]
    if invalid:
        raise ValueError(f"unsupported backends: {', '.join(invalid)}")
    return names  # type: ignore[return-value]


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * percentile
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    if lower == upper:
        return ordered[lower]
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _reference_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps) * weight


def _reference_flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    is_causal: bool,
) -> torch.Tensor:
    return F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=is_causal)


def _reference_embedding_bag_mean(embedding_weight: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    embedded = F.embedding(values, embedding_weight, padding_idx=0)
    valid = values.ne(0).to(embedded.dtype).unsqueeze(-1)
    return (embedded * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)


def _default_error_tolerances(dtype: torch.dtype) -> tuple[float, float]:
    if dtype == torch.float32:
        return 1e-5, 1e-5
    if dtype == torch.float16:
        return 1e-2, 1e-2
    if dtype == torch.bfloat16:
        return 2e-2, 2e-2
    raise ValueError(f"unsupported benchmark dtype for error tolerances: {dtype}")


def _comparison_tensors(output: torch.Tensor, reference: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    compare_dtype = torch.float32 if output.dtype in {torch.float16, torch.bfloat16} else output.dtype
    return output.to(compare_dtype), reference.to(compare_dtype)


def _compute_error_metrics(output: torch.Tensor, reference: torch.Tensor) -> tuple[float, float]:
    compare_output, compare_reference = _comparison_tensors(output, reference)
    abs_diff = (compare_output - compare_reference).abs()
    max_abs_error = float(abs_diff.max().item())
    denom = compare_reference.abs().clamp_min(torch.finfo(compare_reference.dtype).eps)
    max_rel_error = float((abs_diff / denom).max().item())
    return max_abs_error, max_rel_error


def _validate_output_accuracy(
    output: torch.Tensor,
    reference: torch.Tensor,
    *,
    atol: float,
    rtol: float,
) -> tuple[float, float]:
    compare_output, compare_reference = _comparison_tensors(output, reference)
    torch.testing.assert_close(compare_output, compare_reference, atol=atol, rtol=rtol)
    return _compute_error_metrics(compare_output, compare_reference)


def _benchmark_failure_status(backend: BenchmarkBackend, error: Exception) -> str:
    return "unsupported" if backend == "tilelang" and not isinstance(error, AssertionError) else "error"


def _benchmark_callable(
    function: Callable[[], torch.Tensor],
    *,
    device: torch.device,
    warmup_steps: int,
    steps: int,
) -> tuple[float, float]:
    if device.type == "cuda" and torch.cuda.is_available():
        try:
            from tilelang.profiler.bench import do_bench as tilelang_do_bench
        except ImportError:
            tilelang_do_bench = None
        if tilelang_do_bench is not None:
            step_time_ms = float(
                tilelang_do_bench(
                    function,
                    _n_warmup=warmup_steps,
                    _n_repeat=steps,
                    return_mode="mean",
                )
            )
            return (step_time_ms * steps) / 1000.0, step_time_ms

    for _ in range(warmup_steps):
        function()
    _synchronize(device)

    started = time.perf_counter()
    for _ in range(steps):
        function()
    _synchronize(device)
    elapsed = time.perf_counter() - started
    step_time_ms = (elapsed / steps) * 1000.0 if steps > 0 else 0.0
    return elapsed, step_time_ms


def _prepare_rms_norm_callable(
    *,
    backend: BenchmarkBackend,
    args: argparse.Namespace,
    device: torch.device,
    x: torch.Tensor,
    weight: torch.Tensor,
) -> tuple[Callable[[], torch.Tensor], str, float | None]:
    if backend == "torch":
        return lambda: _reference_rms_norm(x, weight, args.eps), "torch", None

    resolved_backend = resolved_rms_norm_backend(
        x,
        backend,
        eps=args.eps,
        block_rows=args.block_rows,
    )

    clear_tilelang_kernel_cache()
    started = time.perf_counter()
    kernel = compile_rms_norm_kernel(
        x,
        weight,
        args.eps,
        block_rows=args.block_rows,
    )
    _synchronize(device)
    compile_sec = time.perf_counter() - started
    return lambda: kernel(x, weight), resolved_backend, compile_sec


def _prepare_flash_attention_callable(
    *,
    backend: BenchmarkBackend,
    args: argparse.Namespace,
    device: torch.device,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> tuple[Callable[[], torch.Tensor], str, float | None]:
    if backend == "torch":
        return lambda: _reference_flash_attention(q, k, v, is_causal=args.is_causal), "torch", None

    resolved_backend = resolved_flash_attention_backend(
        q,
        k,
        v,
        backend,
        is_causal=args.is_causal,
    )

    clear_flash_attention_kernel_cache()
    started = time.perf_counter()
    kernel = compile_flash_attention_kernel(
        q,
        k,
        v,
        is_causal=args.is_causal,
    )
    _synchronize(device)
    compile_sec = time.perf_counter() - started
    key_lengths = torch.full((q.shape[0],), k.shape[2], dtype=torch.int32, device=q.device)
    return lambda: kernel(q, k, v, key_lengths), resolved_backend, compile_sec


def _prepare_embedding_bag_mean_callable(
    *,
    backend: BenchmarkBackend,
    args: argparse.Namespace,
    device: torch.device,
    embedding_weight: torch.Tensor,
    values: torch.Tensor,
) -> tuple[Callable[[], torch.Tensor], str, float | None]:
    if backend == "torch":
        return lambda: _reference_embedding_bag_mean(embedding_weight, values), "torch", None

    resolved_backend = resolved_embedding_bag_mean_backend(
        embedding_weight,
        values,
        backend,
        block_rows=args.block_rows,
        block_cols=args.block_cols,
    )

    clear_embedding_bag_mean_kernel_cache()
    started = time.perf_counter()
    kernel = compile_embedding_bag_mean_kernel(
        embedding_weight,
        values,
        block_rows=args.block_rows,
        block_cols=args.block_cols,
    )
    _synchronize(device)
    compile_sec = time.perf_counter() - started
    return lambda: kernel(embedding_weight, values), resolved_backend, compile_sec


def _build_inputs(args: argparse.Namespace, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed)
    x = torch.randn((args.rows, args.cols), generator=generator, dtype=args.dtype).to(device)
    weight = torch.randn((args.cols,), generator=generator, dtype=args.dtype).to(device)
    return x, weight


def _build_flash_attention_inputs(
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed)
    q = torch.randn((args.batch, args.heads, args.query_len, args.head_dim), generator=generator, dtype=args.dtype).to(device)
    k = torch.randn((args.batch, args.heads, args.kv_len, args.head_dim), generator=generator, dtype=args.dtype).to(device)
    v = torch.randn((args.batch, args.heads, args.kv_len, args.head_dim), generator=generator, dtype=args.dtype).to(device)
    return q, k, v


def _build_embedding_bag_mean_inputs(
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed)
    embedding_weight = torch.randn(
        (args.embedding_vocab_size + 1, args.embedding_dim),
        generator=generator,
        dtype=args.dtype,
    ).to(device)
    values = torch.randint(
        1,
        args.embedding_vocab_size + 1,
        (args.batch, args.embedding_bag_size),
        generator=generator,
        dtype=torch.long,
    )
    if args.embedding_padding_prob > 0.0:
        padding_mask = torch.rand(values.shape, generator=generator) < args.embedding_padding_prob
        values = values.masked_fill(padding_mask, 0)
    return embedding_weight, values.to(device)


def _benchmark_rms_norm_backend(
    *,
    backend: BenchmarkBackend,
    args: argparse.Namespace,
    device: torch.device,
    x: torch.Tensor,
    weight: torch.Tensor,
) -> OperatorBenchmarkResult:
    compile_sec: float | None = None
    try:
        reference = _reference_rms_norm(x, weight, args.eps)
        run, resolved_backend, compile_sec = _prepare_rms_norm_callable(
            backend=backend,
            args=args,
            device=device,
            x=x,
            weight=weight,
        )
        elapsed, step_time_ms = _benchmark_callable(
            run,
            device=device,
            warmup_steps=args.warmup_steps,
            steps=args.steps,
        )
        output = run()
        _synchronize(device)
        max_abs_error, max_rel_error = _validate_output_accuracy(
            output,
            reference,
            atol=args.atol,
            rtol=args.rtol,
        )
        return OperatorBenchmarkResult(
            backend=backend,
            status="ok",
            resolved_backend=resolved_backend,
            elapsed_sec=elapsed,
            step_time_ms=step_time_ms,
            ops_per_sec=args.steps / elapsed if elapsed > 0 else 0.0,
            measured_steps=args.steps,
            warmup_steps=args.warmup_steps,
            compile_sec=compile_sec,
            max_abs_error=max_abs_error,
            max_rel_error=max_rel_error,
        )
    except Exception as error:
        return OperatorBenchmarkResult(
            backend=backend,
            status=_benchmark_failure_status(backend, error),
            resolved_backend=None,
            elapsed_sec=0.0,
            step_time_ms=0.0,
            ops_per_sec=0.0,
            measured_steps=args.steps,
            warmup_steps=args.warmup_steps,
            compile_sec=compile_sec,
            max_abs_error=None,
            max_rel_error=None,
            error=f"{type(error).__name__}: {error}",
        )


def _benchmark_flash_attention_backend(
    *,
    backend: BenchmarkBackend,
    args: argparse.Namespace,
    device: torch.device,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> OperatorBenchmarkResult:
    compile_sec: float | None = None
    try:
        reference = _reference_flash_attention(q, k, v, is_causal=args.is_causal)
        run, resolved_backend, compile_sec = _prepare_flash_attention_callable(
            backend=backend,
            args=args,
            device=device,
            q=q,
            k=k,
            v=v,
        )
        elapsed, step_time_ms = _benchmark_callable(
            run,
            device=device,
            warmup_steps=args.warmup_steps,
            steps=args.steps,
        )
        output = run()
        _synchronize(device)
        max_abs_error, max_rel_error = _validate_output_accuracy(
            output,
            reference,
            atol=args.atol,
            rtol=args.rtol,
        )
        return OperatorBenchmarkResult(
            backend=backend,
            status="ok",
            resolved_backend=resolved_backend,
            elapsed_sec=elapsed,
            step_time_ms=step_time_ms,
            ops_per_sec=args.steps / elapsed if elapsed > 0 else 0.0,
            measured_steps=args.steps,
            warmup_steps=args.warmup_steps,
            compile_sec=compile_sec,
            max_abs_error=max_abs_error,
            max_rel_error=max_rel_error,
        )
    except Exception as error:
        return OperatorBenchmarkResult(
            backend=backend,
            status=_benchmark_failure_status(backend, error),
            resolved_backend=None,
            elapsed_sec=0.0,
            step_time_ms=0.0,
            ops_per_sec=0.0,
            measured_steps=args.steps,
            warmup_steps=args.warmup_steps,
            compile_sec=compile_sec,
            max_abs_error=None,
            max_rel_error=None,
            error=f"{type(error).__name__}: {error}",
        )


def _benchmark_embedding_bag_mean_backend(
    *,
    backend: BenchmarkBackend,
    args: argparse.Namespace,
    device: torch.device,
    embedding_weight: torch.Tensor,
    values: torch.Tensor,
) -> OperatorBenchmarkResult:
    compile_sec: float | None = None
    try:
        reference = _reference_embedding_bag_mean(embedding_weight, values)
        run, resolved_backend, compile_sec = _prepare_embedding_bag_mean_callable(
            backend=backend,
            args=args,
            device=device,
            embedding_weight=embedding_weight,
            values=values,
        )
        elapsed, step_time_ms = _benchmark_callable(
            run,
            device=device,
            warmup_steps=args.warmup_steps,
            steps=args.steps,
        )
        output = run()
        _synchronize(device)
        max_abs_error, max_rel_error = _validate_output_accuracy(
            output,
            reference,
            atol=args.atol,
            rtol=args.rtol,
        )
        return OperatorBenchmarkResult(
            backend=backend,
            status="ok",
            resolved_backend=resolved_backend,
            elapsed_sec=elapsed,
            step_time_ms=step_time_ms,
            ops_per_sec=args.steps / elapsed if elapsed > 0 else 0.0,
            measured_steps=args.steps,
            warmup_steps=args.warmup_steps,
            compile_sec=compile_sec,
            max_abs_error=max_abs_error,
            max_rel_error=max_rel_error,
        )
    except Exception as error:
        return OperatorBenchmarkResult(
            backend=backend,
            status=_benchmark_failure_status(backend, error),
            resolved_backend=None,
            elapsed_sec=0.0,
            step_time_ms=0.0,
            ops_per_sec=0.0,
            measured_steps=args.steps,
            warmup_steps=args.warmup_steps,
            compile_sec=compile_sec,
            max_abs_error=None,
            max_rel_error=None,
            error=f"{type(error).__name__}: {error}",
        )


def _summarize_runs(
    *,
    backend: BenchmarkBackend,
    repeats: int,
    runs: list[OperatorBenchmarkResult],
) -> OperatorBenchmarkSummary:
    successful_runs = [run for run in runs if run.status == "ok"]
    if not successful_runs:
        first = runs[0] if runs else None
        return OperatorBenchmarkSummary(
            backend=backend,
            status=first.status if first else "error",
            resolved_backend=first.resolved_backend if first else None,
            elapsed_sec=0.0,
            elapsed_sec_mean=0.0,
            elapsed_sec_median=0.0,
            elapsed_sec_p95=0.0,
            step_time_ms=0.0,
            step_time_ms_mean=0.0,
            step_time_ms_median=0.0,
            step_time_ms_p95=0.0,
            ops_per_sec=0.0,
            ops_per_sec_mean=0.0,
            ops_per_sec_median=0.0,
            ops_per_sec_p95=0.0,
            measured_steps=first.measured_steps if first else 0,
            warmup_steps=first.warmup_steps if first else 0,
            repeats=repeats,
            compile_sec=first.compile_sec if first else None,
            max_abs_error=None,
            max_rel_error=None,
            successful_repeats=0,
            runs=[asdict(run) for run in runs],
            error=first.error if first else "benchmark produced no successful runs",
        )

    elapsed_values = [run.elapsed_sec for run in successful_runs]
    step_values = [run.step_time_ms for run in successful_runs]
    throughput_values = [run.ops_per_sec for run in successful_runs]
    max_abs_errors = [run.max_abs_error for run in successful_runs if run.max_abs_error is not None]
    max_rel_errors = [run.max_rel_error for run in successful_runs if run.max_rel_error is not None]
    representative = successful_runs[0]
    return OperatorBenchmarkSummary(
        backend=backend,
        status="ok",
        resolved_backend=representative.resolved_backend,
        elapsed_sec=statistics.fmean(elapsed_values),
        elapsed_sec_mean=statistics.fmean(elapsed_values),
        elapsed_sec_median=statistics.median(elapsed_values),
        elapsed_sec_p95=_percentile(elapsed_values, 0.95),
        step_time_ms=statistics.fmean(step_values),
        step_time_ms_mean=statistics.fmean(step_values),
        step_time_ms_median=statistics.median(step_values),
        step_time_ms_p95=_percentile(step_values, 0.95),
        ops_per_sec=statistics.fmean(throughput_values),
        ops_per_sec_mean=statistics.fmean(throughput_values),
        ops_per_sec_median=statistics.median(throughput_values),
        ops_per_sec_p95=_percentile(throughput_values, 0.95),
        measured_steps=representative.measured_steps,
        warmup_steps=representative.warmup_steps,
        repeats=repeats,
        compile_sec=representative.compile_sec,
        max_abs_error=max(max_abs_errors) if max_abs_errors else None,
        max_rel_error=max(max_rel_errors) if max_rel_errors else None,
        successful_repeats=len(successful_runs),
        runs=[asdict(run) for run in runs],
        error=None,
    )


def run_benchmark(args: argparse.Namespace) -> dict[str, object]:
    os.environ.setdefault("TILELANG_PRINT_ON_COMPILATION", "0")
    if args.steps < 1:
        raise ValueError("steps must be >= 1")
    if args.warmup_steps < 0:
        raise ValueError("warmup_steps must be >= 0")
    if args.repeats < 1:
        raise ValueError("repeats must be >= 1")
    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but torch.cuda.is_available() is False")

    torch.manual_seed(args.seed)
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    results = []
    if args.operator == "rms_norm":
        x, weight = _build_inputs(args, device)
        for backend in args.backends:
            runs = [
                _benchmark_rms_norm_backend(
                    backend=backend,
                    args=args,
                    device=device,
                    x=x,
                    weight=weight,
                )
                for _ in range(args.repeats)
            ]
            results.append(asdict(_summarize_runs(backend=backend, repeats=args.repeats, runs=runs)))
        summary: dict[str, object] = {
            "operator": "rms_norm",
            "rows": args.rows,
            "cols": args.cols,
        }
    elif args.operator == "flash_attention":
        q, k, v = _build_flash_attention_inputs(args, device)
        for backend in args.backends:
            runs = [
                _benchmark_flash_attention_backend(
                    backend=backend,
                    args=args,
                    device=device,
                    q=q,
                    k=k,
                    v=v,
                )
                for _ in range(args.repeats)
            ]
            results.append(asdict(_summarize_runs(backend=backend, repeats=args.repeats, runs=runs)))
        summary = {
            "operator": "flash_attention",
            "batch": args.batch,
            "heads": args.heads,
            "query_len": args.query_len,
            "kv_len": args.kv_len,
            "head_dim": args.head_dim,
            "is_causal": args.is_causal,
        }
    elif args.operator == "embedding_bag_mean":
        embedding_weight, values = _build_embedding_bag_mean_inputs(args, device)
        for backend in args.backends:
            runs = [
                _benchmark_embedding_bag_mean_backend(
                    backend=backend,
                    args=args,
                    device=device,
                    embedding_weight=embedding_weight,
                    values=values,
                )
                for _ in range(args.repeats)
            ]
            results.append(asdict(_summarize_runs(backend=backend, repeats=args.repeats, runs=runs)))
        summary = {
            "operator": "embedding_bag_mean",
            "batch": args.batch,
            "embedding_vocab_size": args.embedding_vocab_size,
            "embedding_dim": args.embedding_dim,
            "embedding_bag_size": args.embedding_bag_size,
            "embedding_padding_prob": args.embedding_padding_prob,
            "block_cols": args.block_cols,
        }
    else:
        raise ValueError(f"unsupported operator benchmark: {args.operator}")

    summary.update({
        "device": str(device),
        "dtype": str(args.dtype).replace("torch.", ""),
        "eps": args.eps,
        "steps": args.steps,
        "warmup_steps": args.warmup_steps,
        "repeats": args.repeats,
        "seed": args.seed,
        "block_rows": args.block_rows,
        "atol": args.atol,
        "rtol": args.rtol,
        "tilelang_installed": tilelang_available(),
        "backends": list(args.backends),
        "results": results,
    })
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operator", choices=("rms_norm", "flash_attention", "embedding_bag_mean"), default="rms_norm")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--rows", type=int, default=8192)
    parser.add_argument("--cols", type=int, default=128)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--query-len", type=int, default=128)
    parser.add_argument("--kv-len", type=int, default=128)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--embedding-vocab-size", type=int, default=1_000_000)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--embedding-bag-size", type=int, default=4)
    parser.add_argument("--embedding-padding-prob", type=float, default=0.25)
    parser.add_argument("--is-causal", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--block-rows", type=int, default=1)
    parser.add_argument("--block-cols", type=int, default=None)
    parser.add_argument("--torch-threads", type=int, default=0)
    parser.add_argument(
        "--dtype",
        choices=("float16", "bfloat16", "float32"),
        default="float16" if torch.cuda.is_available() else "float32",
    )
    parser.add_argument(
        "--backends",
        default=",".join(DEFAULT_BACKENDS),
        help="comma-separated subset of torch,tilelang",
    )
    parser.add_argument("--atol", type=float, default=None, help="absolute tolerance for accuracy checks")
    parser.add_argument("--rtol", type=float, default=None, help="relative tolerance for accuracy checks")
    args = parser.parse_args(argv)
    args.backends = _parse_backend_names(args.backends)
    args.dtype = getattr(torch, args.dtype)
    default_atol, default_rtol = _default_error_tolerances(args.dtype)
    if args.atol is None:
        args.atol = default_atol
    if args.rtol is None:
        args.rtol = default_rtol
    return args


def main(argv: Sequence[str] | None = None) -> int:
    write_stdout_line(dumps(run_benchmark(parse_args(argv)), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
