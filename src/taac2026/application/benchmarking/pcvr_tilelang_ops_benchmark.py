"""Benchmark TileLang-backed PCVR operators on synthetic tensors."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import statistics
import time
from collections.abc import Callable, Sequence
from typing import Any, Literal

import torch

from taac2026.infrastructure.io.json import dumps
from taac2026.infrastructure.io.streams import write_stdout_line
from taac2026.infrastructure.accelerators import (
    clear_tilelang_kernel_cache,
    compile_rms_norm_kernel,
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


def _build_inputs(args: argparse.Namespace, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed)
    x = torch.randn((args.rows, args.cols), generator=generator, dtype=args.dtype).to(device)
    weight = torch.randn((args.cols,), generator=generator, dtype=args.dtype).to(device)
    return x, weight


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
        max_abs_error = float((output - reference).abs().max().item())
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
        )
    except Exception as error:
        status = "unsupported" if backend == "tilelang" else "error"
        return OperatorBenchmarkResult(
            backend=backend,
            status=status,
            resolved_backend=None,
            elapsed_sec=0.0,
            step_time_ms=0.0,
            ops_per_sec=0.0,
            measured_steps=args.steps,
            warmup_steps=args.warmup_steps,
            compile_sec=compile_sec,
            max_abs_error=None,
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
            successful_repeats=0,
            runs=[asdict(run) for run in runs],
            error=first.error if first else "benchmark produced no successful runs",
        )

    elapsed_values = [run.elapsed_sec for run in successful_runs]
    step_values = [run.step_time_ms for run in successful_runs]
    throughput_values = [run.ops_per_sec for run in successful_runs]
    max_abs_errors = [run.max_abs_error for run in successful_runs if run.max_abs_error is not None]
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
        successful_repeats=len(successful_runs),
        runs=[asdict(run) for run in runs],
        error=None,
    )


def run_benchmark(args: argparse.Namespace) -> dict[str, object]:
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

    x, weight = _build_inputs(args, device)
    results = []
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

    return {
        "operator": "rms_norm",
        "device": str(device),
        "dtype": str(args.dtype).replace("torch.", ""),
        "rows": args.rows,
        "cols": args.cols,
        "eps": args.eps,
        "steps": args.steps,
        "warmup_steps": args.warmup_steps,
        "repeats": args.repeats,
        "seed": args.seed,
        "block_rows": args.block_rows,
        "tilelang_installed": tilelang_available(),
        "backends": list(args.backends),
        "results": results,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--rows", type=int, default=8192)
    parser.add_argument("--cols", type=int, default=128)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--block-rows", type=int, default=1)
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
    args = parser.parse_args(argv)
    args.backends = _parse_backend_names(args.backends)
    args.dtype = getattr(torch, args.dtype)
    return args


def main(argv: Sequence[str] | None = None) -> int:
    write_stdout_line(dumps(run_benchmark(parse_args(argv)), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())