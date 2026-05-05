"""Benchmark dense optimizer step throughput on a synthetic PCVR trainer workload."""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict, dataclass
import math
import statistics
import tempfile
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any, NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from taac2026.infrastructure.io.json import dumps
from taac2026.infrastructure.runtime.trainer import PCVRPointwiseTrainer
from taac2026.infrastructure.runtime.execution import (
    DENSE_OPTIMIZER_TYPE_CHOICES,
    EarlyStopping,
    RuntimeExecutionConfig,
)


DEFAULT_BENCHMARK_OPTIMIZERS: tuple[str, ...] = (
    "adamw",
    "fused_adamw",
    "orthogonal_adamw",
    "muon",
)


class ModelInput(NamedTuple):
    user_int_feats: torch.Tensor
    item_int_feats: torch.Tensor
    user_dense_feats: torch.Tensor
    item_dense_feats: torch.Tensor
    seq_data: dict[str, torch.Tensor]
    seq_lens: dict[str, torch.Tensor]
    seq_time_buckets: dict[str, torch.Tensor]


class SyntheticPCVRModel(nn.Module):
    def __init__(self, *, feature_dim: int, hidden_dim: int, depth: int) -> None:
        super().__init__()
        input_dim = feature_dim * 2
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.hidden_layers = nn.ModuleList(
            nn.Linear(hidden_dim, hidden_dim) for _ in range(max(0, depth - 1))
        )
        self.output_layer = nn.Linear(hidden_dim, 1)

    def forward(self, model_input: ModelInput) -> torch.Tensor:
        features = torch.cat(
            [model_input.user_dense_feats, model_input.item_dense_feats],
            dim=-1,
        )
        hidden = F.gelu(self.input_layer(features))
        for layer in self.hidden_layers:
            hidden = F.gelu(layer(hidden))
        return self.output_layer(hidden)

    def predict(self, model_input: ModelInput) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.forward(model_input)
        return logits, torch.empty(0, device=logits.device)


@dataclass(slots=True)
class OptimizerBenchmarkResult:
    optimizer: str
    status: str
    optimizer_class: str | None
    elapsed_sec: float
    step_time_ms: float
    steps_per_sec: float
    measured_steps: int
    warmup_steps: int
    final_loss: float | None
    error: str | None = None


@dataclass(slots=True)
class OptimizerBenchmarkSummary:
    optimizer: str
    status: str
    optimizer_class: str | None
    elapsed_sec: float
    elapsed_sec_mean: float
    elapsed_sec_median: float
    elapsed_sec_p95: float
    step_time_ms: float
    step_time_ms_mean: float
    step_time_ms_median: float
    step_time_ms_p95: float
    steps_per_sec: float
    steps_per_sec_mean: float
    steps_per_sec_median: float
    steps_per_sec_p95: float
    measured_steps: int
    warmup_steps: int
    repeats: int
    final_loss: float | None
    final_loss_mean: float | None
    successful_repeats: int
    runs: list[dict[str, Any]]
    error: str | None = None


def _parse_optimizer_names(value: str) -> tuple[str, ...]:
    names = tuple(name.strip().lower() for name in value.split(",") if name.strip())
    if not names:
        raise ValueError("at least one optimizer must be specified")
    invalid = [name for name in names if name not in DENSE_OPTIMIZER_TYPE_CHOICES]
    if invalid:
        raise ValueError(
            f"unsupported optimizer names: {', '.join(invalid)}; expected values from {DENSE_OPTIMIZER_TYPE_CHOICES}"
        )
    return names


def _build_synthetic_batch(
    *,
    batch_size: int,
    feature_dim: int,
    seed: int,
) -> dict[str, object]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    labels = torch.randint(0, 2, (batch_size,), generator=generator, dtype=torch.int64).float()
    return {
        "label": labels,
        "_seq_domains": [],
        "user_int_feats": torch.zeros((batch_size, 1), dtype=torch.long),
        "item_int_feats": torch.zeros((batch_size, 1), dtype=torch.long),
        "user_dense_feats": torch.randn((batch_size, feature_dim), generator=generator),
        "item_dense_feats": torch.randn((batch_size, feature_dim), generator=generator),
    }


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
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _summarize_optimizer_runs(
    *,
    optimizer_name: str,
    repeats: int,
    runs: list[OptimizerBenchmarkResult],
) -> OptimizerBenchmarkSummary:
    successful_runs = [run for run in runs if run.status == "ok"]
    if not successful_runs:
        error = runs[0].error if runs else "benchmark produced no successful runs"
        optimizer_class = runs[0].optimizer_class if runs else None
        status = runs[0].status if runs else "error"
        measured_steps = runs[0].measured_steps if runs else 0
        warmup_steps = runs[0].warmup_steps if runs else 0
        return OptimizerBenchmarkSummary(
            optimizer=optimizer_name,
            status=status,
            optimizer_class=optimizer_class,
            elapsed_sec=0.0,
            elapsed_sec_mean=0.0,
            elapsed_sec_median=0.0,
            elapsed_sec_p95=0.0,
            step_time_ms=0.0,
            step_time_ms_mean=0.0,
            step_time_ms_median=0.0,
            step_time_ms_p95=0.0,
            steps_per_sec=0.0,
            steps_per_sec_mean=0.0,
            steps_per_sec_median=0.0,
            steps_per_sec_p95=0.0,
            measured_steps=measured_steps,
            warmup_steps=warmup_steps,
            repeats=repeats,
            final_loss=None,
            final_loss_mean=None,
            successful_repeats=0,
            runs=[asdict(run) for run in runs],
            error=error,
        )

    elapsed_values = [run.elapsed_sec for run in successful_runs]
    step_time_values = [run.step_time_ms for run in successful_runs]
    throughput_values = [run.steps_per_sec for run in successful_runs]
    final_losses = [run.final_loss for run in successful_runs if run.final_loss is not None]
    representative = successful_runs[0]
    return OptimizerBenchmarkSummary(
        optimizer=optimizer_name,
        status="ok",
        optimizer_class=representative.optimizer_class,
        elapsed_sec=statistics.fmean(elapsed_values),
        elapsed_sec_mean=statistics.fmean(elapsed_values),
        elapsed_sec_median=statistics.median(elapsed_values),
        elapsed_sec_p95=_percentile(elapsed_values, 0.95),
        step_time_ms=statistics.fmean(step_time_values),
        step_time_ms_mean=statistics.fmean(step_time_values),
        step_time_ms_median=statistics.median(step_time_values),
        step_time_ms_p95=_percentile(step_time_values, 0.95),
        steps_per_sec=statistics.fmean(throughput_values),
        steps_per_sec_mean=statistics.fmean(throughput_values),
        steps_per_sec_median=statistics.median(throughput_values),
        steps_per_sec_p95=_percentile(throughput_values, 0.95),
        measured_steps=representative.measured_steps,
        warmup_steps=representative.warmup_steps,
        repeats=repeats,
        final_loss=statistics.fmean(final_losses) if final_losses else None,
        final_loss_mean=statistics.fmean(final_losses) if final_losses else None,
        successful_repeats=len(successful_runs),
        runs=[asdict(run) for run in runs],
        error=None,
    )


def _benchmark_optimizer(
    *,
    optimizer_name: str,
    args: argparse.Namespace,
    device: torch.device,
    base_state: dict[str, Any],
    batch: dict[str, object],
) -> OptimizerBenchmarkResult:
    model = SyntheticPCVRModel(
        feature_dim=args.feature_dim,
        hidden_dim=args.hidden_dim,
        depth=args.depth,
    )
    model.load_state_dict(base_state)
    model.to(device)

    try:
        with tempfile.TemporaryDirectory(prefix="pcvr_optimizer_bench_") as tempdir:
            trainer = PCVRPointwiseTrainer(
                model=model,
                model_input_type=ModelInput,
                train_loader=[],
                valid_loader=[],
                lr=args.lr,
                max_steps=args.steps,
                device=str(device),
                save_dir=Path(tempdir),
                early_stopping=EarlyStopping(Path(tempdir) / "best" / "model.safetensors", patience=2),
                dense_optimizer_type=optimizer_name,
                runtime_execution=RuntimeExecutionConfig(
                    amp=args.amp,
                    amp_dtype=args.amp_dtype,
                    compile=args.compile,
                ),
            )
            for _ in range(args.warmup_steps):
                trainer._train_step(batch)
            _synchronize(device)

            final_loss: float | None = None
            started = time.perf_counter()
            for _ in range(args.steps):
                final_loss = trainer._train_step(batch)
            _synchronize(device)
            elapsed = time.perf_counter() - started
    except Exception as error:
        status = "unsupported" if optimizer_name == "fused_adamw" else "error"
        return OptimizerBenchmarkResult(
            optimizer=optimizer_name,
            status=status,
            optimizer_class=None,
            elapsed_sec=0.0,
            step_time_ms=0.0,
            steps_per_sec=0.0,
            measured_steps=args.steps,
            warmup_steps=args.warmup_steps,
            final_loss=None,
            error=f"{type(error).__name__}: {error}",
        )

    step_time_ms = (elapsed / args.steps) * 1000.0 if args.steps > 0 else 0.0
    steps_per_sec = args.steps / elapsed if elapsed > 0 else 0.0
    return OptimizerBenchmarkResult(
        optimizer=optimizer_name,
        status="ok",
        optimizer_class=trainer.dense_optimizer.__class__.__name__,
        elapsed_sec=elapsed,
        step_time_ms=step_time_ms,
        steps_per_sec=steps_per_sec,
        measured_steps=args.steps,
        warmup_steps=args.warmup_steps,
        final_loss=final_loss,
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

    base_model = SyntheticPCVRModel(
        feature_dim=args.feature_dim,
        hidden_dim=args.hidden_dim,
        depth=args.depth,
    )
    base_state = deepcopy(base_model.state_dict())
    batch = _build_synthetic_batch(
        batch_size=args.batch_size,
        feature_dim=args.feature_dim,
        seed=args.seed + 1,
    )
    results = []
    for optimizer_name in args.optimizers:
        runs = [
            _benchmark_optimizer(
                optimizer_name=optimizer_name,
                args=args,
                device=device,
                base_state=base_state,
                batch=batch,
            )
            for _ in range(args.repeats)
        ]
        results.append(
            asdict(
                _summarize_optimizer_runs(
                    optimizer_name=optimizer_name,
                    repeats=args.repeats,
                    runs=runs,
                )
            )
        )
    return {
        "device": str(device),
        "batch_size": args.batch_size,
        "feature_dim": args.feature_dim,
        "hidden_dim": args.hidden_dim,
        "depth": args.depth,
        "steps": args.steps,
        "warmup_steps": args.warmup_steps,
        "repeats": args.repeats,
        "seed": args.seed,
        "amp": args.amp,
        "amp_dtype": args.amp_dtype,
        "compile": args.compile,
        "optimizers": list(args.optimizers),
        "results": results,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--feature-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--torch-threads", type=int, default=0)
    parser.add_argument(
        "--optimizers",
        default=",".join(DEFAULT_BENCHMARK_OPTIMIZERS),
        help="comma-separated subset of dense optimizers to benchmark",
    )
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--amp-dtype",
        dest="amp_dtype",
        choices=("bfloat16", "float16"),
        default="bfloat16",
    )
    parser.add_argument(
        "--compile",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    args = parser.parse_args(argv)
    args.optimizers = _parse_optimizer_names(args.optimizers)
    return args


def main(argv: Sequence[str] | None = None) -> int:
    print(dumps(run_benchmark(parse_args(argv)), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())