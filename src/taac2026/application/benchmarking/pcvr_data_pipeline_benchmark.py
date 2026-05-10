"""Benchmark PCVR data pipeline throughput without running a model."""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal

import torch
import pyarrow.parquet as pq
import tyro

from taac2026.infrastructure.io.json import dumps
from taac2026.infrastructure.io.rich_output import print_rich_summary
from taac2026.infrastructure.io.streams import write_stdout_line
from taac2026.domain.config import (
    PCVRDataCacheMode,
    PCVRDataCacheConfig,
    PCVRDataPipelineConfig,
    PCVRDataSamplingStrategy,
    PCVRDomainDropoutConfig,
    PCVRFeatureMaskConfig,
    PCVRSequenceCropConfig,
)
from taac2026.infrastructure.data.dataset import PCVRParquetDataset, get_pcvr_data
from taac2026.infrastructure.modeling.model_contract import parse_seq_max_lens


BenchmarkMode = Literal["loader", "convert"]
PipelinePreset = Literal["none", "cache", "opt", "augment", "opt-augment"]


@dataclass(slots=True)
class PCVRDataPipelineBenchmarkArgs:
    dataset_path: Path
    schema_path: Path
    batch_size: int = 256
    num_workers: int = 0
    buffer_batches: int = 1
    valid_ratio: float = 0.1
    train_ratio: float = 1.0
    seed: int = 42
    seq_max_lens: str = "seq_a:256,seq_b:256,seq_c:512,seq_d:512"
    max_batches: int = 0
    warmup_batches: int = 5
    passes: int = 1
    torch_threads: int = 0
    no_shuffle: bool = False
    benchmark_mode: BenchmarkMode = "loader"
    converter_batches: int = 128
    sampling_strategy: PCVRDataSamplingStrategy = "step_random"
    train_steps_per_sweep: int = 0
    pipeline_preset: Annotated[PipelinePreset, tyro.conf.arg(aliases=("--preset",))] = "none"
    cache_mode: PCVRDataCacheMode | None = None
    cache_batches: int = 512
    views_per_row: int = 2
    seq_window_min_len: int = 8
    feature_mask_probability: float = 0.05
    domain_dropout_probability: float = 0.05
    strict_time_filter: bool = True
    json: bool = False


def _build_pipeline_config(args: Any) -> PCVRDataPipelineConfig:
    transforms = []
    cache_mode = getattr(args, "cache_mode", None)
    if cache_mode is None:
        if args.pipeline_preset in {"cache", "augment"}:
            cache_mode = "lru"
        elif args.pipeline_preset in {"opt", "opt-augment"}:
            cache_mode = "opt"
        else:
            cache_mode = "none"
    cache = (
        PCVRDataCacheConfig()
        if cache_mode == "none"
        else PCVRDataCacheConfig(mode=cache_mode, max_batches=args.cache_batches)
    )
    if args.pipeline_preset in {"augment", "opt-augment"}:
        transforms.extend(
            [
                PCVRSequenceCropConfig(
                    views_per_row=args.views_per_row,
                    seq_window_mode="random_tail",
                    seq_window_min_len=args.seq_window_min_len,
                ),
                PCVRFeatureMaskConfig(probability=args.feature_mask_probability),
                PCVRDomainDropoutConfig(probability=args.domain_dropout_probability),
            ]
        )
    return PCVRDataPipelineConfig(
        cache=cache,
        transforms=tuple(transforms),
        seed=args.seed,
        strict_time_filter=args.strict_time_filter,
    )


def _consume_batches(iterator: Any, batch_count: int) -> int:
    rows = 0
    for _ in range(batch_count):
        batch = next(iterator)
        rows += int(batch["label"].shape[0])
    return rows


def _consume_measured_batches(
    train_loader: Any,
    *,
    batches_per_pass: int,
    passes: int,
    max_batches: int,
) -> tuple[int, int, int]:
    measured_rows = 0
    measured_batches = 0
    measured_passes = 0

    if max_batches > 0:
        iterator = iter(train_loader)
        while measured_batches < max_batches:
            try:
                batch = next(iterator)
            except StopIteration:
                break
            measured_rows += int(batch["label"].shape[0])
            measured_batches += 1
        if measured_batches > 0:
            measured_passes = math.ceil(measured_batches / max(1, batches_per_pass))
        return measured_rows, measured_batches, measured_passes

    for _ in range(max(1, passes)):
        iterator = iter(train_loader)
        pass_batches = 0
        while pass_batches < batches_per_pass:
            try:
                batch = next(iterator)
            except StopIteration:
                break
            measured_rows += int(batch["label"].shape[0])
            measured_batches += 1
            pass_batches += 1
        if pass_batches == 0:
            break
        measured_passes += 1
    return measured_rows, measured_batches, measured_passes


def _estimated_batches_per_pass(
    *,
    train_rows: int,
    batch_size: int,
    pipeline_config: PCVRDataPipelineConfig,
) -> int:
    row_multiplier = 1
    for transform in pipeline_config.transforms:
        if isinstance(transform, PCVRSequenceCropConfig) and transform.enabled:
            row_multiplier *= max(1, int(transform.views_per_row))
    return max(1, math.ceil((train_rows * row_multiplier) / max(1, batch_size)))


def run_benchmark(args: PCVRDataPipelineBenchmarkArgs) -> dict[str, object]:
    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)

    pipeline_config = _build_pipeline_config(args)
    if args.benchmark_mode == "convert":
        return _run_converter_benchmark(args, pipeline_config)

    train_loader, _valid_loader, train_dataset = get_pcvr_data(
        data_dir=str(args.dataset_path),
        schema_path=str(args.schema_path),
        batch_size=args.batch_size,
        valid_ratio=args.valid_ratio,
        train_ratio=args.train_ratio,
        num_workers=args.num_workers,
        buffer_batches=args.buffer_batches,
        shuffle_train=not args.no_shuffle,
        sampling_strategy=args.sampling_strategy,
        train_steps_per_sweep=args.train_steps_per_sweep,
        seed=args.seed,
        seq_max_lens=parse_seq_max_lens(args.seq_max_lens),
        data_pipeline_config=pipeline_config,
        max_steps=(args.max_batches + args.warmup_batches if args.max_batches > 0 else 0),
    )

    warmup_rows = 0
    iterator_after_warmup: Any | None = None
    try:
        iterator_after_warmup = iter(train_loader)
        warmup_rows = _consume_batches(iterator_after_warmup, args.warmup_batches)
    except StopIteration:
        return {
            "dataset_path": str(args.dataset_path),
            "schema_path": str(args.schema_path),
            "pipeline_preset": args.pipeline_preset,
            "train_rows": train_dataset.num_rows,
            "measured_rows": 0,
            "measured_batches": 0,
            "warmup_rows": warmup_rows,
            "passes": args.passes,
            "measured_passes": 0,
            "elapsed_sec": 0.0,
            "rows_per_sec": 0.0,
            "batches_per_sec": 0.0,
        }

    batches_per_pass = _estimated_batches_per_pass(
        train_rows=train_dataset.num_rows,
        batch_size=args.batch_size,
        pipeline_config=pipeline_config,
    )
    started = time.perf_counter()
    measured_rows, measured_batches, measured_passes = _consume_measured_batches(
        iterator_after_warmup if args.max_batches > 0 else train_loader,
        batches_per_pass=batches_per_pass,
        passes=args.passes,
        max_batches=args.max_batches,
    )
    elapsed = time.perf_counter() - started
    cache = getattr(train_dataset.pipeline, "cache", None)
    cache_stats_fn = getattr(cache, "stats", None)
    cache_stats = cache_stats_fn() if callable(cache_stats_fn) else {}

    return {
        "dataset_path": str(args.dataset_path),
        "schema_path": str(args.schema_path),
        "benchmark_mode": args.benchmark_mode,
        "pipeline_preset": args.pipeline_preset,
        "cache_mode": pipeline_config.cache.mode,
        "cache_impl": train_dataset.pipeline.cache.__class__.__name__,
        "shared_cache": train_dataset.pipeline.cache.__class__.__name__ == "PCVRSharedBatchCache",
        "data_cache_stats": cache_stats,
        "train_rows": train_dataset.num_rows,
        "train_row_groups": len(train_dataset.row_groups),
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "buffer_batches": args.buffer_batches,
        "shuffle": not args.no_shuffle,
        "sampling_strategy": args.sampling_strategy,
        "train_steps_per_sweep": args.train_steps_per_sweep,
        "warmup_batches": args.warmup_batches,
        "warmup_rows": warmup_rows,
        "passes": args.passes,
        "batches_per_pass": batches_per_pass,
        "measured_passes": measured_passes,
        "measured_rows": measured_rows,
        "measured_batches": measured_batches,
        "elapsed_sec": elapsed,
        "rows_per_sec": measured_rows / elapsed if elapsed > 0 else 0.0,
        "batches_per_sec": measured_batches / elapsed if elapsed > 0 else 0.0,
    }


def _collect_converter_batches(
    dataset: PCVRParquetDataset,
    batch_count: int,
) -> list[object]:
    record_batches: list[object] = []
    for file_path, row_group_index, _row_count in dataset.row_groups:
        parquet_file = pq.ParquetFile(file_path)
        for record_batch in parquet_file.iter_batches(
            batch_size=dataset.batch_size,
            row_groups=[row_group_index],
            columns=dataset.record_batch_columns(),
        ):
            record_batches.append(record_batch)
            if len(record_batches) >= batch_count:
                return record_batches
    return record_batches


def _run_converter_benchmark(
    args: PCVRDataPipelineBenchmarkArgs,
    pipeline_config: PCVRDataPipelineConfig,
) -> dict[str, object]:
    dataset = PCVRParquetDataset(
        parquet_path=str(args.dataset_path),
        schema_path=str(args.schema_path),
        batch_size=args.batch_size,
        seq_max_lens=parse_seq_max_lens(args.seq_max_lens),
        shuffle=False,
        buffer_batches=0,
        data_pipeline_config=pipeline_config,
        is_training=True,
        dataset_role="convert_benchmark",
    )
    measured_batches_target = args.max_batches if args.max_batches > 0 else args.converter_batches
    measured_batches_target = max(1, int(measured_batches_target))
    preload_batches = max(1, int(args.warmup_batches) + measured_batches_target)
    record_batches = _collect_converter_batches(dataset, preload_batches)
    if not record_batches:
        return {
            "dataset_path": str(args.dataset_path),
            "schema_path": str(args.schema_path),
            "benchmark_mode": args.benchmark_mode,
            "pipeline_preset": args.pipeline_preset,
            "train_rows": dataset.num_rows,
            "measured_rows": 0,
            "measured_batches": 0,
            "warmup_batches": args.warmup_batches,
            "elapsed_sec": 0.0,
            "rows_per_sec": 0.0,
            "batches_per_sec": 0.0,
        }

    for warmup_index in range(max(0, int(args.warmup_batches))):
        dataset.convert_record_batch(record_batches[warmup_index % len(record_batches)])

    measured_rows = 0
    started = time.perf_counter()
    for batch_index in range(measured_batches_target):
        converted = dataset.convert_record_batch(
            record_batches[batch_index % len(record_batches)]
        )
        measured_rows += int(converted["label"].shape[0])
    elapsed = time.perf_counter() - started

    return {
        "dataset_path": str(args.dataset_path),
        "schema_path": str(args.schema_path),
        "benchmark_mode": args.benchmark_mode,
        "pipeline_preset": args.pipeline_preset,
        "train_rows": dataset.num_rows,
        "train_row_groups": len(dataset.row_groups),
        "batch_size": args.batch_size,
        "preloaded_batches": len(record_batches),
        "measured_rows": measured_rows,
        "measured_batches": measured_batches_target,
        "warmup_batches": args.warmup_batches,
        "strict_time_filter": dataset.strict_time_filter_enabled,
        "elapsed_sec": elapsed,
        "rows_per_sec": measured_rows / elapsed if elapsed > 0 else 0.0,
        "batches_per_sec": measured_batches_target / elapsed if elapsed > 0 else 0.0,
    }


def parse_args(argv: Sequence[str] | None = None) -> PCVRDataPipelineBenchmarkArgs:
    return tyro.cli(PCVRDataPipelineBenchmarkArgs, description=__doc__, args=argv)


def _format_pipeline_summary(summary: dict[str, object]) -> None:
    fields = [
        ("Dataset", str(summary.get("dataset_path", "<unknown>"))),
        ("Schema", str(summary.get("schema_path", "<unknown>"))),
        ("Mode", str(summary.get("benchmark_mode", "<unknown>"))),
        ("Preset", str(summary.get("pipeline_preset", "<unknown>"))),
        ("Cache mode", str(summary.get("cache_mode", "<unknown>"))),
        ("Train rows", f"{summary.get('train_rows', 0):,}"),
        ("Batch size", str(summary.get("batch_size", 0))),
    ]
    sections = [
        ("Results", [
            ("Measured rows", f"{summary.get('measured_rows', 0):,}"),
            ("Measured batches", str(summary.get("measured_batches", 0))),
            ("Elapsed", f"{summary.get('elapsed_sec', 0.0):.3f}s"),
            ("Rows/sec", f"{summary.get('rows_per_sec', 0.0):.1f}"),
            ("Batches/sec", f"{summary.get('batches_per_sec', 0.0):.1f}"),
        ]),
    ]
    print_rich_summary("PCVR data pipeline benchmark", fields, sections=sections, border_style="blue")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    summary = run_benchmark(args)
    if args.json:
        write_stdout_line(dumps(summary, indent=2))
    else:
        _format_pipeline_summary(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
