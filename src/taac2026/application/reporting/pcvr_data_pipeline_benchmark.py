"""Benchmark PCVR data pipeline throughput without running a model."""

from __future__ import annotations

import argparse
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

from taac2026.infrastructure.io.json_utils import dumps
from taac2026.infrastructure.pcvr.config import (
    PCVRDataCacheConfig,
    PCVRDataPipelineConfig,
    PCVRDomainDropoutConfig,
    PCVRFeatureMaskConfig,
    PCVRSequenceCropConfig,
)
from taac2026.infrastructure.pcvr.data import get_pcvr_data
from taac2026.infrastructure.pcvr.protocol import parse_seq_max_lens


def _build_pipeline_config(args: argparse.Namespace) -> PCVRDataPipelineConfig:
    transforms = []
    cache = PCVRDataCacheConfig()
    if args.pipeline_preset in {"cache", "augment"}:
        cache = PCVRDataCacheConfig(mode="memory", max_batches=args.cache_batches)
    if args.pipeline_preset == "augment":
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


def run_benchmark(args: argparse.Namespace) -> dict[str, object]:
    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)

    pipeline_config = _build_pipeline_config(args)
    train_loader, _valid_loader, train_dataset = get_pcvr_data(
        data_dir=str(args.dataset_path),
        schema_path=str(args.schema_path),
        batch_size=args.batch_size,
        valid_ratio=args.valid_ratio,
        train_ratio=args.train_ratio,
        num_workers=args.num_workers,
        buffer_batches=args.buffer_batches,
        shuffle_train=not args.no_shuffle,
        seed=args.seed,
        seq_max_lens=parse_seq_max_lens(args.seq_max_lens),
        data_pipeline_config=pipeline_config,
    )

    iterator = iter(train_loader)
    warmup_rows = 0
    try:
        warmup_rows = _consume_batches(iterator, args.warmup_batches)
    except StopIteration:
        return {
            "dataset_path": str(args.dataset_path),
            "schema_path": str(args.schema_path),
            "pipeline_preset": args.pipeline_preset,
            "train_rows": train_dataset.num_rows,
            "measured_rows": 0,
            "measured_batches": 0,
            "warmup_rows": warmup_rows,
            "elapsed_sec": 0.0,
            "rows_per_sec": 0.0,
            "batches_per_sec": 0.0,
        }

    measured_rows = 0
    measured_batches = 0
    started = time.perf_counter()
    while args.max_batches <= 0 or measured_batches < args.max_batches:
        try:
            batch = next(iterator)
        except StopIteration:
            break
        measured_rows += int(batch["label"].shape[0])
        measured_batches += 1
    elapsed = time.perf_counter() - started

    return {
        "dataset_path": str(args.dataset_path),
        "schema_path": str(args.schema_path),
        "pipeline_preset": args.pipeline_preset,
        "train_rows": train_dataset.num_rows,
        "train_row_groups": len(train_dataset._rg_list),
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "buffer_batches": args.buffer_batches,
        "shuffle": not args.no_shuffle,
        "warmup_batches": args.warmup_batches,
        "warmup_rows": warmup_rows,
        "measured_rows": measured_rows,
        "measured_batches": measured_batches,
        "elapsed_sec": elapsed,
        "rows_per_sec": measured_rows / elapsed if elapsed > 0 else 0.0,
        "batches_per_sec": measured_batches / elapsed if elapsed > 0 else 0.0,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--schema-path", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--buffer-batches", type=int, default=1)
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    parser.add_argument("--train-ratio", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--seq-max-lens", default="seq_a:256,seq_b:256,seq_c:512,seq_d:512"
    )
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--warmup-batches", type=int, default=5)
    parser.add_argument("--torch-threads", type=int, default=0)
    parser.add_argument("--no-shuffle", action="store_true")
    parser.add_argument(
        "--preset",
        dest="pipeline_preset",
        choices=("none", "cache", "augment"),
        default="none",
        help="alias of --pipeline-preset",
    )
    parser.add_argument(
        "--pipeline-preset",
        choices=("none", "cache", "augment"),
        default=None,
    )
    parser.add_argument("--cache-batches", type=int, default=512)
    parser.add_argument("--views-per-row", type=int, default=2)
    parser.add_argument("--seq-window-min-len", type=int, default=8)
    parser.add_argument("--feature-mask-probability", type=float, default=0.05)
    parser.add_argument("--domain-dropout-probability", type=float, default=0.05)
    parser.add_argument(
        "--strict-time-filter", action=argparse.BooleanOptionalAction, default=True
    )
    args = parser.parse_args(argv)
    if args.pipeline_preset is None:
        args.pipeline_preset = args.preset
    return args


def main(argv: Sequence[str] | None = None) -> int:
    summary = run_benchmark(parse_args(argv))
    print(dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
