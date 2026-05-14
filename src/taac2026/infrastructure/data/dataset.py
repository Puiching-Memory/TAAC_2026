"""Public PCVR data-loading entrypoints."""

from __future__ import annotations

import random
from typing import Any

import torch
from torch.utils.data import DataLoader

from taac2026.domain.config import (
    PCVR_DATA_SAMPLING_STRATEGY_CHOICES,
    PCVR_DATA_SPLIT_STRATEGY_CHOICES,
    PCVRDataPipelineConfig,
)
from taac2026.domain.schema import BUCKET_BOUNDARIES, NUM_TIME_BUCKETS, FeatureSchema
from taac2026.infrastructure.data.observation import (
    PCVRRowGroupSplitPlan,
    PCVRTimestampRange,
    build_pcvr_observed_schema_report,
    collect_pcvr_row_groups,
    count_pcvr_rows_in_timestamp_range,
    normalize_pcvr_timestamp_range,
    pcvr_timestamp_range_to_dict,
    plan_pcvr_row_group_split,
)
from taac2026.infrastructure.data.parquet_dataset import PCVRHashSplitFilter, PCVRParquetDataset
from taac2026.infrastructure.logging import logger


_STEP_RANDOM_BUFFER_BATCHES = 1
_TORCH_SHARING_STRATEGY_CONFIGURED = False


def ensure_torch_file_system_sharing_strategy() -> None:
    global _TORCH_SHARING_STRATEGY_CONFIGURED
    if _TORCH_SHARING_STRATEGY_CONFIGURED:
        return
    torch.multiprocessing.set_sharing_strategy("file_system")
    _TORCH_SHARING_STRATEGY_CONFIGURED = True


def get_pcvr_data(
    data_dir: str,
    schema_path: str,
    batch_size: int = 256,
    valid_ratio: float = 0.1,
    train_ratio: float = 1.0,
    split_strategy: str = "row_group_tail",
    train_steps_per_sweep: int = 0,
    train_timestamp_start: int = 0,
    train_timestamp_end: int = 0,
    valid_timestamp_start: int = 0,
    valid_timestamp_end: int = 0,
    sampling_strategy: str = "step_random",
    num_workers: int = 16,
    buffer_batches: int = 1,
    shuffle_train: bool = True,
    seed: int = 42,
    clip_vocab: bool = True,
    seq_max_lens: dict[str, int] | None = None,
    data_pipeline_config: PCVRDataPipelineConfig | None = None,
    **kwargs: Any,
) -> tuple[DataLoader, DataLoader, Any]:
    ensure_torch_file_system_sharing_strategy()
    random.seed(seed)
    _validate_get_pcvr_data_args(split_strategy, sampling_strategy)

    row_groups = collect_pcvr_row_groups(data_dir)
    split_plan, train_timestamp_range, valid_timestamp_range, train_hash_filter, valid_hash_filter = _resolve_split_plan(
        row_groups,
        split_strategy=split_strategy,
        valid_ratio=valid_ratio,
        train_ratio=train_ratio,
        seed=seed,
        train_timestamp_start=train_timestamp_start,
        train_timestamp_end=train_timestamp_end,
        valid_timestamp_start=valid_timestamp_start,
        valid_timestamp_end=valid_timestamp_end,
    )
    effective_sampling_strategy = _effective_sampling_strategy(
        sampling_strategy,
        train_timestamp_range=train_timestamp_range,
        train_hash_filter=train_hash_filter,
    )
    effective_buffer_batches = _effective_buffer_batches(
        buffer_batches,
        sampling_strategy=effective_sampling_strategy,
    )
    _log_split_plan(split_plan, train_ratio=train_ratio)

    train_pipeline_config = data_pipeline_config or PCVRDataPipelineConfig()
    valid_pipeline_config = PCVRDataPipelineConfig(cache=train_pipeline_config.cache)
    planned_steps = int(kwargs.get("max_steps", 0) or 0)

    train_source_dataset = PCVRParquetDataset(
        parquet_path=data_dir,
        schema_path=schema_path,
        batch_size=batch_size,
        seq_max_lens=seq_max_lens,
        shuffle=shuffle_train,
        buffer_batches=effective_buffer_batches,
        row_group_range=split_plan.train_row_group_range,
        timestamp_range=train_timestamp_range,
        hash_split_filter=train_hash_filter,
        clip_vocab=clip_vocab,
        data_pipeline_config=train_pipeline_config,
        is_training=True,
        dataset_role="train",
    )
    train_dataset = _wrap_train_dataset(
        train_source_dataset,
        sampling_strategy=effective_sampling_strategy,
        train_timestamp_range=train_timestamp_range,
        train_steps_per_sweep=train_steps_per_sweep,
        planned_steps=planned_steps,
        seed=seed,
    )
    _enable_shared_cache_if_needed(
        train_dataset,
        num_workers=num_workers,
        data_pipeline_config=train_pipeline_config,
    )

    valid_dataset = PCVRParquetDataset(
        parquet_path=data_dir,
        schema_path=schema_path,
        batch_size=batch_size,
        seq_max_lens=seq_max_lens,
        shuffle=False,
        buffer_batches=0,
        row_group_range=split_plan.valid_row_group_range,
        timestamp_range=valid_timestamp_range,
        hash_split_filter=valid_hash_filter,
        clip_vocab=clip_vocab,
        data_pipeline_config=valid_pipeline_config,
        is_training=True,
        dataset_role="valid",
    )

    train_loader = _make_loader(train_dataset, num_workers=num_workers)
    valid_loader = _make_loader(valid_dataset, num_workers=0)
    logger.info(
        "Parquet train: {} rows, valid: {} rows, batch_size={}, "
        "buffer_batches={}, sampling_strategy={}",
        split_plan.train_rows,
        split_plan.valid_rows,
        batch_size,
        effective_buffer_batches,
        effective_sampling_strategy,
    )
    return train_loader, valid_loader, train_dataset


def _validate_get_pcvr_data_args(split_strategy: str, sampling_strategy: str) -> None:
    if split_strategy not in PCVR_DATA_SPLIT_STRATEGY_CHOICES:
        raise ValueError(f"unsupported split_strategy={split_strategy!r}")
    if sampling_strategy not in PCVR_DATA_SAMPLING_STRATEGY_CHOICES:
        raise ValueError(f"unsupported sampling_strategy={sampling_strategy!r}")


def _resolve_split_plan(
    row_groups: list[tuple[str, int, int]],
    *,
    split_strategy: str,
    valid_ratio: float,
    train_ratio: float,
    seed: int,
    train_timestamp_start: int,
    train_timestamp_end: int,
    valid_timestamp_start: int,
    valid_timestamp_end: int,
) -> tuple[
    PCVRRowGroupSplitPlan,
    PCVRTimestampRange | None,
    PCVRTimestampRange | None,
    PCVRHashSplitFilter | None,
    PCVRHashSplitFilter | None,
]:
    if split_strategy == "row_group_tail":
        return (
            plan_pcvr_row_group_split(
                row_groups,
                valid_ratio=valid_ratio,
                train_ratio=train_ratio,
            ),
            None,
            None,
            None,
            None,
        )
    if split_strategy == "timestamp_range":
        split_plan, train_range, valid_range = _timestamp_split_plan(
            row_groups,
            train_timestamp_start=train_timestamp_start,
            train_timestamp_end=train_timestamp_end,
            valid_timestamp_start=valid_timestamp_start,
            valid_timestamp_end=valid_timestamp_end,
        )
        return split_plan, train_range, valid_range, None, None
    split_plan, train_filter, valid_filter = _hash_split_plan(
        row_groups,
        split_strategy=split_strategy,
        valid_ratio=valid_ratio,
        train_ratio=train_ratio,
        seed=seed,
    )
    return split_plan, None, None, train_filter, valid_filter


def _hash_split_plan(
    row_groups: list[tuple[str, int, int]],
    *,
    split_strategy: str,
    valid_ratio: float,
    train_ratio: float,
    seed: int,
) -> tuple[PCVRRowGroupSplitPlan, PCVRHashSplitFilter, PCVRHashSplitFilter]:
    if not 0.0 < valid_ratio < 1.0:
        raise ValueError("hash split strategies require 0 < valid_ratio < 1")
    total_rows = sum(row_count for _path, _row_group_index, row_count in row_groups)
    estimated_valid_rows = max(1, round(total_rows * valid_ratio))
    estimated_train_rows = max(1, round((total_rows - estimated_valid_rows) * max(0.0, min(1.0, train_ratio))))
    split_plan = PCVRRowGroupSplitPlan(
        total_row_groups=len(row_groups),
        train_row_groups=len(row_groups),
        valid_row_groups=len(row_groups),
        train_row_group_range=(0, len(row_groups)),
        valid_row_group_range=(0, len(row_groups)),
        train_rows=estimated_train_rows,
        valid_rows=estimated_valid_rows,
        reuse_train_for_valid=False,
    )
    logger.info(
        "Hash split: strategy={}, valid_ratio={}, train_ratio={}, seed={}",
        split_strategy,
        valid_ratio,
        train_ratio,
        seed,
    )
    return (
        split_plan,
        PCVRHashSplitFilter(strategy=split_strategy, role="train", valid_ratio=valid_ratio, seed=seed),
        PCVRHashSplitFilter(strategy=split_strategy, role="valid", valid_ratio=valid_ratio, seed=seed),
    )


def _timestamp_split_plan(
    row_groups: list[tuple[str, int, int]],
    *,
    train_timestamp_start: int,
    train_timestamp_end: int,
    valid_timestamp_start: int,
    valid_timestamp_end: int,
) -> tuple[PCVRRowGroupSplitPlan, PCVRTimestampRange, PCVRTimestampRange]:
    train_range = normalize_pcvr_timestamp_range(
        train_timestamp_start,
        train_timestamp_end,
        label="train_timestamp_range",
    )
    valid_range = normalize_pcvr_timestamp_range(
        valid_timestamp_start,
        valid_timestamp_end,
        label="valid_timestamp_range",
    )
    if train_range is None or valid_range is None:
        raise ValueError(
            "timestamp_range split requires non-empty train and valid timestamp ranges"
        )
    split_plan = PCVRRowGroupSplitPlan(
        total_row_groups=len(row_groups),
        train_row_groups=len(row_groups),
        valid_row_groups=len(row_groups),
        train_row_group_range=(0, len(row_groups)),
        valid_row_group_range=(0, len(row_groups)),
        train_rows=count_pcvr_rows_in_timestamp_range(row_groups, train_range),
        valid_rows=count_pcvr_rows_in_timestamp_range(row_groups, valid_range),
        reuse_train_for_valid=False,
    )
    if split_plan.train_rows <= 0 or split_plan.valid_rows <= 0:
        raise ValueError(
            "timestamp_range split produced an empty train or valid dataset: "
            f"train_rows={split_plan.train_rows}, valid_rows={split_plan.valid_rows}"
        )
    logger.info(
        "Timestamp split: train={}, valid={}",
        pcvr_timestamp_range_to_dict(train_range),
        pcvr_timestamp_range_to_dict(valid_range),
    )
    return split_plan, train_range, valid_range


def _effective_sampling_strategy(
    sampling_strategy: str,
    *,
    train_timestamp_range: PCVRTimestampRange | None,
    train_hash_filter: PCVRHashSplitFilter | None,
) -> str:
    if sampling_strategy != "step_random" or (train_timestamp_range is None and train_hash_filter is None):
        return sampling_strategy
    logger.warning(
        "step_random sampling is not supported with row-level split filtering; "
        "using row_group_sweep for the training split"
    )
    return "row_group_sweep"


def _effective_buffer_batches(buffer_batches: int, *, sampling_strategy: str) -> int:
    value = int(buffer_batches)
    if sampling_strategy != "step_random" or value <= _STEP_RANDOM_BUFFER_BATCHES:
        return value
    logger.info(
        "step_random sampling already randomizes batch access; using buffer_batches={} "
        "instead of {} to avoid per-worker shuffle-buffer memory growth",
        _STEP_RANDOM_BUFFER_BATCHES,
        value,
    )
    return _STEP_RANDOM_BUFFER_BATCHES


def _log_split_plan(split_plan: PCVRRowGroupSplitPlan, *, train_ratio: float) -> None:
    if train_ratio < 1.0 and not split_plan.reuse_train_for_valid:
        logger.info(
            "train_ratio={}: using {} train Row Groups",
            train_ratio,
            split_plan.train_row_groups,
        )
    if split_plan.reuse_train_for_valid:
        logger.warning(
            "Single Row Group parquet detected; reusing the same Row Group for train and valid "
            "to keep smoke runs functional"
        )
    logger.info(
        "Row Group split: {} train ({} rows), {} valid ({} rows)",
        split_plan.train_row_groups,
        split_plan.train_rows,
        split_plan.valid_row_groups,
        split_plan.valid_rows,
    )


def _wrap_train_dataset(
    train_source_dataset: PCVRParquetDataset,
    *,
    sampling_strategy: str,
    train_timestamp_range: PCVRTimestampRange | None,
    train_steps_per_sweep: int,
    planned_steps: int,
    seed: int,
) -> Any:
    if sampling_strategy != "step_random" or train_timestamp_range is not None:
        return train_source_dataset
    from taac2026.infrastructure.data.step_dataset import PCVRStepDataset

    return PCVRStepDataset(
        train_source_dataset,
        train_steps_per_sweep=train_steps_per_sweep,
        planned_steps=planned_steps,
        seed=seed,
    )


def _enable_shared_cache_if_needed(
    train_dataset: Any,
    *,
    num_workers: int,
    data_pipeline_config: PCVRDataPipelineConfig,
) -> None:
    if num_workers <= 1 or not data_pipeline_config.cache.enabled:
        return
    train_dataset.pipeline.cache = train_dataset.build_shared_batch_cache(
        num_workers=num_workers
    )


def _make_loader(dataset: Any, *, num_workers: int) -> DataLoader:
    loader_kwargs: dict[str, Any] = {}
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2
    make_sampler = getattr(dataset, "make_sampler", None)
    if callable(make_sampler):
        loader_kwargs["sampler"] = make_sampler()
    return DataLoader(
        dataset,
        batch_size=None,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        **loader_kwargs,
    )


__all__ = [
    "BUCKET_BOUNDARIES",
    "NUM_TIME_BUCKETS",
    "FeatureSchema",
    "PCVRParquetDataset",
    "PCVRRowGroupSplitPlan",
    "PCVRTimestampRange",
    "build_pcvr_observed_schema_report",
    "collect_pcvr_row_groups",
    "count_pcvr_rows_in_timestamp_range",
    "ensure_torch_file_system_sharing_strategy",
    "get_pcvr_data",
    "normalize_pcvr_timestamp_range",
    "pcvr_timestamp_range_to_dict",
    "plan_pcvr_row_group_split",
]
