"""Step-indexed PCVR training dataset built on PyTorch map-style Dataset."""

from __future__ import annotations

import zlib
from collections import OrderedDict
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset, Sampler

from taac2026.domain.config import PCVRDataPipelineConfig
from taac2026.infrastructure.data.batches import (
    PCVRBatch,
    pcvr_batch_row_count,
    take_pcvr_rows,
)
from taac2026.infrastructure.data.cache import PCVRSharedBatchCache
from taac2026.infrastructure.data.pipeline import stable_pcvr_batch_seed_from_path_crc


_STEP_RANDOM_MAX_OPEN_PARQUET_FILES = 32
_DELEGATED_SOURCE_DATASET_ATTRIBUTES = frozenset(
    {
        "row_groups",
        "user_int_schema",
        "item_int_schema",
        "user_dense_schema",
        "item_dense_schema",
        "user_int_vocab_sizes",
        "item_int_vocab_sizes",
        "seq_domains",
        "seq_feature_ids",
        "seq_vocab_sizes",
        "seq_domain_vocab_sizes",
        "ts_fids",
        "sideinfo_fids",
    }
)


PCVRBatchKey = tuple[str, int, int]


@dataclass(frozen=True, slots=True)
class PCVRStepPlan:
    global_step: int
    batch_key: PCVRBatchKey


class PCVRStepIndexSampler(Sampler[int]):
    """Sampler that emits a fixed absolute optimizer-step window."""

    def __init__(self, *, step_count: int, start_step: int = 0) -> None:
        self.step_count = max(1, int(step_count))
        self.start_step = max(0, int(start_step))

    def __iter__(self) -> Iterator[int]:
        yield from range(self.start_step, self.start_step + self.step_count)

    def __len__(self) -> int:
        return self.step_count

    def set_start_step(self, start_step: int) -> None:
        self.start_step = max(0, int(start_step))


class PCVRStepDataset(Dataset[PCVRBatch]):
    """Map-style dataset where each item is one optimizer-step batch.

    The wrapped ``PCVRParquetDataset`` remains the schema and conversion engine
    for this first migration slice; PyTorch DataLoader now owns indexing,
    workers, prefetching, and pin-memory behavior.
    """

    def __init__(
        self,
        source_dataset: Any,
        *,
        train_steps_per_sweep: int = 0,
        planned_steps: int = 0,
        seed: int = 0,
    ) -> None:
        self.source_dataset = source_dataset
        self.batch_size = int(getattr(source_dataset, "batch_size", 1))
        self.buffer_batches = int(getattr(source_dataset, "buffer_batches", 0))
        self.shuffle = bool(getattr(source_dataset, "shuffle", True))
        self.is_training = bool(getattr(source_dataset, "is_training", True))
        self.dataset_role = str(getattr(source_dataset, "dataset_role", "train"))
        self.row_group_range = getattr(source_dataset, "row_group_range", None)
        self.timestamp_range = getattr(source_dataset, "timestamp_range", None)
        self.data_pipeline_config = getattr(
            source_dataset,
            "data_pipeline_config",
            PCVRDataPipelineConfig(),
        )
        self.pipeline = getattr(source_dataset, "pipeline", None)
        self.num_rows = int(getattr(source_dataset, "num_rows", 0))
        self.sampling_seed = int(seed)
        self.planned_steps = max(0, int(planned_steps))
        self.train_steps_per_sweep = max(0, int(train_steps_per_sweep))
        self._start_step = 0
        self._global_batch_keys_cache: tuple[PCVRBatchKey, ...] | None = None
        self._global_batch_cumulative_rows_cache: np.ndarray | None = None
        self._worker_parquet_files: OrderedDict[str, pq.ParquetFile] = OrderedDict()
        self._worker_row_group_iterators: dict[
            tuple[str, int], tuple[Iterator[pa.RecordBatch], int]
        ] = {}

    def __len__(self) -> int:
        return self.logical_sweep_steps()

    def __getattr__(self, name: str) -> Any:
        if name in _DELEGATED_SOURCE_DATASET_ATTRIBUTES:
            return getattr(self.source_dataset, name)
        raise AttributeError(name)

    @property
    def uses_step_random_sampling(self) -> bool:
        return True

    def set_start_step(self, start_step: int) -> None:
        self._start_step = max(0, int(start_step))

    def make_sampler(self) -> PCVRStepIndexSampler:
        return PCVRStepIndexSampler(
            step_count=self.logical_sweep_steps(),
            start_step=self._start_step,
        )

    def logical_sweep_steps(self) -> int:
        if self.train_steps_per_sweep > 0:
            return self.train_steps_per_sweep
        if self.planned_steps > 0:
            return self.planned_steps
        source_steps = getattr(self.source_dataset, "logical_sweep_steps", None)
        if callable(source_steps):
            return max(1, int(source_steps()))
        source_len = getattr(self.source_dataset, "__len__", None)
        if callable(source_len):
            return max(1, int(source_len()))
        return 1

    def __getitem__(self, index: int) -> PCVRBatch:
        global_step = int(index)
        step_plan = self.plan_step(global_step)
        file_path, row_group_index, batch_index = step_plan.batch_key
        path_crc = zlib.crc32(file_path.encode("utf-8"))
        generator = torch.Generator().manual_seed(
            stable_pcvr_batch_seed_from_path_crc(
                base_seed=self._base_random_seed(),
                worker_id=global_step,
                path_crc=path_crc,
                row_group_index=row_group_index,
                batch_index=batch_index,
            )
        )
        batch = self._read_base_batch(step_plan.batch_key, generator=generator)
        if batch is None:
            raise RuntimeError(
                "step-indexed sampling produced an empty timestamp-filtered batch; "
                "use scan sampling for timestamp_range splits"
            )
        return self._fit_step_batch(batch, global_step=global_step, generator=generator)

    def plan_step(self, global_step: int) -> PCVRStepPlan:
        return PCVRStepPlan(
            global_step=int(global_step),
            batch_key=self._batch_key_for_global_step(int(global_step)),
        )

    def iter_step_batch_keys(
        self,
        *,
        steps: int | None = None,
        start_step: int = 0,
    ) -> Iterator[PCVRBatchKey]:
        total_steps = self.logical_sweep_steps() if steps is None else max(0, int(steps))
        for offset in range(total_steps):
            yield self._batch_key_for_global_step(int(start_step) + offset)

    def build_shared_batch_cache(self, num_workers: int) -> PCVRSharedBatchCache:
        del num_workers
        tensor_specs = self.source_dataset.shared_cache_tensor_specs()
        cache_config = self.data_pipeline_config.cache
        policy = "lru" if cache_config.mode == "none" else cache_config.mode
        cache = PCVRSharedBatchCache(
            enabled=cache_config.enabled,
            max_batches=cache_config.max_batches,
            policy=policy,
            tensor_specs=tensor_specs,
            static_values={"_seq_domains": list(self.source_dataset.seq_domains)},
        )
        key_universe = self._global_batch_keys()
        if policy == "opt":
            cache.configure_access_trace(
                self.iter_step_batch_keys(steps=self._cache_trace_steps()),
                cyclic=False,
                key_universe=key_universe,
            )
        else:
            cache.configure_key_universe(key_universe)
        return cache

    def _read_base_batch(self, batch_key: PCVRBatchKey, *, generator: torch.Generator) -> PCVRBatch | None:
        if self.pipeline is None:
            return self.source_dataset.filter_batch_by_timestamp_range(self._materialize_base_batch(batch_key))
        return self.pipeline.materialize(
            batch_key,
            lambda batch_key=batch_key: self._materialize_base_batch(batch_key),
            generator=generator,
            preprocess=self.source_dataset.filter_batch_by_timestamp_range,
        )

    def _materialize_base_batch(self, batch_key: PCVRBatchKey) -> PCVRBatch:
        file_path, row_group_index, batch_index = batch_key
        record_batch = self.source_dataset.read_record_batch_for_key(
            parquet_files=self._worker_parquet_files,
            row_group_iterators=self._worker_row_group_iterators,
            file_path=file_path,
            row_group_index=row_group_index,
            batch_index=batch_index,
            reuse_iterators=False,
            max_open_parquet_files=_STEP_RANDOM_MAX_OPEN_PARQUET_FILES,
        )
        return self.source_dataset.convert_record_batch(record_batch)

    def _fit_step_batch(
        self,
        batch: PCVRBatch,
        *,
        global_step: int,
        generator: torch.Generator,
    ) -> PCVRBatch:
        row_count = pcvr_batch_row_count(batch)
        if row_count <= self.batch_size:
            return batch
        row_order = torch.randperm(row_count, generator=generator)
        start = (int(global_step) * self.batch_size) % row_count
        if start + self.batch_size <= row_count:
            selected = row_order[start : start + self.batch_size]
        else:
            selected = torch.cat(
                (row_order[start:], row_order[: (start + self.batch_size) % row_count])
            )
        return take_pcvr_rows(batch, selected)

    def _base_random_seed(self) -> int:
        configured_seed = getattr(self.data_pipeline_config, "seed", None)
        if configured_seed is not None:
            return int(configured_seed)
        return self.sampling_seed

    def _cache_trace_steps(self) -> int:
        if self.planned_steps > 0:
            return self.planned_steps
        return self.logical_sweep_steps()

    def _batch_key_for_global_step(self, global_step: int) -> PCVRBatchKey:
        batch_keys = self._global_batch_keys()
        cumulative_rows = self._global_batch_cumulative_rows()
        total_rows = int(cumulative_rows[-1]) if cumulative_rows.size else 0
        if not batch_keys or total_rows <= 0:
            raise IndexError("cannot sample from an empty PCVR step dataset")
        rng_seed = (
            self._base_random_seed()
            + int(global_step) * 1_000_003
            + len(batch_keys) * 101
        ) % (2**63 - 1)
        rng = np.random.default_rng(rng_seed)
        row_position = int(rng.integers(0, total_rows, dtype=np.int64))
        batch_position = int(np.searchsorted(cumulative_rows, row_position, side="right"))
        return batch_keys[batch_position]

    def _global_batch_keys(self) -> tuple[PCVRBatchKey, ...]:
        if self._global_batch_keys_cache is None:
            self._global_batch_keys_cache = tuple(
                self.source_dataset.global_batch_keys()
            )
        return self._global_batch_keys_cache

    def _global_batch_cumulative_rows(self) -> np.ndarray:
        if self._global_batch_cumulative_rows_cache is None:
            self._global_batch_cumulative_rows_cache = np.asarray(
                self.source_dataset.global_batch_cumulative_rows(),
                dtype=np.int64,
            )
        return self._global_batch_cumulative_rows_cache


__all__ = [
    "PCVRBatchKey",
    "PCVRStepDataset",
    "PCVRStepIndexSampler",
    "PCVRStepPlan",
]