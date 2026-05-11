from __future__ import annotations

import gc
import zlib
from collections import OrderedDict
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch.utils.data import IterableDataset

from taac2026.domain.config import PCVRDataPipelineConfig
from taac2026.infrastructure.data.batch_converter import (
    PCVRRecordBatchConverter,
    SEQUENCE_STATS_DIM,
    build_pcvr_column_plan,
)
from taac2026.infrastructure.data.observation import (
    PCVRTimestampRange,
    count_pcvr_rows_in_timestamp_range,
    pcvr_timestamp_range_to_dict,
)
from taac2026.infrastructure.data.pipeline import (
    PCVRBatchTransform,
    PCVRDataPipeline,
    PCVRMemoryBatchCache,
    PCVRSharedBatchCache,
    PCVRSharedTensorSpec,
    PCVRShuffleBuffer,
    build_pcvr_batch_transforms,
    stable_pcvr_batch_seed_from_path_crc,
)
from taac2026.infrastructure.data.schema_layout import load_pcvr_schema_layout
from taac2026.infrastructure.io.json import dumps
from taac2026.infrastructure.logging import logger


class PCVRParquetDataset(IterableDataset):
    def __init__(
        self,
        parquet_path: str,
        schema_path: str,
        batch_size: int = 256,
        seq_max_lens: dict[str, int] | None = None,
        shuffle: bool = True,
        buffer_batches: int = 1,
        row_group_range: tuple[int, int] | None = None,
        timestamp_range: PCVRTimestampRange | None = None,
        clip_vocab: bool = True,
        is_training: bool = True,
        data_pipeline_config: PCVRDataPipelineConfig | None = None,
        transforms: Sequence[PCVRBatchTransform] | None = None,
        dataset_role: str = "dataset",
    ) -> None:
        super().__init__()
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.buffer_batches = int(buffer_batches)
        self.clip_vocab = bool(clip_vocab)
        self.is_training = bool(is_training)
        self.dataset_role = str(dataset_role).strip() or "dataset"
        self.row_group_range = row_group_range
        self.timestamp_range = timestamp_range
        self.schema_path = Path(schema_path).expanduser().resolve()
        self.data_pipeline_config = data_pipeline_config or PCVRDataPipelineConfig()
        self.strict_time_filter_enabled = bool(
            self.is_training
            and self.data_pipeline_config.enabled
            and self.data_pipeline_config.strict_time_filter
        )

        self._parquet_files = _resolve_parquet_files(parquet_path)
        self._row_groups = _row_groups_for_files(self._parquet_files, row_group_range)
        self.num_rows = count_pcvr_rows_in_timestamp_range(
            self._row_groups,
            self.timestamp_range,
        )
        self._global_batch_keys: tuple[tuple[str, int, int], ...] | None = None
        self._global_batch_cumulative_rows: tuple[int, ...] | None = None
        self._actual_len: int | None = None

        self.layout = load_pcvr_schema_layout(self.schema_path, seq_max_lens or {})
        first_file = pq.ParquetFile(self._parquet_files[0])
        self.column_plan = build_pcvr_column_plan(
            self.layout,
            first_file.schema_arrow.names,
        )
        self.converter = PCVRRecordBatchConverter(
            layout=self.layout,
            column_plan=self.column_plan,
            batch_size=self.batch_size,
            clip_vocab=self.clip_vocab,
            is_training=self.is_training,
            strict_time_filter=self.strict_time_filter_enabled,
        )
        self._publish_schema_attributes()
        self._log_dataset_shape()
        self.pipeline = self._build_pipeline(transforms)

    def _publish_schema_attributes(self) -> None:
        self.user_int_schema = self.layout.user_int_schema
        self.item_int_schema = self.layout.item_int_schema
        self.user_dense_schema = self.layout.user_dense_schema
        self.item_dense_schema = self.layout.item_dense_schema
        self.user_int_vocab_sizes = self.layout.user_int_vocab_sizes
        self.item_int_vocab_sizes = self.layout.item_int_vocab_sizes
        self.seq_domains = self.layout.seq_domains
        self.seq_feature_ids = self.layout.seq_feature_ids
        self.seq_vocab_sizes = self.layout.seq_vocab_sizes
        self.seq_domain_vocab_sizes = self.layout.seq_domain_vocab_sizes
        self.ts_fids = self.layout.ts_fids
        self.sideinfo_fids = self.layout.sideinfo_fids
        self.sequence_max_lengths = self.layout.seq_maxlen

    def _log_dataset_shape(self) -> None:
        logger.info(
            "Loaded PCVR schema for {} dataset: path={}, row_groups={}, user_int={} ({} dims), "
            "item_int={} ({} dims), user_dense={} ({} dims), seq_domains={}",
            self.dataset_role,
            self.layout.schema_path,
            self.row_group_range if self.row_group_range is not None else "all",
            len(self.layout.user_int_cols),
            self.user_int_schema.total_dim,
            len(self.layout.item_int_cols),
            self.item_int_schema.total_dim,
            len(self.layout.user_dense_cols),
            self.user_dense_schema.total_dim,
            ", ".join(self.seq_domains) if self.seq_domains else "<none>",
        )
        logger.info("PCVR {} schema payload: {}", self.dataset_role, dumps(self.layout.raw_payload))
        logger.info(
            "PCVRParquetDataset: {} rows from {} file(s), batch_size={}, "
            "buffer_batches={}, shuffle={}, timestamp_range={}",
            self.num_rows,
            len(self._parquet_files),
            self.batch_size,
            self.buffer_batches,
            self.shuffle,
            pcvr_timestamp_range_to_dict(self.timestamp_range),
        )

    def _build_pipeline(
        self,
        transforms: Sequence[PCVRBatchTransform] | None,
    ) -> PCVRDataPipeline:
        pipeline_transforms = list(transforms or [])
        if self.is_training:
            pipeline_transforms.extend(build_pcvr_batch_transforms(self.data_pipeline_config))
        return PCVRDataPipeline(
            cache=PCVRMemoryBatchCache.from_config(self.data_pipeline_config.cache),
            transforms=tuple(pipeline_transforms),
        )

    def __len__(self) -> int:
        if self._actual_len is None:
            self._actual_len = self._count_iter_batches()
        return self._actual_len

    def _count_iter_batches(self) -> int:
        total = 0
        current_file_path: str | None = None
        current_parquet_file: pq.ParquetFile | None = None
        for file_path, row_group_index, _row_count in self._row_groups:
            if file_path != current_file_path:
                current_file_path = file_path
                current_parquet_file = pq.ParquetFile(file_path)
            if current_parquet_file is None:
                continue
            for _ in current_parquet_file.iter_batches(
                batch_size=self.batch_size,
                row_groups=[row_group_index],
                columns=["timestamp"],
            ):
                total += 1
        return max(1, total)

    def logical_sweep_steps(self) -> int:
        return max(1, len(self))

    @property
    def uses_step_random_sampling(self) -> bool:
        return False

    @property
    def row_groups(self) -> tuple[tuple[str, int, int], ...]:
        return tuple(self._row_groups)

    def record_batch_columns(self) -> list[str] | None:
        return self.column_plan.record_batch_columns()

    def convert_record_batch(self, batch: pa.RecordBatch) -> dict[str, Any]:
        return self.converter.convert(batch)

    def dump_oob_stats(self, path: str | None = None) -> None:
        self.converter.dump_oob_stats(path)

    def filter_batch_by_timestamp_range(
        self,
        batch_dict: dict[str, Any],
    ) -> dict[str, Any] | None:
        if self.timestamp_range is None:
            return batch_dict
        timestamps = batch_dict["timestamp"]
        row_count = int(timestamps.shape[0])
        mask = torch.ones(row_count, dtype=torch.bool)
        start, end = self.timestamp_range
        if start is not None:
            mask &= timestamps >= start
        if end is not None:
            mask &= timestamps < end
        if bool(mask.all()):
            return batch_dict
        if not bool(mask.any()):
            return None
        keep = mask.tolist()
        return {
            key: _filter_value_by_mask(value, mask, keep, row_count)
            for key, value in batch_dict.items()
        }

    def iter_base_batch_keys(
        self,
        rg_list: Sequence[tuple[str, int, int]],
    ) -> Iterator[tuple[str, int, int]]:
        for file_path, row_group_index, row_count in rg_list:
            batch_count = (int(row_count) + self.batch_size - 1) // self.batch_size
            for batch_index in range(batch_count):
                yield (file_path, row_group_index, batch_index)

    def global_batch_keys(self) -> tuple[tuple[str, int, int], ...]:
        if self._global_batch_keys is None:
            self._global_batch_keys = tuple(self.iter_base_batch_keys(self._row_groups))
        return self._global_batch_keys

    def global_batch_cumulative_rows(self) -> tuple[int, ...]:
        if self._global_batch_cumulative_rows is None:
            self._global_batch_cumulative_rows = tuple(
                _cumulative_batch_rows(self._row_groups, batch_size=self.batch_size)
            )
        return self._global_batch_cumulative_rows

    def read_record_batch_for_key(
        self,
        *,
        parquet_files: dict[str, pq.ParquetFile] | OrderedDict[str, pq.ParquetFile],
        row_group_iterators: dict[tuple[str, int], tuple[Iterator[pa.RecordBatch], int]],
        file_path: str,
        row_group_index: int,
        batch_index: int,
        reuse_iterators: bool = True,
        max_open_parquet_files: int = 0,
    ) -> pa.RecordBatch:
        parquet_file = _cached_parquet_file(
            parquet_files,
            row_group_iterators,
            file_path=file_path,
            max_open_parquet_files=max_open_parquet_files,
        )
        if not reuse_iterators:
            return self._read_record_batch_without_reuse(
                parquet_file,
                file_path=file_path,
                row_group_index=row_group_index,
                batch_index=batch_index,
            )
        return self._read_record_batch_with_reuse(
            parquet_file,
            row_group_iterators,
            file_path=file_path,
            row_group_index=row_group_index,
            batch_index=batch_index,
        )

    def _read_record_batch_without_reuse(
        self,
        parquet_file: pq.ParquetFile,
        *,
        file_path: str,
        row_group_index: int,
        batch_index: int,
    ) -> pa.RecordBatch:
        for current_index, record_batch in enumerate(
            parquet_file.iter_batches(
                batch_size=self.batch_size,
                row_groups=[row_group_index],
                columns=self.record_batch_columns(),
            )
        ):
            if current_index == batch_index:
                return record_batch
        raise IndexError(_batch_index_error(batch_index, row_group_index, file_path))

    def _read_record_batch_with_reuse(
        self,
        parquet_file: pq.ParquetFile,
        row_group_iterators: dict[tuple[str, int], tuple[Iterator[pa.RecordBatch], int]],
        *,
        file_path: str,
        row_group_index: int,
        batch_index: int,
    ) -> pa.RecordBatch:
        iterator_key = (file_path, row_group_index)
        iterator_state = row_group_iterators.get(iterator_key)
        if iterator_state is None:
            iterator = self._iter_record_batches(parquet_file, row_group_index)
            next_batch_index = 0
        else:
            iterator, next_batch_index = iterator_state

        if batch_index < next_batch_index:
            iterator = self._iter_record_batches(parquet_file, row_group_index)
            next_batch_index = 0

        while next_batch_index <= batch_index:
            try:
                record_batch = next(iterator)
            except StopIteration as exc:
                raise IndexError(
                    _batch_index_error(batch_index, row_group_index, file_path)
                ) from exc
            if next_batch_index == batch_index:
                row_group_iterators[iterator_key] = (iterator, next_batch_index + 1)
                return record_batch
            next_batch_index += 1
        raise RuntimeError("failed to materialize requested parquet batch")

    def _iter_record_batches(
        self,
        parquet_file: pq.ParquetFile,
        row_group_index: int,
    ) -> Iterator[pa.RecordBatch]:
        return parquet_file.iter_batches(
            batch_size=self.batch_size,
            row_groups=[row_group_index],
            columns=self.record_batch_columns(),
        )

    def shared_cache_tensor_specs(self) -> dict[str, PCVRSharedTensorSpec]:
        tensor_specs = _base_tensor_specs(
            batch_size=self.batch_size,
            user_int_dim=self.user_int_schema.total_dim,
            user_dense_dim=self.user_dense_schema.total_dim,
            item_int_dim=self.item_int_schema.total_dim,
        )
        for domain in self.seq_domains:
            tensor_specs[domain] = PCVRSharedTensorSpec(
                shape=(
                    self.batch_size,
                    len(self.sideinfo_fids[domain]),
                    self.sequence_max_lengths[domain],
                ),
                dtype=torch.long,
            )
            tensor_specs[f"{domain}_len"] = PCVRSharedTensorSpec(
                shape=(self.batch_size,),
                dtype=torch.long,
            )
            tensor_specs[f"{domain}_time_bucket"] = PCVRSharedTensorSpec(
                shape=(self.batch_size, self.sequence_max_lengths[domain]),
                dtype=torch.long,
            )
            tensor_specs[f"{domain}_stats"] = PCVRSharedTensorSpec(
                shape=(self.batch_size, SEQUENCE_STATS_DIM),
                dtype=torch.float32,
            )
        return tensor_specs

    def build_shared_batch_cache(self, num_workers: int) -> PCVRSharedBatchCache:
        del num_workers
        cache_config = self.data_pipeline_config.cache
        policy = "lru" if cache_config.mode == "none" else cache_config.mode
        cache = PCVRSharedBatchCache(
            enabled=cache_config.enabled,
            max_batches=cache_config.max_batches,
            policy=policy,
            tensor_specs=self.shared_cache_tensor_specs(),
            static_values={"_seq_domains": list(self.seq_domains)},
        )
        if policy == "opt":
            cache.configure_access_trace(
                self.iter_base_batch_keys(self._row_groups),
                cyclic=True,
                key_universe=self.global_batch_keys(),
            )
        else:
            cache.configure_key_universe(self.global_batch_keys())
        return cache

    def __iter__(self) -> Iterator[dict[str, Any]]:
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        row_groups = _row_groups_for_worker(self._row_groups, worker_info)
        self._configure_worker_cache(row_groups)

        shuffle_buffer = PCVRShuffleBuffer(
            batch_size=self.batch_size,
            buffer_batches=self.buffer_batches,
            shuffle=self.shuffle,
        )
        needs_generator = self.pipeline.requires_generator or shuffle_buffer.requires_generator
        base_seed = self.data_pipeline_config.seed if self.data_pipeline_config.seed is not None else 0
        last_generator: torch.Generator | None = None

        for batch_context in self._iter_scan_batches(row_groups):
            generator = None
            if needs_generator:
                generator = _batch_generator(
                    base_seed=base_seed,
                    worker_id=worker_id,
                    path_crc=batch_context.path_crc,
                    row_group_index=batch_context.row_group_index,
                    batch_index=batch_context.batch_index,
                )
                last_generator = generator
            batch_dict = self.pipeline.materialize(
                batch_context.cache_key,
                lambda record_batch=batch_context.record_batch: self.convert_record_batch(record_batch),
                generator=generator,
                preprocess=self.filter_batch_by_timestamp_range,
            )
            if batch_dict is None:
                continue
            yield from shuffle_buffer.push(batch_dict, generator=generator)

        yield from shuffle_buffer.flush(generator=last_generator)
        del shuffle_buffer
        gc.collect()

    def _configure_worker_cache(self, row_groups: Sequence[tuple[str, int, int]]) -> None:
        if isinstance(self.pipeline.cache, PCVRSharedBatchCache):
            return
        cache_policy = getattr(self.pipeline.cache, "policy", "lru")
        if cache_policy == "opt":
            self.pipeline.configure_access_trace(
                self.iter_base_batch_keys(row_groups),
                cyclic=True,
                key_universe=self.global_batch_keys(),
            )
        else:
            self.pipeline.configure_key_universe(self.global_batch_keys())

    def _iter_scan_batches(
        self,
        row_groups: Sequence[tuple[str, int, int]],
    ) -> Iterator[_ScanBatchContext]:
        current_file_path: str | None = None
        current_parquet_file: pq.ParquetFile | None = None
        current_path_crc = 0
        for file_path, row_group_index, _row_count in row_groups:
            if file_path != current_file_path:
                current_file_path = file_path
                current_parquet_file = pq.ParquetFile(file_path)
                current_path_crc = zlib.crc32(file_path.encode("utf-8"))
            if current_parquet_file is None:
                continue
            for batch_index, record_batch in enumerate(
                self._iter_record_batches(current_parquet_file, row_group_index)
            ):
                yield _ScanBatchContext(
                    cache_key=(file_path, row_group_index, batch_index),
                    record_batch=record_batch,
                    path_crc=current_path_crc,
                    row_group_index=row_group_index,
                    batch_index=batch_index,
                )


class _ScanBatchContext:
    def __init__(
        self,
        *,
        cache_key: tuple[str, int, int],
        record_batch: pa.RecordBatch,
        path_crc: int,
        row_group_index: int,
        batch_index: int,
    ) -> None:
        self.cache_key = cache_key
        self.record_batch = record_batch
        self.path_crc = path_crc
        self.row_group_index = row_group_index
        self.batch_index = batch_index


def _resolve_parquet_files(parquet_path: str) -> list[str]:
    parquet_root = Path(parquet_path).expanduser()
    if not parquet_root.is_dir():
        return [str(parquet_root)]
    files = sorted(parquet_root.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No .parquet files in {parquet_path}")
    return [str(path) for path in files]


def _row_groups_for_files(
    parquet_files: Sequence[str],
    row_group_range: tuple[int, int] | None,
) -> list[tuple[str, int, int]]:
    row_groups: list[tuple[str, int, int]] = []
    for file_path in parquet_files:
        parquet_file = pq.ParquetFile(file_path)
        for row_group_index in range(parquet_file.metadata.num_row_groups):
            row_groups.append(
                (
                    file_path,
                    row_group_index,
                    parquet_file.metadata.row_group(row_group_index).num_rows,
                )
            )
    if row_group_range is None:
        return row_groups
    start, end = row_group_range
    return row_groups[start:end]


def _row_groups_for_worker(
    row_groups: Sequence[tuple[str, int, int]],
    worker_info: Any,
) -> Sequence[tuple[str, int, int]]:
    if worker_info is None or worker_info.num_workers <= 1:
        return row_groups
    return [
        row_group
        for index, row_group in enumerate(row_groups)
        if index % worker_info.num_workers == worker_info.id
    ]


def _filter_value_by_mask(
    value: Any,
    mask: torch.Tensor,
    keep: list[bool],
    row_count: int,
) -> Any:
    if isinstance(value, torch.Tensor) and value.ndim > 0 and value.shape[0] == row_count:
        return value[mask]
    if isinstance(value, list) and len(value) == row_count:
        return [item for item, should_keep in zip(value, keep, strict=True) if should_keep]
    return value


def _cached_parquet_file(
    parquet_files: dict[str, pq.ParquetFile] | OrderedDict[str, pq.ParquetFile],
    row_group_iterators: dict[tuple[str, int], tuple[Iterator[pa.RecordBatch], int]],
    *,
    file_path: str,
    max_open_parquet_files: int,
) -> pq.ParquetFile:
    parquet_file = parquet_files.get(file_path)
    if parquet_file is None:
        parquet_file = pq.ParquetFile(file_path)
        parquet_files[file_path] = parquet_file
        _evict_parquet_files(parquet_files, row_group_iterators, max_open_parquet_files)
        return parquet_file
    if isinstance(parquet_files, OrderedDict):
        parquet_files.move_to_end(file_path)
    return parquet_file


def _evict_parquet_files(
    parquet_files: dict[str, pq.ParquetFile] | OrderedDict[str, pq.ParquetFile],
    row_group_iterators: dict[tuple[str, int], tuple[Iterator[pa.RecordBatch], int]],
    max_open_parquet_files: int,
) -> None:
    if max_open_parquet_files <= 0 or not isinstance(parquet_files, OrderedDict):
        return
    while len(parquet_files) > max_open_parquet_files:
        evicted_path, _evicted_file = parquet_files.popitem(last=False)
        for iterator_key in list(row_group_iterators):
            if iterator_key[0] == evicted_path:
                row_group_iterators.pop(iterator_key, None)


def _cumulative_batch_rows(
    row_groups: Sequence[tuple[str, int, int]],
    *,
    batch_size: int,
) -> Iterator[int]:
    running_rows = 0
    for _file_path, _row_group_index, row_count in row_groups:
        rows_left = int(row_count)
        while rows_left > 0:
            batch_rows = min(batch_size, rows_left)
            running_rows += batch_rows
            yield running_rows
            rows_left -= batch_rows


def _base_tensor_specs(
    *,
    batch_size: int,
    user_int_dim: int,
    user_dense_dim: int,
    item_int_dim: int,
) -> dict[str, PCVRSharedTensorSpec]:
    return {
        "user_int_feats": PCVRSharedTensorSpec(
            shape=(batch_size, user_int_dim),
            dtype=torch.long,
        ),
        "user_int_missing_mask": PCVRSharedTensorSpec(
            shape=(batch_size, user_int_dim),
            dtype=torch.bool,
        ),
        "user_dense_feats": PCVRSharedTensorSpec(
            shape=(batch_size, user_dense_dim),
            dtype=torch.float32,
        ),
        "user_dense_missing_mask": PCVRSharedTensorSpec(
            shape=(batch_size, user_dense_dim),
            dtype=torch.bool,
        ),
        "item_int_feats": PCVRSharedTensorSpec(
            shape=(batch_size, item_int_dim),
            dtype=torch.long,
        ),
        "item_int_missing_mask": PCVRSharedTensorSpec(
            shape=(batch_size, item_int_dim),
            dtype=torch.bool,
        ),
        "item_dense_feats": PCVRSharedTensorSpec(
            shape=(batch_size, 0),
            dtype=torch.float32,
        ),
        "item_dense_missing_mask": PCVRSharedTensorSpec(
            shape=(batch_size, 0),
            dtype=torch.bool,
        ),
        "label": PCVRSharedTensorSpec(shape=(batch_size,), dtype=torch.long),
        "timestamp": PCVRSharedTensorSpec(shape=(batch_size,), dtype=torch.long),
    }


def _batch_generator(
    *,
    base_seed: int,
    worker_id: int,
    path_crc: int,
    row_group_index: int,
    batch_index: int,
) -> torch.Generator:
    seed = stable_pcvr_batch_seed_from_path_crc(
        base_seed=base_seed,
        worker_id=worker_id,
        path_crc=path_crc,
        row_group_index=row_group_index,
        batch_index=batch_index,
    )
    return torch.Generator().manual_seed(seed)


def _batch_index_error(batch_index: int, row_group_index: int, file_path: str) -> str:
    return f"batch_index {batch_index} out of range for row group {row_group_index} in {file_path}"
