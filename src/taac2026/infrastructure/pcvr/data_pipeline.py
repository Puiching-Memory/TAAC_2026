"""Composable PCVR data pipeline components."""

from __future__ import annotations

import zlib
from collections import OrderedDict
from collections.abc import Callable, Hashable, Iterator
from typing import Any, Protocol

import torch

from taac2026.infrastructure.pcvr.config import (
    PCVRDataCacheConfig,
    PCVRDataPipelineConfig,
    PCVRDataTransformConfig,
    PCVRDomainDropoutConfig,
    PCVRFeatureMaskConfig,
    PCVRSequenceCropConfig,
)


PCVRBatch = dict[str, Any]
PCVRBatchFactory = Callable[[], PCVRBatch]


class PCVRBatchTransform(Protocol):
    def __call__(self, batch: PCVRBatch, *, generator: torch.Generator) -> PCVRBatch:
        """Return a transformed PCVR batch."""


def pcvr_batch_row_count(batch: PCVRBatch) -> int:
    label = batch.get("label")
    if isinstance(label, torch.Tensor) and label.ndim > 0:
        return int(label.shape[0])
    for value in batch.values():
        if isinstance(value, torch.Tensor) and value.ndim > 0:
            return int(value.shape[0])
    return 0


def clone_pcvr_batch(batch: PCVRBatch) -> PCVRBatch:
    cloned: PCVRBatch = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            cloned[key] = value.clone()
        elif isinstance(value, list):
            cloned[key] = list(value)
        elif isinstance(value, tuple):
            cloned[key] = tuple(value)
        else:
            cloned[key] = value
    return cloned


def repeat_pcvr_rows(batch: PCVRBatch, repeats: int) -> PCVRBatch:
    if repeats <= 1:
        return clone_pcvr_batch(batch)

    row_count = pcvr_batch_row_count(batch)
    repeated: PCVRBatch = {}
    for key, value in batch.items():
        if key == "_seq_domains":
            repeated[key] = list(value)
        elif (
            isinstance(value, torch.Tensor)
            and value.ndim > 0
            and value.shape[0] == row_count
        ):
            repeated[key] = value.repeat_interleave(repeats, dim=0)
        elif isinstance(value, list) and len(value) == row_count:
            repeated[key] = [item for item in value for _repeat_index in range(repeats)]
        else:
            repeated[key] = clone_pcvr_batch({key: value})[key]
    return repeated


def take_pcvr_rows(batch: PCVRBatch, row_indices: torch.Tensor) -> PCVRBatch:
    row_count = pcvr_batch_row_count(batch)
    indices = row_indices.detach().cpu().tolist()
    sliced: PCVRBatch = {}
    for key, value in batch.items():
        if key == "_seq_domains":
            sliced[key] = list(value)
        elif (
            isinstance(value, torch.Tensor)
            and value.ndim > 0
            and value.shape[0] == row_count
        ):
            sliced[key] = value.index_select(0, row_indices.to(value.device))
        elif isinstance(value, list) and len(value) == row_count:
            sliced[key] = [value[index] for index in indices]
        else:
            sliced[key] = clone_pcvr_batch({key: value})[key]
    return sliced


def concat_pcvr_batches(batches: list[PCVRBatch]) -> PCVRBatch:
    if not batches:
        return {}

    row_counts = [pcvr_batch_row_count(batch) for batch in batches]
    merged: PCVRBatch = {}
    for key, first_value in batches[0].items():
        if key == "_seq_domains":
            merged[key] = list(first_value)
        elif isinstance(first_value, torch.Tensor):
            merged[key] = torch.cat([batch[key] for batch in batches], dim=0)
        elif isinstance(first_value, list) and all(
            len(batch[key]) == row_count
            for batch, row_count in zip(batches, row_counts, strict=True)
        ):
            merged[key] = [item for batch in batches for item in batch[key]]
        else:
            merged[key] = clone_pcvr_batch({key: first_value})[key]
    return merged


class PCVRMemoryBatchCache:
    """Small per-process LRU cache for converted base PCVR batches."""

    def __init__(self, *, enabled: bool = False, max_batches: int = 0) -> None:
        self.enabled = enabled
        self.max_batches = max(0, int(max_batches))
        self._items: OrderedDict[Hashable, PCVRBatch] = OrderedDict()

    @classmethod
    def from_config(cls, config: PCVRDataCacheConfig | None) -> PCVRMemoryBatchCache:
        if config is None:
            return cls()
        return cls(enabled=config.enabled, max_batches=config.max_batches)

    def get(self, key: Hashable) -> PCVRBatch | None:
        if not self.enabled:
            return None
        batch = self._items.get(key)
        if batch is None:
            return None
        self._items.move_to_end(key)
        return clone_pcvr_batch(batch)

    def put(self, key: Hashable, batch: PCVRBatch) -> None:
        if not self.enabled:
            return
        self._items[key] = clone_pcvr_batch(batch)
        self._items.move_to_end(key)
        while self.max_batches > 0 and len(self._items) > self.max_batches:
            self._items.popitem(last=False)

    def __len__(self) -> int:
        return len(self._items)


class PCVRDataPipeline:
    """Compose cache and row-level transforms around Parquet batch conversion."""

    def __init__(
        self,
        *,
        cache: PCVRMemoryBatchCache | None = None,
        transforms: list[PCVRBatchTransform] | tuple[PCVRBatchTransform, ...] = (),
    ) -> None:
        self.cache = cache or PCVRMemoryBatchCache()
        self.transforms = tuple(transforms)

    @property
    def requires_generator(self) -> bool:
        return bool(self.transforms)

    def read_base_batch(self, key: Hashable, factory: PCVRBatchFactory) -> PCVRBatch:
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        batch = factory()
        self.cache.put(key, batch)
        return batch

    def apply_transforms(
        self, batch: PCVRBatch, *, generator: torch.Generator | None = None
    ) -> PCVRBatch:
        if not self.transforms:
            return batch
        if generator is None:
            generator = torch.Generator()
        transformed = batch
        for transform in self.transforms:
            transformed = transform(transformed, generator=generator)
        return transformed


class PCVRShuffleBuffer:
    """Row-level shuffle buffer for pre-batched PCVR tensor dictionaries."""

    def __init__(self, *, batch_size: int, buffer_batches: int, shuffle: bool) -> None:
        self.batch_size = int(batch_size)
        self.buffer_batches = int(buffer_batches)
        self.shuffle = shuffle
        self._buffer: list[PCVRBatch] = []

    @property
    def requires_generator(self) -> bool:
        return self.shuffle and self.buffer_batches > 1

    def push(
        self, batch: PCVRBatch, *, generator: torch.Generator | None = None
    ) -> Iterator[PCVRBatch]:
        if self.shuffle and self.buffer_batches > 1:
            self._buffer.append(batch)
            if len(self._buffer) >= self.buffer_batches:
                yield from self.flush(generator=generator)
        else:
            yield batch

    def flush(self, *, generator: torch.Generator | None = None) -> Iterator[PCVRBatch]:
        if not self._buffer:
            return
        merged = concat_pcvr_batches(self._buffer)
        total_rows = pcvr_batch_row_count(merged)
        if self.shuffle:
            row_order = torch.randperm(total_rows, generator=generator)
        else:
            row_order = torch.arange(total_rows)
        for start in range(0, total_rows, self.batch_size):
            end = min(start + self.batch_size, total_rows)
            yield take_pcvr_rows(merged, row_order[start:end])
        self._buffer.clear()


class PCVRSequenceCropTransform:
    """Create time-safe sequence views from each batch."""

    def __init__(self, config: PCVRSequenceCropConfig) -> None:
        self.config = config

    def __call__(self, batch: PCVRBatch, *, generator: torch.Generator) -> PCVRBatch:
        if not self.config.enabled:
            return clone_pcvr_batch(batch)

        augmented = repeat_pcvr_rows(batch, max(1, int(self.config.views_per_row)))
        self._apply_sequence_crop(augmented, generator=generator)
        return augmented

    def _apply_sequence_crop(
        self, batch: PCVRBatch, *, generator: torch.Generator
    ) -> None:
        for domain in batch.get("_seq_domains", []):
            sequence = batch.get(domain)
            lengths = batch.get(f"{domain}_len")
            time_buckets = batch.get(f"{domain}_time_bucket")
            if not isinstance(sequence, torch.Tensor) or not isinstance(
                lengths, torch.Tensor
            ):
                continue
            if not isinstance(time_buckets, torch.Tensor):
                time_buckets = torch.zeros(
                    sequence.shape[0], sequence.shape[2], dtype=torch.long
                )
                batch[f"{domain}_time_bucket"] = time_buckets
            self._crop_domain(sequence, lengths, time_buckets, generator=generator)

    def _crop_domain(
        self,
        sequence: torch.Tensor,
        lengths: torch.Tensor,
        time_buckets: torch.Tensor,
        *,
        generator: torch.Generator,
    ) -> None:
        max_len = int(sequence.shape[2])
        if max_len <= 0 or sequence.shape[0] == 0:
            return

        row_count = int(sequence.shape[0])
        device = sequence.device
        length_values = lengths.clamp(min=0, max=max_len)
        active_rows = length_values > 0
        if not bool(active_rows.any()):
            return

        min_len_config = max(1, int(self.config.seq_window_min_len))
        min_lengths = torch.minimum(
            torch.full_like(length_values, min_len_config), length_values
        ).clamp(min=1)

        if self.config.seq_window_mode == "tail":
            window_lengths = length_values
        else:
            span = (length_values - min_lengths + 1).clamp(min=1)
            draws = torch.floor(
                torch.rand(row_count, generator=generator, device=device)
                * span.to(torch.float32)
            ).to(length_values.dtype)
            window_lengths = torch.where(
                active_rows, min_lengths + draws, torch.zeros_like(length_values)
            )

        if self.config.seq_window_mode == "rolling":
            max_starts = (length_values - window_lengths).clamp(min=0)
            start_draws = torch.floor(
                torch.rand(row_count, generator=generator, device=device)
                * (max_starts + 1).to(torch.float32)
            ).to(length_values.dtype)
            starts = torch.where(
                active_rows, start_draws, torch.zeros_like(length_values)
            )
        else:
            starts = (length_values - window_lengths).clamp(min=0)

        positions = starts.view(-1, 1) + torch.arange(max_len, device=device).view(
            1, -1
        )
        positions = positions.clamp(max=max_len - 1).to(torch.long)
        valid_positions = torch.arange(max_len, device=device).view(
            1, -1
        ) < window_lengths.view(-1, 1)

        sequence_positions = positions.view(row_count, 1, max_len).expand(
            row_count, sequence.shape[1], max_len
        )
        cropped_sequence = sequence.gather(2, sequence_positions)
        sequence.copy_(
            cropped_sequence.masked_fill(
                ~valid_positions.view(row_count, 1, max_len), 0
            )
        )

        cropped_time = time_buckets.gather(1, positions)
        time_buckets.copy_(cropped_time.masked_fill(~valid_positions, 0))
        lengths.copy_(window_lengths.to(lengths.dtype))


class PCVRDomainDropoutTransform:
    """Drop whole sequence domains for selected rows."""

    def __init__(self, config: PCVRDomainDropoutConfig) -> None:
        self.config = config

    def __call__(self, batch: PCVRBatch, *, generator: torch.Generator) -> PCVRBatch:
        if not self.config.enabled:
            return clone_pcvr_batch(batch)
        augmented = clone_pcvr_batch(batch)
        self._apply_domain_dropout(augmented, generator=generator)
        return augmented

    def _apply_domain_dropout(
        self, batch: PCVRBatch, *, generator: torch.Generator
    ) -> None:
        probability = float(self.config.probability)
        if probability <= 0.0:
            return
        for domain in batch.get("_seq_domains", []):
            sequence = batch.get(domain)
            lengths = batch.get(f"{domain}_len")
            time_buckets = batch.get(f"{domain}_time_bucket")
            if not isinstance(sequence, torch.Tensor) or not isinstance(
                lengths, torch.Tensor
            ):
                continue
            drop_mask = torch.rand(sequence.shape[0], generator=generator) < probability
            if not bool(drop_mask.any()):
                continue
            sequence[drop_mask] = 0
            lengths[drop_mask] = 0
            if isinstance(time_buckets, torch.Tensor):
                time_buckets[drop_mask] = 0


class PCVRFeatureMaskTransform:
    """Mask sparse features and sequence events in a batch."""

    def __init__(self, config: PCVRFeatureMaskConfig) -> None:
        self.config = config

    def __call__(self, batch: PCVRBatch, *, generator: torch.Generator) -> PCVRBatch:
        if not self.config.enabled:
            return clone_pcvr_batch(batch)
        augmented = clone_pcvr_batch(batch)
        self._apply_feature_masking(augmented, generator=generator)
        return augmented

    def _apply_feature_masking(
        self, batch: PCVRBatch, *, generator: torch.Generator
    ) -> None:
        probability = float(self.config.probability)
        if probability <= 0.0:
            return

        for feature_key in ("user_int_feats", "item_int_feats"):
            features = batch.get(feature_key)
            if isinstance(features, torch.Tensor) and features.numel() > 0:
                mask = torch.rand(features.shape, generator=generator) < probability
                features[mask] = 0

        for domain in batch.get("_seq_domains", []):
            sequence = batch.get(domain)
            lengths = batch.get(f"{domain}_len")
            time_buckets = batch.get(f"{domain}_time_bucket")
            if not isinstance(sequence, torch.Tensor) or not isinstance(
                lengths, torch.Tensor
            ):
                continue
            valid_positions = torch.arange(sequence.shape[2]).view(
                1, -1
            ) < lengths.view(-1, 1)
            event_mask = (
                torch.rand(valid_positions.shape, generator=generator) < probability
            ) & valid_positions
            if bool(event_mask.any()):
                expanded_mask = event_mask.unsqueeze(1).expand_as(sequence)
                sequence[expanded_mask] = 0
                if isinstance(time_buckets, torch.Tensor):
                    time_buckets[event_mask] = 0
                self._compact_domain(
                    sequence,
                    lengths,
                    time_buckets if isinstance(time_buckets, torch.Tensor) else None,
                    row_mask=event_mask.any(dim=1),
                )

    def _compact_domain(
        self,
        sequence: torch.Tensor,
        lengths: torch.Tensor,
        time_buckets: torch.Tensor | None,
        *,
        row_mask: torch.Tensor | None = None,
    ) -> None:
        max_len = int(sequence.shape[2])
        valid_positions = (sequence > 0).any(dim=1)
        if row_mask is None:
            row_indices = range(sequence.shape[0])
        else:
            row_indices = torch.nonzero(row_mask, as_tuple=False).flatten().tolist()
        for row_index in row_indices:
            row_positions = torch.nonzero(
                valid_positions[row_index], as_tuple=False
            ).flatten()
            new_len = min(int(row_positions.numel()), max_len)
            if new_len <= 0:
                sequence[row_index].zero_()
                lengths[row_index] = 0
                if time_buckets is not None:
                    time_buckets[row_index].zero_()
                continue
            selected_sequence = sequence[row_index, :, row_positions[:new_len]].clone()
            sequence[row_index].zero_()
            sequence[row_index, :, :new_len] = selected_sequence
            lengths[row_index] = new_len
            if time_buckets is not None:
                selected_time = time_buckets[row_index, row_positions[:new_len]].clone()
                time_buckets[row_index].zero_()
                time_buckets[row_index, :new_len] = selected_time


def build_pcvr_batch_transform(config: PCVRDataTransformConfig) -> PCVRBatchTransform:
    if isinstance(config, PCVRSequenceCropConfig):
        return PCVRSequenceCropTransform(config)
    if isinstance(config, PCVRFeatureMaskConfig):
        return PCVRFeatureMaskTransform(config)
    if isinstance(config, PCVRDomainDropoutConfig):
        return PCVRDomainDropoutTransform(config)
    raise TypeError(f"unsupported PCVR data transform config: {type(config).__name__}")


def build_pcvr_batch_transforms(
    config: PCVRDataPipelineConfig,
) -> tuple[PCVRBatchTransform, ...]:
    return tuple(
        build_pcvr_batch_transform(transform)
        for transform in config.transforms
        if transform.enabled and not _is_noop_pcvr_batch_transform_config(transform)
    )


def _is_noop_pcvr_batch_transform_config(config: PCVRDataTransformConfig) -> bool:
    if isinstance(config, PCVRSequenceCropConfig):
        return config.views_per_row <= 1 and config.seq_window_mode == "tail"
    if isinstance(config, (PCVRFeatureMaskConfig, PCVRDomainDropoutConfig)):
        return float(config.probability) <= 0.0
    return False


def stable_pcvr_batch_seed_from_path_crc(
    *,
    base_seed: int,
    worker_id: int,
    path_crc: int,
    row_group_index: int,
    batch_index: int,
) -> int:
    seed = (
        int(base_seed)
        + worker_id * 1_000_003
        + int(path_crc)
        + row_group_index * 10_007
        + batch_index * 101
    )
    return seed % (2**63 - 1)


def stable_pcvr_batch_seed(
    *,
    base_seed: int,
    worker_id: int,
    file_path: str,
    row_group_index: int,
    batch_index: int,
) -> int:
    path_crc = zlib.crc32(file_path.encode("utf-8"))
    return stable_pcvr_batch_seed_from_path_crc(
        base_seed=base_seed,
        worker_id=worker_id,
        path_crc=path_crc,
        row_group_index=row_group_index,
        batch_index=batch_index,
    )
