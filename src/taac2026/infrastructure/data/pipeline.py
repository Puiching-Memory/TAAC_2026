"""Composable PCVR data pipeline components."""

from __future__ import annotations

import zlib
from collections.abc import Hashable, Iterable

import torch

from taac2026.infrastructure.data.batches import (
    PCVRBatch,
    PCVRBatchFactory,
    PCVRBatchTransform,
    PCVRSharedTensorSpec,
    clone_pcvr_batch,
    concat_pcvr_batches,
    pcvr_batch_row_count,
    repeat_pcvr_rows,
    take_pcvr_rows,
)
from taac2026.infrastructure.data.cache import PCVRMemoryBatchCache, PCVRSharedBatchCache
from taac2026.infrastructure.data.shuffle import PCVRShuffleBuffer
from taac2026.infrastructure.data.transforms import (
    PCVRDomainDropoutTransform,
    PCVRFeatureMaskTransform,
    PCVRSequenceCropTransform,
    build_pcvr_batch_transform,
    build_pcvr_batch_transforms,
)


class PCVRDataPipeline:
    """Compose cache and row-level transforms around Parquet batch conversion."""

    def __init__(
        self,
        *,
        cache: PCVRMemoryBatchCache | None = None,
        transforms: list[PCVRBatchTransform] | tuple[PCVRBatchTransform, ...] = (),
    ) -> None:
        self.cache = cache if cache is not None else PCVRMemoryBatchCache()
        self.transforms = tuple(transforms)

    @property
    def requires_generator(self) -> bool:
        return bool(self.transforms)

    def configure_access_trace(
        self,
        trace: Iterable[Hashable],
        *,
        cyclic: bool = True,
        key_universe: Iterable[Hashable] = (),
    ) -> None:
        self.cache.configure_access_trace(trace, cyclic=cyclic, key_universe=key_universe)

    def configure_key_universe(self, keys: Iterable[Hashable]) -> None:
        self.cache.configure_key_universe(keys)

    def read_base_batch(self, key: Hashable, factory: PCVRBatchFactory) -> PCVRBatch:
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        batch = factory()
        self.cache.put(key, batch)
        return batch

    def apply_transforms(self, batch: PCVRBatch, *, generator: torch.Generator | None = None) -> PCVRBatch:
        if not self.transforms:
            return batch
        if generator is None:
            generator = torch.Generator()
        transformed = batch
        for transform in self.transforms:
            transformed = transform(transformed, generator=generator)
        return transformed


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


__all__ = [
    "PCVRBatch",
    "PCVRBatchFactory",
    "PCVRBatchTransform",
    "PCVRDataPipeline",
    "PCVRDomainDropoutTransform",
    "PCVRFeatureMaskTransform",
    "PCVRMemoryBatchCache",
    "PCVRSequenceCropTransform",
    "PCVRSharedBatchCache",
    "PCVRSharedTensorSpec",
    "PCVRShuffleBuffer",
    "build_pcvr_batch_transform",
    "build_pcvr_batch_transforms",
    "clone_pcvr_batch",
    "concat_pcvr_batches",
    "pcvr_batch_row_count",
    "repeat_pcvr_rows",
    "stable_pcvr_batch_seed",
    "stable_pcvr_batch_seed_from_path_crc",
    "take_pcvr_rows",
]
