"""Composable PCVR data pipeline components."""

from __future__ import annotations

import zlib
from collections.abc import Callable, Hashable, Iterable
from dataclasses import dataclass
from typing import Protocol

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
    PCVRNonSequentialSparseDropoutTransform,
    PCVRSequenceCropTransform,
    build_pcvr_batch_transform,
    build_pcvr_batch_transforms,
)


class PCVRDataPipelineStage(Protocol):
    name: str

    def __call__(self, batch: PCVRBatch, *, generator: torch.Generator) -> PCVRBatch:
        """Return a processed PCVR batch."""


@dataclass(frozen=True, slots=True)
class PCVRTransformStage:
    transform: PCVRBatchTransform
    name: str

    def __call__(self, batch: PCVRBatch, *, generator: torch.Generator) -> PCVRBatch:
        return self.transform(batch, generator=generator)


def _transform_stage(transform: PCVRBatchTransform) -> PCVRTransformStage:
    return PCVRTransformStage(
        transform=transform,
        name=getattr(transform, "name", type(transform).__name__),
    )


class PCVRDataPipeline:
    """Compose cache and row-level transforms around Parquet batch conversion."""

    def __init__(
        self,
        *,
        cache: PCVRMemoryBatchCache | None = None,
        transforms: list[PCVRBatchTransform] | tuple[PCVRBatchTransform, ...] = (),
        stages: list[PCVRDataPipelineStage] | tuple[PCVRDataPipelineStage, ...] = (),
    ) -> None:
        self.cache = cache if cache is not None else PCVRMemoryBatchCache()
        self.transforms = tuple(transforms)
        self.stages = (*tuple(stages), *tuple(_transform_stage(transform) for transform in self.transforms))

    @property
    def requires_generator(self) -> bool:
        return bool(self.stages)

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
        if not self.stages:
            return batch
        if generator is None:
            generator = torch.Generator()
        transformed = batch
        for stage in self.stages:
            transformed = stage(transformed, generator=generator)
        return transformed

    def materialize(
        self,
        key: Hashable,
        factory: PCVRBatchFactory,
        *,
        generator: torch.Generator | None = None,
        preprocess: Callable[[PCVRBatch], PCVRBatch | None] | None = None,
    ) -> PCVRBatch | None:
        batch = self.read_base_batch(key, factory)
        if preprocess is not None:
            batch = preprocess(batch)
            if batch is None:
                return None
        return self.apply_transforms(batch, generator=generator)


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
    "PCVRDataPipelineStage",
    "PCVRDomainDropoutTransform",
    "PCVRFeatureMaskTransform",
    "PCVRMemoryBatchCache",
    "PCVRNonSequentialSparseDropoutTransform",
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
