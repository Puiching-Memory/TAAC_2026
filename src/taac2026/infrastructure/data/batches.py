"""Shared PCVR batch tensor utilities."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

import torch


PCVRBatch = dict[str, Any]
PCVRBatchFactory = Callable[[], PCVRBatch]


@dataclass(frozen=True, slots=True)
class PCVRSharedTensorSpec:
    shape: tuple[int, ...]
    dtype: torch.dtype


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
        elif isinstance(value, torch.Tensor) and value.ndim > 0 and value.shape[0] == row_count:
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
        elif isinstance(value, torch.Tensor) and value.ndim > 0 and value.shape[0] == row_count:
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
        elif isinstance(first_value, list):
            if all(
                key in batch and isinstance(batch[key], list) and len(batch[key]) == row_count
                for batch, row_count in zip(batches, row_counts, strict=True)
            ):
                merged[key] = [item for batch in batches for item in batch[key]]
        else:
            merged[key] = clone_pcvr_batch({key: first_value})[key]
    return merged


__all__ = [
    "PCVRBatch",
    "PCVRBatchFactory",
    "PCVRBatchTransform",
    "PCVRSharedTensorSpec",
    "clone_pcvr_batch",
    "concat_pcvr_batches",
    "pcvr_batch_row_count",
    "repeat_pcvr_rows",
    "take_pcvr_rows",
]
