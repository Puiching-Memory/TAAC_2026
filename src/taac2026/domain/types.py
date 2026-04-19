from __future__ import annotations

from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, Any

import torch
from torch.utils import _pytree

if TYPE_CHECKING:
    from torchrec.sparse.jagged_tensor import KeyedJaggedTensor
else:
    KeyedJaggedTensor = Any


@dataclass(slots=True)
class BatchTensors:
    dense_features: torch.Tensor
    labels: torch.Tensor
    user_indices: torch.Tensor
    item_indices: torch.Tensor
    item_logq: torch.Tensor
    sparse_features: KeyedJaggedTensor | None = None
    sequence_features: KeyedJaggedTensor | None = None

    @property
    def batch_size(self) -> int:
        return int(self.labels.shape[0])

    def to(self, device: torch.device | str) -> "BatchTensors":
        def move_optional_feature(feature: Any | None) -> Any | None:
            return None if feature is None else feature.to(device)

        return BatchTensors(
            dense_features=self.dense_features.to(device),
            labels=self.labels.to(device),
            user_indices=self.user_indices.to(device),
            item_indices=self.item_indices.to(device),
            item_logq=self.item_logq.to(device),
            sparse_features=move_optional_feature(self.sparse_features),
            sequence_features=move_optional_feature(self.sequence_features),
        )


_BATCH_TENSOR_FIELD_NAMES = tuple(field.name for field in fields(BatchTensors))


def _flatten_batch_tensors(batch: BatchTensors) -> tuple[list[object], tuple[str, ...]]:
    return [getattr(batch, name) for name in _BATCH_TENSOR_FIELD_NAMES], _BATCH_TENSOR_FIELD_NAMES


def _flatten_batch_tensors_with_keys(
    batch: BatchTensors,
) -> tuple[list[tuple[_pytree.KeyEntry, object]], tuple[str, ...]]:
    return [(_pytree.GetAttrKey(name), getattr(batch, name)) for name in _BATCH_TENSOR_FIELD_NAMES], _BATCH_TENSOR_FIELD_NAMES


def _dump_batch_tensors_context(context: tuple[str, ...]) -> list[str]:
    return list(context)


def _load_batch_tensors_context(context: list[str]) -> tuple[str, ...]:
    return tuple(context)


def _unflatten_batch_tensors(values: list[object], context: tuple[str, ...]) -> BatchTensors:
    return BatchTensors(**dict(zip(context, values, strict=True)))


_pytree.register_pytree_node(
    BatchTensors,
    _flatten_batch_tensors,
    _unflatten_batch_tensors,
    serialized_type_name="taac2026.domain.types.BatchTensors",
    to_dumpable_context=_dump_batch_tensors_context,
    from_dumpable_context=_load_batch_tensors_context,
    flatten_with_keys_fn=_flatten_batch_tensors_with_keys,
)


@dataclass(slots=True)
class DataStats:
    dense_dim: int
    pos_weight: float
    train_size: int
    val_size: int


__all__ = ["BatchTensors", "DataStats"]
