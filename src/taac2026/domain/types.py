from __future__ import annotations

from dataclasses import dataclass, fields

import torch
from torch.utils import _pytree


@dataclass(slots=True)
class BatchTensors:
    candidate_tokens: torch.Tensor
    candidate_mask: torch.Tensor
    context_tokens: torch.Tensor
    context_mask: torch.Tensor
    history_tokens: torch.Tensor
    history_mask: torch.Tensor
    sequence_tokens: torch.Tensor
    sequence_mask: torch.Tensor
    dense_features: torch.Tensor
    labels: torch.Tensor
    user_indices: torch.Tensor
    item_indices: torch.Tensor
    item_logq: torch.Tensor
    user_tokens: torch.Tensor | None = None
    user_mask: torch.Tensor | None = None
    candidate_post_tokens: torch.Tensor | None = None
    candidate_post_mask: torch.Tensor | None = None
    candidate_author_tokens: torch.Tensor | None = None
    candidate_author_mask: torch.Tensor | None = None
    history_post_tokens: torch.Tensor | None = None
    history_author_tokens: torch.Tensor | None = None
    history_action_tokens: torch.Tensor | None = None
    history_time_gap: torch.Tensor | None = None
    history_group_ids: torch.Tensor | None = None

    @property
    def batch_size(self) -> int:
        return int(self.labels.shape[0])

    def to(self, device: torch.device | str) -> "BatchTensors":
        def move_optional(tensor: torch.Tensor | None) -> torch.Tensor | None:
            return None if tensor is None else tensor.to(device)

        return BatchTensors(
            candidate_tokens=self.candidate_tokens.to(device),
            candidate_mask=self.candidate_mask.to(device),
            context_tokens=self.context_tokens.to(device),
            context_mask=self.context_mask.to(device),
            history_tokens=self.history_tokens.to(device),
            history_mask=self.history_mask.to(device),
            sequence_tokens=self.sequence_tokens.to(device),
            sequence_mask=self.sequence_mask.to(device),
            dense_features=self.dense_features.to(device),
            labels=self.labels.to(device),
            user_indices=self.user_indices.to(device),
            item_indices=self.item_indices.to(device),
            item_logq=self.item_logq.to(device),
            user_tokens=move_optional(self.user_tokens),
            user_mask=move_optional(self.user_mask),
            candidate_post_tokens=move_optional(self.candidate_post_tokens),
            candidate_post_mask=move_optional(self.candidate_post_mask),
            candidate_author_tokens=move_optional(self.candidate_author_tokens),
            candidate_author_mask=move_optional(self.candidate_author_mask),
            history_post_tokens=move_optional(self.history_post_tokens),
            history_author_tokens=move_optional(self.history_author_tokens),
            history_action_tokens=move_optional(self.history_action_tokens),
            history_time_gap=move_optional(self.history_time_gap),
            history_group_ids=move_optional(self.history_group_ids),
        )


_BATCH_TENSOR_FIELD_NAMES = tuple(field.name for field in fields(BatchTensors))


def _flatten_batch_tensors(batch: BatchTensors) -> tuple[list[torch.Tensor | None], tuple[str, ...]]:
    return [getattr(batch, name) for name in _BATCH_TENSOR_FIELD_NAMES], _BATCH_TENSOR_FIELD_NAMES


def _unflatten_batch_tensors(values: list[torch.Tensor | None], context: tuple[str, ...]) -> BatchTensors:
    return BatchTensors(**dict(zip(context, values, strict=True)))


_pytree.register_pytree_node(BatchTensors, _flatten_batch_tensors, _unflatten_batch_tensors)


@dataclass(slots=True)
class DataStats:
    dense_dim: int
    pos_weight: float
    train_size: int
    val_size: int


__all__ = ["BatchTensors", "DataStats"]
