from __future__ import annotations

import logging
from functools import lru_cache
from typing import TYPE_CHECKING, Mapping, Sequence

import torch

if TYPE_CHECKING:
    from torchrec.sparse.jagged_tensor import KeyedJaggedTensor


MaskedFeatureBatch = tuple[torch.Tensor, torch.Tensor]


@lru_cache(maxsize=1)
def _keyed_jagged_tensor_type():
    root_logger = logging.getLogger()
    previous_level = root_logger.level
    root_logger.setLevel(max(previous_level, logging.ERROR))
    try:
        from torchrec.sparse.jagged_tensor import KeyedJaggedTensor
    finally:
        root_logger.setLevel(previous_level)
    return KeyedJaggedTensor


def keyed_jagged_from_masked_batches(feature_batches: Mapping[str, MaskedFeatureBatch]) -> KeyedJaggedTensor:
    if not feature_batches:
        raise ValueError("feature_batches must not be empty")

    keys = list(feature_batches)
    batch_size: int | None = None
    lengths: list[torch.Tensor] = []
    values: list[torch.Tensor] = []

    for key in keys:
        feature_values, feature_mask = feature_batches[key]
        if feature_values.ndim != 2 or feature_mask.ndim != 2:
            raise ValueError(f"Feature batch '{key}' must be rank-2")
        if feature_values.shape != feature_mask.shape:
            raise ValueError(f"Feature batch '{key}' values and mask shapes must match")
        if batch_size is None:
            batch_size = int(feature_values.shape[0])
        elif batch_size != int(feature_values.shape[0]):
            raise ValueError("All feature batches must share the same batch size")

        feature_values = feature_values.to(dtype=torch.long)
        feature_mask = feature_mask.to(dtype=torch.bool)
        lengths.append(feature_mask.to(dtype=torch.int32).sum(dim=1))
        values.append(feature_values.masked_select(feature_mask))

    keyed_jagged_tensor = _keyed_jagged_tensor_type()
    return keyed_jagged_tensor.from_lengths_sync(
        keys=keys,
        values=torch.cat(values, dim=0),
        lengths=torch.cat(lengths, dim=0),
    )


def build_batch_torchrec_features(
    sequence_names: Sequence[str],
    *,
    user_tokens: torch.Tensor,
    user_mask: torch.Tensor,
    context_tokens: torch.Tensor,
    context_mask: torch.Tensor,
    candidate_tokens: torch.Tensor,
    candidate_mask: torch.Tensor,
    candidate_post_tokens: torch.Tensor,
    candidate_post_mask: torch.Tensor,
    candidate_author_tokens: torch.Tensor,
    candidate_author_mask: torch.Tensor,
    history_tokens: torch.Tensor,
    history_mask: torch.Tensor,
    history_post_tokens: torch.Tensor,
    history_author_tokens: torch.Tensor,
    history_action_tokens: torch.Tensor,
    history_time_gap: torch.Tensor,
    history_group_ids: torch.Tensor,
    sequence_tokens: torch.Tensor,
    sequence_mask: torch.Tensor,
) -> tuple[KeyedJaggedTensor, KeyedJaggedTensor]:
    if sequence_tokens.ndim != 3 or sequence_mask.ndim != 3:
        raise ValueError("sequence_tokens and sequence_mask must be rank-3")
    if sequence_tokens.shape != sequence_mask.shape:
        raise ValueError("sequence_tokens and sequence_mask shapes must match")
    if int(sequence_tokens.shape[1]) != len(sequence_names):
        raise ValueError("sequence_names must align with sequence_tokens second dimension")

    sparse_features = keyed_jagged_from_masked_batches(
        {
            "user_tokens": (user_tokens, user_mask),
            "context_tokens": (context_tokens, context_mask),
            "candidate_tokens": (candidate_tokens, candidate_mask),
            "candidate_post_tokens": (candidate_post_tokens, candidate_post_mask),
            "candidate_author_tokens": (candidate_author_tokens, candidate_author_mask),
        }
    )

    sequence_feature_batches: dict[str, MaskedFeatureBatch] = {
        "history_tokens": (history_tokens, history_mask),
        "history_post_tokens": (history_post_tokens, history_mask),
        "history_author_tokens": (history_author_tokens, history_mask),
        "history_action_tokens": (history_action_tokens, history_mask),
        "history_time_gap": (history_time_gap, history_mask),
        "history_group_ids": (history_group_ids, history_mask),
    }
    for sequence_index, sequence_name in enumerate(sequence_names):
        sequence_feature_batches[f"sequence:{sequence_name}"] = (
            sequence_tokens[:, sequence_index, :],
            sequence_mask[:, sequence_index, :],
        )

    sequence_features = keyed_jagged_from_masked_batches(sequence_feature_batches)
    return sparse_features, sequence_features


__all__ = [
    "build_batch_torchrec_features",
    "keyed_jagged_from_masked_batches",
]