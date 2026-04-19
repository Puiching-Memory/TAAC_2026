from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from config.baseline.data import DENSE_FEATURE_DIM, load_dataloaders
from taac2026.infrastructure.io.files import stable_hash64
from tests.support import TestWorkspace, build_edge_case_rows, create_test_workspace


LEGACY_SEQUENCE_FIELD_NAMES = (
    "history_tokens",
    "history_mask",
    "history_post_tokens",
    "history_author_tokens",
    "history_action_tokens",
    "history_time_gap",
    "history_group_ids",
    "sequence_tokens",
    "sequence_mask",
)

LEGACY_SPARSE_FIELD_NAMES = (
    "candidate_tokens",
    "candidate_mask",
    "context_tokens",
    "context_mask",
    "user_tokens",
    "user_mask",
    "candidate_post_tokens",
    "candidate_post_mask",
    "candidate_author_tokens",
    "candidate_author_mask",
)


@pytest.fixture
def test_workspace(tmp_path: Path) -> TestWorkspace:
    return create_test_workspace(tmp_path)


def _sequence_by_key(batch) -> dict[str, object]:
    assert batch.sequence_features is not None
    return batch.sequence_features.to_dict()


def _dense_sequence_tokens(sequence_by_key: dict[str, object], name: str, desired_length: int) -> tuple[torch.Tensor, torch.Tensor]:
    jagged = sequence_by_key[name]
    tokens = jagged.to_padded_dense(desired_length=desired_length, padding_value=0).to(dtype=torch.long)
    lengths = jagged.lengths().to(device=tokens.device)
    positions = torch.arange(desired_length, device=tokens.device).unsqueeze(0)
    return tokens, positions < lengths.unsqueeze(1)


def _dense_sparse_tokens(sparse_by_key: dict[str, object], name: str, desired_length: int) -> tuple[torch.Tensor, torch.Tensor]:
    jagged = sparse_by_key[name]
    tokens = jagged.to_padded_dense(desired_length=desired_length, padding_value=0).to(dtype=torch.long)
    lengths = jagged.lengths().to(device=tokens.device)
    positions = torch.arange(desired_length, device=tokens.device).unsqueeze(0)
    return tokens, positions < lengths.unsqueeze(1)


def _dense_sequence_grid(batch, sequence_names: tuple[str, ...], max_seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    sequence_by_key = _sequence_by_key(batch)
    sequence_tokens: list[torch.Tensor] = []
    sequence_mask: list[torch.Tensor] = []
    for sequence_name in sequence_names:
        tokens, mask = _dense_sequence_tokens(sequence_by_key, "sequence:" + sequence_name, max_seq_len)
        sequence_tokens.append(tokens)
        sequence_mask.append(mask)
    return torch.stack(sequence_tokens, dim=1), torch.stack(sequence_mask, dim=1)


def _sequence_length_grid(batch, sequence_names: tuple[str, ...]) -> torch.Tensor:
    sequence_by_key = _sequence_by_key(batch)
    return torch.stack([sequence_by_key["sequence:" + name].lengths() for name in sequence_names], dim=1)


def test_streaming_collate_batch_contract(test_workspace: TestWorkspace) -> None:
    train_loader, val_loader, data_stats = load_dataloaders(
        config=test_workspace.data_config,
        vocab_size=257,
        batch_size=2,
        eval_batch_size=2,
        num_workers=0,
        seed=7,
    )

    train_batch = next(iter(train_loader))
    val_batch = next(iter(val_loader))
    sequence_names = tuple(test_workspace.data_config.sequence_names)
    history_capacity = len(sequence_names) * test_workspace.data_config.max_seq_len
    sequence_by_key = _sequence_by_key(train_batch)
    history_tokens, history_mask = _dense_sequence_tokens(sequence_by_key, "history_tokens", history_capacity)
    history_post_tokens, _ = _dense_sequence_tokens(sequence_by_key, "history_post_tokens", history_capacity)
    history_time_gap, _ = _dense_sequence_tokens(sequence_by_key, "history_time_gap", history_capacity)
    history_group_ids, _ = _dense_sequence_tokens(sequence_by_key, "history_group_ids", history_capacity)
    sequence_tokens, sequence_mask = _dense_sequence_grid(train_batch, sequence_names, test_workspace.data_config.max_seq_len)
    sparse_by_key = train_batch.sparse_features.to_dict()
    user_tokens, user_mask = _dense_sparse_tokens(sparse_by_key, "user_tokens", test_workspace.data_config.max_feature_tokens)
    candidate_tokens, candidate_mask = _dense_sparse_tokens(sparse_by_key, "candidate_tokens", 1)
    candidate_post_tokens, candidate_post_mask = _dense_sparse_tokens(
        sparse_by_key,
        "candidate_post_tokens",
        max(1, test_workspace.data_config.max_event_features),
    )
    candidate_author_tokens, candidate_author_mask = _dense_sparse_tokens(sparse_by_key, "candidate_author_tokens", 2)

    assert data_stats.dense_dim == DENSE_FEATURE_DIM
    assert train_batch.batch_size == 2
    assert sequence_tokens.shape[1] == len(sequence_names)
    assert train_batch.dense_features.shape[1] == data_stats.dense_dim
    assert user_tokens.shape[1] == test_workspace.data_config.max_feature_tokens
    assert history_mask.any().item()
    assert history_post_tokens.shape == history_tokens.shape
    assert history_time_gap.shape == history_tokens.shape
    assert history_group_ids.max().item() <= len(sequence_names)
    assert candidate_tokens.shape[1] == 1
    assert candidate_post_tokens.shape[1] == max(1, test_workspace.data_config.max_event_features)
    assert candidate_author_tokens.shape[1] == 2
    assert candidate_mask.any().item()
    assert candidate_post_mask.any().item()
    assert candidate_author_mask.any().item()
    assert train_batch.sparse_features is not None
    assert train_batch.sequence_features is not None
    for field_name in LEGACY_SEQUENCE_FIELD_NAMES:
        assert not hasattr(train_batch, field_name)
    for field_name in LEGACY_SPARSE_FIELD_NAMES:
        assert not hasattr(train_batch, field_name)
    assert val_batch.labels.ndim == 1
    assert train_batch.user_indices.dtype == torch.long
    assert train_batch.item_logq.dtype == torch.float32
    assert torch.isfinite(train_batch.item_logq).all().item()

    assert set(train_batch.sparse_features.keys()) == {
        "user_tokens",
        "context_tokens",
        "candidate_tokens",
        "candidate_post_tokens",
        "candidate_author_tokens",
    }
    assert set(sequence_by_key) == {
        "history_tokens",
        "history_post_tokens",
        "history_author_tokens",
        "history_action_tokens",
        "history_time_gap",
        "history_group_ids",
        *(f"sequence:{name}" for name in sequence_names),
    }
    assert sparse_by_key["user_tokens"].lengths().tolist() == user_mask.sum(dim=1).to(torch.int32).tolist()
    assert sparse_by_key["candidate_tokens"].lengths().tolist() == candidate_mask.sum(dim=1).to(torch.int32).tolist()
    assert sequence_by_key["history_tokens"].lengths().tolist() == history_mask.sum(dim=1).to(torch.int32).tolist()

    for sequence_index, sequence_name in enumerate(sequence_names):
        expected_lengths = sequence_mask[:, sequence_index, :].sum(dim=1).to(torch.int32)
        assert sequence_by_key[f"sequence:{sequence_name}"].lengths().tolist() == expected_lengths.tolist()

    moved_batch = train_batch.to("cpu")
    assert moved_batch.sparse_features is not None
    assert moved_batch.sequence_features is not None
    assert moved_batch.sparse_features.values().device.type == "cpu"
    assert moved_batch.sequence_features.values().device.type == "cpu"


def test_train_split_item_logq_tracks_frequency(test_workspace: TestWorkspace) -> None:
    train_loader, _, _ = load_dataloaders(
        config=test_workspace.data_config,
        vocab_size=257,
        batch_size=2,
        eval_batch_size=2,
        num_workers=0,
        seed=7,
    )

    item_logq_by_index: dict[int, float] = {}
    for batch in train_loader:
        for item_index, item_logq in zip(batch.item_indices.tolist(), batch.item_logq.tolist(), strict=False):
            item_logq_by_index.setdefault(int(item_index), float(item_logq))

    repeated_item = stable_hash64("item|101")
    single_item = stable_hash64("item|102")
    assert repeated_item in item_logq_by_index
    assert single_item in item_logq_by_index
    assert item_logq_by_index[repeated_item] > item_logq_by_index[single_item]
    assert item_logq_by_index[repeated_item] == pytest.approx(math.log(2.0 / 3.0), abs=1e-5)
    assert item_logq_by_index[single_item] == pytest.approx(math.log(1.0 / 3.0), abs=1e-5)


def test_streaming_pipeline_handles_sparse_and_truncated_sequences(tmp_path: Path) -> None:
    edge_workspace = create_test_workspace(tmp_path / "edge_cases", rows=build_edge_case_rows())
    edge_workspace.data_config.max_seq_len = 3
    train_loader, val_loader, data_stats = load_dataloaders(
        config=edge_workspace.data_config,
        vocab_size=257,
        batch_size=2,
        eval_batch_size=2,
        num_workers=0,
        seed=7,
    )

    batches = [*train_loader, *val_loader]

    assert data_stats.train_size >= 1
    assert data_stats.val_size >= 1
    sequence_names = tuple(edge_workspace.data_config.sequence_names)
    history_capacity = len(sequence_names) * edge_workspace.data_config.max_seq_len
    assert any((_sequence_length_grid(batch, sequence_names).sum(dim=1) == 0).any().item() for batch in batches)
    assert all(torch.isfinite(batch.dense_features).all().item() for batch in batches)
    assert max(int(_sequence_length_grid(batch, sequence_names).max().item()) for batch in batches) <= edge_workspace.data_config.max_seq_len
    assert max(
        int(_dense_sequence_tokens(_sequence_by_key(batch), "history_group_ids", history_capacity)[0].max().item())
        for batch in batches
    ) <= len(sequence_names)
