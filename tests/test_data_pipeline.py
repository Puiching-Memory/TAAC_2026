from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from config.gen.baseline.data import DENSE_FEATURE_DIM, load_dataloaders
from taac2026.infrastructure.io.files import stable_hash64
from tests.support import TestWorkspace, build_edge_case_rows, create_test_workspace


@pytest.fixture
def test_workspace(tmp_path: Path) -> TestWorkspace:
    return create_test_workspace(tmp_path)


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

    assert data_stats.dense_dim == DENSE_FEATURE_DIM
    assert train_batch.batch_size == 2
    assert train_batch.sequence_tokens.shape[1] == len(test_workspace.data_config.sequence_names)
    assert train_batch.dense_features.shape[1] == data_stats.dense_dim
    assert train_batch.user_tokens is not None
    assert train_batch.user_mask is not None
    assert train_batch.user_tokens.shape[1] == test_workspace.data_config.max_feature_tokens
    assert train_batch.history_mask.any().item()
    assert train_batch.history_post_tokens is not None
    assert train_batch.history_author_tokens is not None
    assert train_batch.history_action_tokens is not None
    assert train_batch.history_time_gap is not None
    assert train_batch.history_group_ids is not None
    assert train_batch.history_post_tokens.shape == train_batch.history_tokens.shape
    assert train_batch.history_time_gap.shape == train_batch.history_tokens.shape
    assert train_batch.history_group_ids.max().item() <= len(test_workspace.data_config.sequence_names)
    assert train_batch.candidate_post_tokens is not None
    assert train_batch.candidate_author_tokens is not None
    assert train_batch.candidate_post_mask is not None
    assert train_batch.candidate_author_mask is not None
    assert train_batch.candidate_post_mask.any().item()
    assert train_batch.candidate_author_mask.any().item()
    assert train_batch.sparse_features is not None
    assert train_batch.sequence_features is not None
    assert val_batch.labels.ndim == 1
    assert train_batch.user_indices.dtype == torch.long
    assert train_batch.item_logq.dtype == torch.float32
    assert torch.isfinite(train_batch.item_logq).all().item()

    sparse_by_key = train_batch.sparse_features.to_dict()
    sequence_by_key = train_batch.sequence_features.to_dict()

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
        *(f"sequence:{name}" for name in test_workspace.data_config.sequence_names),
    }
    assert sparse_by_key["user_tokens"].lengths().tolist() == train_batch.user_mask.sum(dim=1).to(torch.int32).tolist()
    assert sparse_by_key["candidate_tokens"].lengths().tolist() == train_batch.candidate_mask.sum(dim=1).to(torch.int32).tolist()
    assert sequence_by_key["history_tokens"].lengths().tolist() == train_batch.history_mask.sum(dim=1).to(torch.int32).tolist()

    for sequence_index, sequence_name in enumerate(test_workspace.data_config.sequence_names):
        expected_lengths = train_batch.sequence_mask[:, sequence_index, :].sum(dim=1).to(torch.int32)
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
    assert any((batch.sequence_mask.sum(dim=(1, 2)) == 0).any().item() for batch in batches)
    assert all(torch.isfinite(batch.dense_features).all().item() for batch in batches)
    assert max(int(batch.sequence_mask.sum(dim=2).max().item()) for batch in batches) <= edge_workspace.data_config.max_seq_len
    assert max(int(batch.history_group_ids.max().item()) for batch in batches) <= len(edge_workspace.data_config.sequence_names)
