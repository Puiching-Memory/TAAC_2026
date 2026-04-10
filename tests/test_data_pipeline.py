from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from config.gen.baseline.data import DENSE_FEATURE_DIM, load_dataloaders
from taac2026.infrastructure.io.files import stable_hash64
from tests.support import TestWorkspace, build_row, create_test_workspace, write_dataset


pytestmark = [pytest.mark.integration, pytest.mark.smoke]


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
    assert val_batch.labels.ndim == 1
    assert train_batch.user_indices.dtype == torch.long
    assert train_batch.item_logq.dtype == torch.float32
    assert torch.isfinite(train_batch.item_logq).all().item()


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


def test_streaming_collate_handles_missing_features_and_empty_sequences(tmp_path: Path) -> None:
    dataset_path = tmp_path / "edge_cases.parquet"
    rows = [
        build_row(
            0,
            1_770_000_100,
            True,
            "u1",
            301,
            user_feature=None,
            item_feature=None,
            context_feature=None,
            cross_feature=None,
            seq_feature={},
        ),
        build_row(
            1,
            1_770_000_200,
            False,
            "u1",
            302,
            seq_feature={
                "action_seq": [{"feature_id": 11, "int_array": []}],
                "content_seq": None,
                "item_seq": [{"feature_id": 13, "int_array": [41]}],
            },
        ),
        build_row(
            2,
            1_770_000_300,
            True,
            "u2",
            303,
            cross_feature=[{"feature_id": 25, "float_array": [0.25, -0.5, 0.75]}],
        ),
    ]
    write_dataset(dataset_path, rows)
    workspace = create_test_workspace(tmp_path)
    workspace.data_config.dataset_path = str(dataset_path)

    train_loader, val_loader, _ = load_dataloaders(
        config=workspace.data_config,
        vocab_size=257,
        batch_size=2,
        eval_batch_size=2,
        num_workers=0,
        seed=7,
    )

    train_batch = next(iter(train_loader))
    val_batch = next(iter(val_loader))

    assert train_batch.batch_size >= 1
    assert val_batch.batch_size >= 1
    assert train_batch.candidate_mask.any().item()
    assert train_batch.user_mask is not None
    assert torch.logical_not(train_batch.sequence_mask).any().item()
    assert torch.isfinite(train_batch.dense_features).all().item()


def test_streaming_collate_truncates_per_sequence_and_preserves_group_ids(tmp_path: Path) -> None:
    dataset_path = tmp_path / "truncated_sequences.parquet"
    base_timestamp = 1_770_100_000
    rows = [
        build_row(
            0,
            base_timestamp,
            True,
            "u9",
            501,
            seq_feature={
                "action_seq": [
                    {"feature_id": 11, "int_array": [10, 11, 12, 13, 14]},
                    {"feature_id": 99, "int_array": [base_timestamp - 60, base_timestamp - 50, base_timestamp - 40, base_timestamp - 30, base_timestamp - 20]},
                ],
                "content_seq": [
                    {"feature_id": 12, "int_array": [20, 21, 22, 23, 24]},
                    {"feature_id": 99, "int_array": [base_timestamp - 55, base_timestamp - 45, base_timestamp - 35, base_timestamp - 25, base_timestamp - 15]},
                ],
                "item_seq": [
                    {"feature_id": 13, "int_array": [30, 31, 32, 33, 34]},
                    {"feature_id": 99, "int_array": [base_timestamp - 54, base_timestamp - 44, base_timestamp - 34, base_timestamp - 24, base_timestamp - 14]},
                ],
            },
        ),
        build_row(1, base_timestamp + 5, False, "u8", 502),
        build_row(2, base_timestamp + 10, True, "u7", 503),
    ]
    write_dataset(dataset_path, rows)
    workspace = create_test_workspace(tmp_path)
    workspace.data_config.dataset_path = str(dataset_path)
    workspace.data_config.max_seq_len = 2

    train_loader, _, _ = load_dataloaders(
        config=workspace.data_config,
        vocab_size=257,
        batch_size=2,
        eval_batch_size=2,
        num_workers=0,
        seed=7,
    )

    batch = next(iter(train_loader))

    assert batch.sequence_mask.shape[1:] == (3, 2)
    assert batch.sequence_mask[0].sum().item() == 6
    assert set(batch.history_group_ids[0][batch.history_mask[0]].tolist()) == {1, 2, 3}
