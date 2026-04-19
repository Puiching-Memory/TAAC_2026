from __future__ import annotations

import torch

from taac2026.domain.features import FeatureSchema, FeatureTableSpec
from taac2026.infrastructure.io.sparse_collate import keyed_jagged_from_masked_batches
from taac2026.infrastructure.nn.embedding import (
    TorchRecEmbeddingBagAdapter,
    TorchRecEmbeddingCollectionAdapter,
    build_embedding_bag_configs,
    build_embedding_configs,
)


def test_build_embedding_bag_configs_respects_table_selection() -> None:
    feature_schema = FeatureSchema(
        tables=(
            FeatureTableSpec(name="user_tokens", family="user", num_embeddings=64, embedding_dim=8),
            FeatureTableSpec(name="candidate_tokens", family="candidate", num_embeddings=128, embedding_dim=8),
            FeatureTableSpec(name="history_tokens", family="history", num_embeddings=256, embedding_dim=8),
        ),
        dense_dim=16,
    )

    configs = build_embedding_bag_configs(feature_schema, table_names=("user_tokens", "candidate_tokens"))

    assert [config.name for config in configs] == ["user_tokens", "candidate_tokens"]
    assert [config.feature_names for config in configs] == [["user_tokens"], ["candidate_tokens"]]
    assert [config.num_embeddings for config in configs] == [64, 128]


def test_build_embedding_configs_respects_table_selection() -> None:
    feature_schema = FeatureSchema(
        tables=(
            FeatureTableSpec(name="user_tokens", family="user", num_embeddings=64, embedding_dim=8),
            FeatureTableSpec(name="candidate_tokens", family="candidate", num_embeddings=128, embedding_dim=8),
            FeatureTableSpec(name="history_tokens", family="history", num_embeddings=256, embedding_dim=8),
        ),
        dense_dim=16,
    )

    configs = build_embedding_configs(feature_schema, table_names=("user_tokens", "history_tokens"))

    assert [config.name for config in configs] == ["user_tokens", "history_tokens"]
    assert [config.feature_names for config in configs] == [["user_tokens"], ["history_tokens"]]


def test_torchrec_embedding_bag_adapter_pools_selected_keys() -> None:
    feature_schema = FeatureSchema(
        tables=(
            FeatureTableSpec(name="user_tokens", family="user", num_embeddings=64, embedding_dim=4),
            FeatureTableSpec(name="candidate_tokens", family="candidate", num_embeddings=64, embedding_dim=4),
            FeatureTableSpec(name="history_tokens", family="history", num_embeddings=64, embedding_dim=4),
        ),
        dense_dim=16,
    )
    features = keyed_jagged_from_masked_batches(
        {
            "user_tokens": (
                torch.tensor([[1, 2, 0], [0, 0, 0]], dtype=torch.long),
                torch.tensor([[True, True, False], [False, False, False]], dtype=torch.bool),
            ),
            "candidate_tokens": (
                torch.tensor([[3, 0, 0], [4, 5, 0]], dtype=torch.long),
                torch.tensor([[True, False, False], [True, True, False]], dtype=torch.bool),
            ),
        }
    )

    adapter = TorchRecEmbeddingBagAdapter(
        feature_schema,
        table_names=("user_tokens", "candidate_tokens"),
    )

    pooled = adapter(features)
    pooled_by_key = adapter.forward_dict(features)

    assert pooled.shape == (2, 8)
    assert adapter.output_dim == 8
    assert adapter.table_names == ("user_tokens", "candidate_tokens")
    assert set(pooled_by_key) == {"user_tokens", "candidate_tokens"}
    assert pooled_by_key["user_tokens"].shape == (2, 4)
    assert pooled_by_key["candidate_tokens"].shape == (2, 4)
    assert torch.isfinite(pooled).all().item()


def test_torchrec_embedding_collection_adapter_preserves_token_lengths() -> None:
    feature_schema = FeatureSchema(
        tables=(
            FeatureTableSpec(name="user_tokens", family="user", num_embeddings=64, embedding_dim=4),
            FeatureTableSpec(name="history_tokens", family="history", num_embeddings=64, embedding_dim=4),
        ),
        dense_dim=16,
    )
    features = keyed_jagged_from_masked_batches(
        {
            "user_tokens": (
                torch.tensor([[1, 2, 0], [3, 0, 0]], dtype=torch.long),
                torch.tensor([[True, True, False], [True, False, False]], dtype=torch.bool),
            ),
            "history_tokens": (
                torch.tensor([[4, 5, 6], [7, 0, 0]], dtype=torch.long),
                torch.tensor([[True, True, True], [True, False, False]], dtype=torch.bool),
            ),
        }
    )

    adapter = TorchRecEmbeddingCollectionAdapter(
        feature_schema,
        table_names=("user_tokens", "history_tokens"),
    )

    embedded = adapter.forward_dict(features)
    history_dense = embedded["history_tokens"].to_padded_dense(desired_length=3)

    assert adapter.table_names == ("user_tokens", "history_tokens")
    assert set(embedded) == {"user_tokens", "history_tokens"}
    assert embedded["user_tokens"].lengths().tolist() == [2, 1]
    assert embedded["history_tokens"].lengths().tolist() == [3, 1]
    assert history_dense.shape == (2, 3, 4)