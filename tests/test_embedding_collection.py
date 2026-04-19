from __future__ import annotations

import torch
from torch import nn

from taac2026.domain.features import FeatureSchema, FeatureTableSpec
from taac2026.domain.types import BatchTensors
from taac2026.infrastructure.io.sparse_collate import keyed_jagged_from_masked_batches
from taac2026.infrastructure.nn.embedding import TorchRecEmbeddingBagAdapter
from taac2026.infrastructure.nn.quantization import quantize_model_for_inference


FEATURE_SCHEMA = FeatureSchema(
    tables=(
        FeatureTableSpec(name="user_tokens", family="user", num_embeddings=64, embedding_dim=4),
        FeatureTableSpec(name="context_tokens", family="context", num_embeddings=64, embedding_dim=4),
        FeatureTableSpec(name="candidate_tokens", family="candidate", num_embeddings=64, embedding_dim=4),
    ),
    dense_dim=2,
)
TABLE_NAMES = ("user_tokens", "context_tokens", "candidate_tokens")


def _build_sparse_features():
    return keyed_jagged_from_masked_batches(
        {
            "user_tokens": (
                torch.tensor([[1, 2, 0], [3, 0, 0]], dtype=torch.long),
                torch.tensor([[True, True, False], [True, False, False]], dtype=torch.bool),
            ),
            "context_tokens": (
                torch.tensor([[4, 0, 0], [5, 6, 0]], dtype=torch.long),
                torch.tensor([[True, False, False], [True, True, False]], dtype=torch.bool),
            ),
            "candidate_tokens": (
                torch.tensor([[7, 8, 0], [9, 0, 0]], dtype=torch.long),
                torch.tensor([[True, True, False], [True, False, False]], dtype=torch.bool),
            ),
        }
    )


def _build_batch() -> BatchTensors:
    return BatchTensors(
        dense_features=torch.tensor([[0.1, 0.2], [0.3, 0.4]], dtype=torch.float32),
        labels=torch.tensor([1.0, 0.0], dtype=torch.float32),
        user_indices=torch.tensor([1, 2], dtype=torch.long),
        item_indices=torch.tensor([11, 12], dtype=torch.long),
        item_logq=torch.tensor([0.0, -0.7], dtype=torch.float32),
        sparse_features=_build_sparse_features(),
    )


class TinyEmbeddingCollectionModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = TorchRecEmbeddingBagAdapter(FEATURE_SCHEMA, table_names=TABLE_NAMES)
        self.output = nn.Linear(self.embedding.output_dim + FEATURE_SCHEMA.dense_dim, 1)

    def forward(self, batch: BatchTensors) -> torch.Tensor:
        if batch.sparse_features is None:
            raise RuntimeError("Batch is missing sparse_features")
        pooled = self.embedding(batch.sparse_features)
        fused = torch.cat([pooled, batch.dense_features], dim=-1)
        return self.output(fused).squeeze(-1)


def test_embedding_collection_model_prefers_sparse_features() -> None:
    model = TinyEmbeddingCollectionModel().eval()
    batch = _build_batch()

    with torch.inference_mode():
        actual = model(batch)

    assert not hasattr(batch, "candidate_tokens")
    assert not hasattr(batch, "context_tokens")
    assert actual.shape == batch.labels.shape


def test_quantized_embedding_collection_model_runs_with_batch_input() -> None:
    model = TinyEmbeddingCollectionModel().eval()
    batch = _build_batch()

    quantized_model, summary = quantize_model_for_inference(model, "int8")

    assert summary["active"] is True
    assert summary["quantizable_linear_layers"] > 0
    assert summary["quantized_embedding_collections"] > 0
    with torch.inference_mode():
        output = quantized_model(batch)

    assert output.shape == batch.labels.shape
    assert torch.isfinite(output).all().item()