from __future__ import annotations

import torch
from torch import nn

from taac2026.domain.config import DataConfig, ModelConfig
from taac2026.domain.features import build_default_feature_schema
from taac2026.domain.types import BatchTensors
from taac2026.infrastructure.nn.embedding import TorchRecEmbeddingBagAdapter
from taac2026.infrastructure.nn.heads import ClassificationHead
from taac2026.infrastructure.nn.pooling import TargetAwarePool, masked_mean


SPARSE_TABLE_NAMES = (
    "user_tokens",
    "context_tokens",
    "candidate_tokens",
    "candidate_post_tokens",
    "candidate_author_tokens",
)


class ResidualMLPBlock(nn.Module):
    """A tiny residual block that is easy to swap or deepen."""

    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states + self.layers(hidden_states)


class ReferenceBaselineModel(nn.Module):
    """Starter model intended to be easy to read, copy, and extend.

    The design deliberately favors explicit submodules over clever abstractions:
    users can replace the history block, widen the fusion head, or add new
    feature branches without first untangling a large unified backbone.
    """

    def __init__(self, data_config: DataConfig, model_config: ModelConfig, dense_dim: int) -> None:
        super().__init__()
        self.hidden_dim = model_config.hidden_dim
        self.recent_seq_len = max(0, model_config.recent_seq_len)
        self.sparse_embedding = TorchRecEmbeddingBagAdapter(
            feature_schema=build_default_feature_schema(data_config, model_config),
            table_names=SPARSE_TABLE_NAMES,
        )

        self.token_embedding = nn.Embedding(
            num_embeddings=model_config.vocab_size,
            embedding_dim=model_config.embedding_dim,
            padding_idx=0,
        )
        self.token_projection = (
            nn.Identity()
            if model_config.embedding_dim == model_config.hidden_dim
            else nn.Linear(model_config.embedding_dim, model_config.hidden_dim)
        )

        self.user_encoder = nn.Sequential(
            nn.Linear(model_config.hidden_dim * 2, model_config.hidden_dim),
            nn.LayerNorm(model_config.hidden_dim),
            nn.GELU(),
        )
        self.candidate_encoder = nn.Sequential(
            nn.Linear(model_config.hidden_dim * 3, model_config.hidden_dim),
            nn.LayerNorm(model_config.hidden_dim),
            nn.GELU(),
        )
        self.history_event_encoder = nn.Sequential(
            nn.Linear(model_config.hidden_dim * 4, model_config.hidden_dim),
            nn.LayerNorm(model_config.hidden_dim),
            nn.GELU(),
        )
        self.history_refinement = nn.ModuleList(
            [ResidualMLPBlock(model_config.hidden_dim, model_config.dropout) for _ in range(max(1, model_config.num_layers))]
        )
        self.sequence_encoder = nn.Sequential(
            nn.Linear(model_config.hidden_dim, model_config.hidden_dim),
            nn.LayerNorm(model_config.hidden_dim),
            nn.GELU(),
        )
        self.dense_encoder = nn.Sequential(
            nn.Linear(dense_dim, model_config.hidden_dim),
            nn.LayerNorm(model_config.hidden_dim),
            nn.GELU(),
        )
        self.history_pool = TargetAwarePool(
            model_config.hidden_dim,
            activation="gelu",
            dropout=model_config.dropout,
        )

        head_hidden_dim = model_config.head_hidden_dim or model_config.hidden_dim * 2
        self.output = ClassificationHead(
            input_dim=model_config.hidden_dim * 7,
            hidden_dims=head_hidden_dim,
            activation="gelu",
            dropout=model_config.dropout,
        )

    def _require(self, tensor: torch.Tensor | None, name: str) -> torch.Tensor:
        if tensor is None:
            raise RuntimeError(f"Batch is missing required tensor: {name}")
        return tensor

    def _require_sparse_features(self, batch: BatchTensors):
        if batch.sparse_features is None:
            raise RuntimeError("Batch is missing required TorchRec sparse feature tensor: sparse_features")
        return batch.sparse_features

    def _embed_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.token_projection(self.token_embedding(tokens))

    def _pooled_sparse_summaries(self, batch: BatchTensors) -> dict[str, torch.Tensor]:
        pooled_sparse = self.sparse_embedding.forward_dict(self._require_sparse_features(batch))
        return {
            name: self.token_projection(pooled_sparse[name])
            for name in SPARSE_TABLE_NAMES
        }

    def _embed_sequence_grid(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        embedded = self._embed_tokens(tokens)
        weights = mask.unsqueeze(-1).float()
        summed = (embedded * weights).sum(dim=(1, 2))
        counts = weights.sum(dim=(1, 2)).clamp_min(1.0)
        return summed / counts

    def _slice_recent(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.recent_seq_len <= 0 or tensor.shape[1] <= self.recent_seq_len:
            return tensor
        return tensor[:, -self.recent_seq_len :]

    def forward(self, batch: BatchTensors) -> torch.Tensor:
        # These optional tensors are produced by this package's data pipeline;
        # keeping them explicit makes the extension contract visible to authors.
        history_post_tokens = self._require(batch.history_post_tokens, "history_post_tokens")
        history_author_tokens = self._require(batch.history_author_tokens, "history_author_tokens")
        history_action_tokens = self._require(batch.history_action_tokens, "history_action_tokens")
        sparse_summaries = self._pooled_sparse_summaries(batch)

        history_mask = self._slice_recent(batch.history_mask)
        history_tokens = self._slice_recent(batch.history_tokens)
        history_post_tokens = self._slice_recent(history_post_tokens)
        history_author_tokens = self._slice_recent(history_author_tokens)
        history_action_tokens = self._slice_recent(history_action_tokens)

        user_representation = self.user_encoder(
            torch.cat([sparse_summaries["user_tokens"], sparse_summaries["context_tokens"]], dim=-1)
        )

        dense_representation = self.dense_encoder(batch.dense_features)

        candidate_representation = self.candidate_encoder(
            torch.cat(
                [
                    sparse_summaries["candidate_post_tokens"],
                    sparse_summaries["candidate_author_tokens"],
                    sparse_summaries["candidate_tokens"],
                ],
                dim=-1,
            )
        )

        history_representation = self.history_event_encoder(
            torch.cat(
                [
                    self._embed_tokens(history_tokens),
                    self._embed_tokens(history_post_tokens),
                    self._embed_tokens(history_author_tokens),
                    self._embed_tokens(history_action_tokens),
                ],
                dim=-1,
            )
        )
        history_representation = history_representation * history_mask.unsqueeze(-1).float()
        for block in self.history_refinement:
            history_representation = block(history_representation)
            history_representation = history_representation * history_mask.unsqueeze(-1).float()

        sequence_summary = self._embed_sequence_grid(batch.sequence_tokens, batch.sequence_mask)
        sequence_representation = self.sequence_encoder(sequence_summary)
        history_context = self.history_pool(candidate_representation, history_representation, history_mask)
        history_summary = masked_mean(history_representation, history_mask)

        fused = torch.cat(
            [
                candidate_representation,
                history_context,
                history_summary,
                user_representation,
                dense_representation,
                sequence_representation,
                candidate_representation * history_context,
            ],
            dim=-1,
        )
        return self.output(fused).squeeze(-1)


def build_model_component(data_config: DataConfig, model_config: ModelConfig, dense_dim: int) -> ReferenceBaselineModel:
    return ReferenceBaselineModel(data_config=data_config, model_config=model_config, dense_dim=dense_dim)
