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

SEQUENCE_FEATURE_KEYS = (
    "history_post_tokens",
    "history_author_tokens",
    "history_action_tokens",
)


class CTRBaselineDINModel(nn.Module):
    def __init__(self, data_config: DataConfig, model_config: ModelConfig, dense_dim: int) -> None:
        super().__init__()
        self.hidden_dim = model_config.hidden_dim
        self.recent_seq_len = max(0, model_config.recent_seq_len)
        self.history_capacity = len(data_config.sequence_names) * data_config.max_seq_len
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
        self.user_projection = nn.Sequential(
            nn.Linear(model_config.hidden_dim * 2, model_config.hidden_dim),
            nn.LayerNorm(model_config.hidden_dim),
            nn.GELU(),
        )
        self.candidate_projection = nn.Sequential(
            nn.Linear(model_config.hidden_dim * 3, model_config.hidden_dim),
            nn.LayerNorm(model_config.hidden_dim),
            nn.GELU(),
        )
        self.history_projection = nn.Sequential(
            nn.Linear(model_config.hidden_dim * 3, model_config.hidden_dim),
            nn.LayerNorm(model_config.hidden_dim),
            nn.GELU(),
        )
        self.dense_projection = nn.Sequential(
            nn.Linear(dense_dim, model_config.hidden_dim),
            nn.LayerNorm(model_config.hidden_dim),
            nn.GELU(),
        )
        self.attention_layer = TargetAwarePool(
            model_config.hidden_dim,
            activation="prelu",
            dropout=model_config.dropout,
        )

        head_hidden_dim = model_config.head_hidden_dim or model_config.hidden_dim * 2
        mlp_hidden_dim = max(64, head_hidden_dim // 2)
        self.output = ClassificationHead(
            input_dim=model_config.hidden_dim * 7,
            hidden_dims=[head_hidden_dim, mlp_hidden_dim],
            activation="prelu",
            dropout=[model_config.dropout, model_config.dropout * 0.5],
        )

    def _require(self, tensor: torch.Tensor | None, name: str) -> torch.Tensor:
        if tensor is None:
            raise RuntimeError(f"Batch is missing required tensor: {name}")
        return tensor

    def _require_sparse_features(self, batch: BatchTensors):
        if batch.sparse_features is None:
            raise RuntimeError("Batch is missing required TorchRec sparse feature tensor: sparse_features")
        return batch.sparse_features

    def _require_sequence_features(self, batch: BatchTensors):
        if batch.sequence_features is None:
            raise RuntimeError("Batch is missing required TorchRec sparse feature tensor: sequence_features")
        return batch.sequence_features

    def _embed_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.token_projection(self.token_embedding(tokens))

    def _pooled_sparse_summaries(self, batch: BatchTensors) -> dict[str, torch.Tensor]:
        pooled_sparse = self.sparse_embedding.forward_dict(self._require_sparse_features(batch))
        return {
            name: self.token_projection(pooled_sparse[name])
            for name in SPARSE_TABLE_NAMES
        }

    def _slice_recent(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.recent_seq_len <= 0 or tensor.shape[1] <= self.recent_seq_len:
            return tensor
        return tensor[:, -self.recent_seq_len :]

    def _dense_sequence_tokens(self, sequence_by_key, name: str) -> tuple[torch.Tensor, torch.Tensor]:
        jagged = sequence_by_key[name]
        tokens = jagged.to_padded_dense(desired_length=self.history_capacity, padding_value=0).to(dtype=torch.long)
        lengths = jagged.lengths().to(device=tokens.device)
        positions = torch.arange(self.history_capacity, device=tokens.device).unsqueeze(0)
        mask = positions < lengths.unsqueeze(1)
        return tokens, mask

    def forward(self, batch: BatchTensors) -> torch.Tensor:
        sequence_by_key = self._require_sequence_features(batch).to_dict()
        missing_keys = [name for name in SEQUENCE_FEATURE_KEYS if name not in sequence_by_key]
        if missing_keys:
            missing = ", ".join(missing_keys)
            raise RuntimeError(f"Batch sequence_features is missing required keys: {missing}")

        history_post_tokens, history_mask = self._dense_sequence_tokens(sequence_by_key, "history_post_tokens")
        history_author_tokens, _ = self._dense_sequence_tokens(sequence_by_key, "history_author_tokens")
        history_action_tokens, _ = self._dense_sequence_tokens(sequence_by_key, "history_action_tokens")
        sparse_summaries = self._pooled_sparse_summaries(batch)

        history_mask = self._slice_recent(history_mask)
        history_post_tokens = self._slice_recent(history_post_tokens)
        history_author_tokens = self._slice_recent(history_author_tokens)
        history_action_tokens = self._slice_recent(history_action_tokens)

        user_representation = self.user_projection(
            torch.cat([sparse_summaries["user_tokens"], sparse_summaries["context_tokens"]], dim=-1)
        )

        dense_representation = self.dense_projection(batch.dense_features)

        candidate_representation = self.candidate_projection(
            torch.cat(
                [
                    sparse_summaries["candidate_post_tokens"],
                    sparse_summaries["candidate_author_tokens"],
                    sparse_summaries["candidate_tokens"],
                ],
                dim=-1,
            )
        )

        history_representation = self.history_projection(
            torch.cat(
                [
                    self._embed_tokens(history_post_tokens),
                    self._embed_tokens(history_author_tokens),
                    self._embed_tokens(history_action_tokens),
                ],
                dim=-1,
            )
        )
        history_representation = history_representation * history_mask.unsqueeze(-1).float()

        history_context = self.attention_layer(candidate_representation, history_representation, history_mask)
        history_summary = masked_mean(history_representation, history_mask)

        fused = torch.cat(
            [
                candidate_representation,
                history_context,
                history_summary,
                user_representation,
                dense_representation,
                candidate_representation * history_context,
                torch.abs(candidate_representation - user_representation),
            ],
            dim=-1,
        )
        return self.output(fused).squeeze(-1)


def build_model_component(data_config: DataConfig, model_config: ModelConfig, dense_dim: int) -> CTRBaselineDINModel:
    return CTRBaselineDINModel(data_config=data_config, model_config=model_config, dense_dim=dense_dim)