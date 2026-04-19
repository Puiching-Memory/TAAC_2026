from __future__ import annotations

import torch
from torch import nn

from taac2026.domain.config import DataConfig, ModelConfig
from taac2026.domain.features import build_default_feature_schema
from taac2026.domain.types import BatchTensors
from taac2026.infrastructure.nn.embedding import TorchRecEmbeddingBagAdapter
from taac2026.infrastructure.nn.transformer import TaacTransformerBlock

from .data import TIME_GAP_BUCKET_COUNT


SPARSE_TABLE_NAMES = (
    "user_tokens",
    "context_tokens",
    "candidate_tokens",
    "candidate_post_tokens",
    "candidate_author_tokens",
)

SEQUENCE_FEATURE_KEYS = (
    "history_post_tokens",
    "history_action_tokens",
    "history_time_gap",
    "history_group_ids",
)


class SequentialTemporalBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.block = TaacTransformerBlock(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            dropout=dropout,
            norm_type="layernorm",
            ffn_type="gelu",
            attention_type="standard",
        )

    def forward(self, hidden_states: torch.Tensor, token_mask: torch.Tensor | None = None) -> torch.Tensor:
        return self.block(hidden_states, token_mask)


class DeepContextNetModel(nn.Module):
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

        self.time_embedding = nn.Embedding(TIME_GAP_BUCKET_COUNT + 1, model_config.hidden_dim, padding_idx=0)
        self.group_embedding = nn.Embedding(len(data_config.sequence_names) + 1, model_config.hidden_dim, padding_idx=0)
        self.global_token = nn.Parameter(torch.randn(1, 1, model_config.hidden_dim) * 0.02)

        self.user_projection = nn.Sequential(
            nn.Linear(model_config.hidden_dim * 3, model_config.hidden_dim),
            nn.LayerNorm(model_config.hidden_dim),
            nn.GELU(),
        )
        self.item_projection = nn.Sequential(
            nn.Linear(model_config.hidden_dim * 3, model_config.hidden_dim),
            nn.LayerNorm(model_config.hidden_dim),
            nn.GELU(),
        )
        self.bottleneck = nn.Linear(model_config.hidden_dim, model_config.hidden_dim)
        self.classifier = nn.Linear(model_config.hidden_dim, 1)
        self.blocks = nn.ModuleList(
            [
                SequentialTemporalBlock(
                    hidden_dim=model_config.hidden_dim,
                    num_heads=model_config.num_heads,
                    ffn_dim=int(model_config.hidden_dim * model_config.ffn_multiplier),
                    dropout=model_config.dropout,
                )
                for _ in range(model_config.num_layers)
            ]
        )
        self.dense_projection = nn.Sequential(
            nn.Linear(dense_dim, model_config.hidden_dim),
            nn.LayerNorm(model_config.hidden_dim),
            nn.GELU(),
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
        history_action_tokens, _ = self._dense_sequence_tokens(sequence_by_key, "history_action_tokens")
        history_time_gap, _ = self._dense_sequence_tokens(sequence_by_key, "history_time_gap")
        history_group_ids, _ = self._dense_sequence_tokens(sequence_by_key, "history_group_ids")
        sparse_summaries = self._pooled_sparse_summaries(batch)

        history_mask = self._slice_recent(history_mask)
        history_post_tokens = self._slice_recent(history_post_tokens)
        history_action_tokens = self._slice_recent(history_action_tokens)
        history_time_gap = self._slice_recent(history_time_gap)
        history_group_ids = self._slice_recent(history_group_ids)

        dense_summary = self.dense_projection(batch.dense_features)
        user_node = self.user_projection(
            torch.cat(
                [sparse_summaries["user_tokens"], sparse_summaries["context_tokens"], dense_summary],
                dim=-1,
            )
        )

        item_node = self.item_projection(
            torch.cat(
                [
                    sparse_summaries["candidate_post_tokens"],
                    sparse_summaries["candidate_author_tokens"],
                    sparse_summaries["candidate_tokens"],
                ],
                dim=-1,
            )
        )

        sequence_tokens = (
            self._embed_tokens(history_post_tokens)
            + self._embed_tokens(history_action_tokens)
            + self.time_embedding(history_time_gap.clamp(min=0, max=TIME_GAP_BUCKET_COUNT))
            + self.group_embedding(history_group_ids.clamp(min=0, max=self.group_embedding.num_embeddings - 1))
        )
        sequence_tokens = sequence_tokens * history_mask.unsqueeze(-1).float()

        batch_size = batch.batch_size
        cls_token = self.global_token.expand(batch_size, -1, -1)
        combined_tokens = torch.cat(
            [
                cls_token,
                user_node.unsqueeze(1),
                item_node.unsqueeze(1),
                sequence_tokens,
            ],
            dim=1,
        )
        token_mask = torch.cat(
            [
                torch.ones(batch_size, 3, dtype=torch.bool, device=combined_tokens.device),
                history_mask,
            ],
            dim=1,
        )

        for block in self.blocks:
            combined_tokens = block(combined_tokens, token_mask)

        latent = combined_tokens[:, 0, :]
        logits = self.classifier(torch.relu(self.bottleneck(latent))).squeeze(-1)
        return logits


def build_model_component(data_config: DataConfig, model_config: ModelConfig, dense_dim: int) -> DeepContextNetModel:
    return DeepContextNetModel(data_config=data_config, model_config=model_config, dense_dim=dense_dim)