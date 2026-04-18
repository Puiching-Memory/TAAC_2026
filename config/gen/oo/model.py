from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from taac2026.domain.config import DataConfig, ModelConfig
from taac2026.domain.features import build_default_feature_schema
from taac2026.domain.types import BatchTensors
from taac2026.infrastructure.nn.embedding import TorchRecEmbeddingBagAdapter
from taac2026.infrastructure.nn.hstu import (
    FourierTimeEncoding,
    RelativeTimeBias,
    TimeAwareHSTU,
)

from .data import TIME_GAP_BUCKET_COUNT
from .utils import masked_mean


SPARSE_TABLE_NAMES = (
    "user_tokens",
    "context_tokens",
    "candidate_tokens",
    "candidate_post_tokens",
    "candidate_author_tokens",
)


class FeatureInteractionEncoder(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, dropout: float, expansion_factor: int = 4) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.expansion_factor = expansion_factor
        self.down_projection = nn.Linear(input_dim, output_dim, bias=False)
        self.projection = nn.Sequential(
            nn.Linear(output_dim, output_dim * expansion_factor, bias=False),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        _, seq_len, _ = hidden_states.shape
        hidden_states = self.down_projection(hidden_states)
        expanded = self.projection(hidden_states).view(-1, seq_len, self.expansion_factor, self.output_dim)
        return torch.einsum("bsd,bsrd->bsd", hidden_states, expanded) + hidden_states


class OOModel(nn.Module):
    def __init__(self, data_config: DataConfig, model_config: ModelConfig, dense_dim: int) -> None:
        super().__init__()
        self.hidden_dim = model_config.hidden_dim
        self.sequence_count = len(data_config.sequence_names)
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
        self.dense_projection = nn.Sequential(
            nn.Linear(dense_dim, model_config.hidden_dim),
            nn.LayerNorm(model_config.hidden_dim),
            nn.SiLU(),
        )
        self.sequence_group_embedding = nn.Embedding(self.sequence_count + 1, model_config.hidden_dim, padding_idx=0)
        self.time_gap_embedding = nn.Embedding(TIME_GAP_BUCKET_COUNT + 1, model_config.hidden_dim, padding_idx=0)
        self.sequence_encoder = FeatureInteractionEncoder(model_config.hidden_dim * 9, model_config.hidden_dim, model_config.dropout)
        self.candidate_encoder = FeatureInteractionEncoder(model_config.hidden_dim * 3, model_config.hidden_dim, model_config.dropout)
        self.time_abs_encoding = FourierTimeEncoding(
            hidden_dim=model_config.hidden_dim,
            num_frequencies=8,
            min_period=1.0,
            max_period=float(TIME_GAP_BUCKET_COUNT + data_config.max_seq_len + 1),
        )
        self.rel_time_bias = RelativeTimeBias(num_buckets=64)
        self.attention_layers = nn.ModuleList(
            [
                TimeAwareHSTU(
                    hidden_dim=model_config.hidden_dim,
                    num_heads=model_config.num_heads,
                    dropout=model_config.attention_dropout,
                )
                for _ in range(model_config.num_layers)
            ]
        )
        head_hidden_dim = model_config.head_hidden_dim or model_config.hidden_dim * 4
        self.score_head = nn.Sequential(
            nn.LayerNorm(model_config.hidden_dim * 3),
            nn.Linear(model_config.hidden_dim * 3, head_hidden_dim),
            nn.SiLU(),
            nn.Dropout(model_config.dropout),
            nn.Linear(head_hidden_dim, 1),
        )

    def embed_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.token_projection(self.token_embedding(tokens))

    def _require_sparse_features(self, batch: BatchTensors):
        if batch.sparse_features is None:
            raise RuntimeError("Batch is missing required TorchRec sparse feature tensor: sparse_features")
        return batch.sparse_features

    def _pooled_sparse_summaries(self, batch: BatchTensors) -> dict[str, torch.Tensor]:
        pooled_sparse = self.sparse_embedding.forward_dict(self._require_sparse_features(batch))
        return {
            name: self.token_projection(pooled_sparse[name])
            for name in SPARSE_TABLE_NAMES
        }

    def build_sequence_inputs(self, batch: BatchTensors) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if batch.history_post_tokens is None or batch.history_author_tokens is None:
            raise RuntimeError("Batch is missing history entity token fields")
        if batch.history_action_tokens is None or batch.history_time_gap is None or batch.history_group_ids is None:
            raise RuntimeError("Batch is missing history metadata fields")

        sparse_summaries = self._pooled_sparse_summaries(batch)
        dense_summary = self.dense_projection(batch.dense_features)

        history_hidden = self.embed_tokens(batch.history_tokens)
        post_hidden = self.embed_tokens(batch.history_post_tokens)
        author_hidden = self.embed_tokens(batch.history_author_tokens)
        action_hidden = self.embed_tokens(batch.history_action_tokens)
        group_hidden = self.sequence_group_embedding(batch.history_group_ids)
        gap_hidden = self.time_gap_embedding(batch.history_time_gap.clamp(min=0, max=TIME_GAP_BUCKET_COUNT))

        seq_len = batch.history_tokens.shape[1]
        user_states = sparse_summaries["user_tokens"].unsqueeze(1).expand(-1, seq_len, -1)
        context_states = sparse_summaries["context_tokens"].unsqueeze(1).expand(-1, seq_len, -1)
        dense_states = dense_summary.unsqueeze(1).expand(-1, seq_len, -1)
        sequence_inputs = torch.cat(
            [
                user_states,
                context_states,
                dense_states,
                history_hidden,
                post_hidden,
                author_hidden,
                action_hidden,
                gap_hidden,
                group_hidden,
            ],
            dim=-1,
        )
        sequence_states = self.sequence_encoder(sequence_inputs) * batch.history_mask.unsqueeze(-1).float()

        sequence_positions = torch.arange(seq_len, device=sequence_states.device, dtype=torch.float32).unsqueeze(0)
        recency = (
            float(TIME_GAP_BUCKET_COUNT + 1) - batch.history_time_gap.float().clamp(min=0.0, max=float(TIME_GAP_BUCKET_COUNT + 1))
        ) / float(TIME_GAP_BUCKET_COUNT + 1)
        pseudo_timestamps = (sequence_positions + recency) * batch.history_mask.float()
        return sequence_states, batch.history_mask, pseudo_timestamps

    def encode_sequence(self, batch: BatchTensors) -> torch.Tensor:
        sequence_states, sequence_mask, pseudo_timestamps = self.build_sequence_inputs(batch)
        sequence_states = sequence_states * math.sqrt(self.hidden_dim)
        time_states = self.time_abs_encoding(pseudo_timestamps) * sequence_mask.unsqueeze(-1).float()
        sequence_states = sequence_states + time_states

        seq_len = sequence_states.shape[1]
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=sequence_states.device))
        attention_mask = causal_mask.unsqueeze(0) & sequence_mask.unsqueeze(1) & sequence_mask.unsqueeze(2)
        rel_ts_bias = self.rel_time_bias(pseudo_timestamps, sequence_mask)
        for layer in self.attention_layers:
            sequence_states = layer(sequence_states, attention_mask=attention_mask, rel_ts_bias=rel_ts_bias)

        sequence_states = F.normalize(sequence_states, dim=-1)
        last_indices = sequence_mask.long().sum(dim=1).clamp_min(1) - 1
        batch_indices = torch.arange(batch.batch_size, device=sequence_states.device)
        return sequence_states[batch_indices, last_indices]

    def encode_candidate(self, batch: BatchTensors) -> torch.Tensor:
        sparse_summaries = self._pooled_sparse_summaries(batch)

        candidate_inputs = torch.cat(
            [
                sparse_summaries["candidate_tokens"],
                sparse_summaries["candidate_post_tokens"],
                sparse_summaries["candidate_author_tokens"],
            ],
            dim=-1,
        ).unsqueeze(1)
        candidate_states = self.candidate_encoder(candidate_inputs).squeeze(1)
        return F.normalize(candidate_states, dim=-1)

    def forward(self, batch: BatchTensors) -> torch.Tensor:
        sequence_repr = self.encode_sequence(batch)
        candidate_repr = self.encode_candidate(batch)
        dot_logits = (sequence_repr * candidate_repr).sum(dim=-1)
        match_features = torch.cat([sequence_repr, candidate_repr, sequence_repr * candidate_repr], dim=-1)
        mlp_logits = self.score_head(match_features).squeeze(-1)
        return dot_logits + mlp_logits


def build_model_component(data_config: DataConfig, model_config: ModelConfig, dense_dim: int) -> OOModel:
    return OOModel(data_config=data_config, model_config=model_config, dense_dim=dense_dim)