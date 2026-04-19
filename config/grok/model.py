from __future__ import annotations

import math

import torch
from torch import nn

from taac2026.domain.config import DataConfig, ModelConfig
from taac2026.domain.features import build_default_feature_schema
from taac2026.domain.types import BatchTensors
from taac2026.infrastructure.nn.embedding import TorchRecEmbeddingBagAdapter
from taac2026.infrastructure.nn.heads import ClassificationHead
from taac2026.infrastructure.nn.norms import RMSNorm
from taac2026.infrastructure.nn.pooling import TargetAwarePool, masked_mean

from .data import TIME_GAP_BUCKET_COUNT


SPARSE_TABLE_NAMES = (
    "user_tokens",
    "context_tokens",
    "candidate_tokens",
    "candidate_post_tokens",
    "candidate_author_tokens",
)

SEQUENCE_FEATURE_KEYS = (
    "history_tokens",
    "history_post_tokens",
    "history_author_tokens",
    "history_action_tokens",
    "history_time_gap",
    "history_group_ids",
)


def make_grok_attention_mask(
    user_tokens: int,
    history_tokens: int,
    candidate_tokens: int,
    device: torch.device,
) -> torch.Tensor:
    total_tokens = user_tokens + history_tokens + candidate_tokens
    attention_mask = torch.zeros(total_tokens, total_tokens, dtype=torch.bool, device=device)
    prefix_tokens = user_tokens + history_tokens
    if prefix_tokens > 0:
        attention_mask[:prefix_tokens, :prefix_tokens] = torch.tril(
            torch.ones(prefix_tokens, prefix_tokens, dtype=torch.bool, device=device)
        )
    candidate_start = prefix_tokens
    for query_index in range(candidate_start, total_tokens):
        attention_mask[query_index, :prefix_tokens] = True
        attention_mask[query_index, query_index] = True
    return attention_mask


def _ffn_size(hidden_dim: int, widening_factor: float) -> int:
    width = int(widening_factor * hidden_dim) * 2 // 3
    return width + (8 - width) % 8


def _rotate_half(hidden_states: torch.Tensor) -> torch.Tensor:
    first_half, second_half = torch.chunk(hidden_states, 2, dim=-1)
    return torch.cat([-second_half, first_half], dim=-1)


def _apply_rotary_embedding(hidden_states: torch.Tensor) -> torch.Tensor:
    _, sequence_length, _, head_dim = hidden_states.shape
    if head_dim % 2 != 0:
        raise ValueError("head_dim must be even to apply rotary embeddings")
    positions = torch.arange(sequence_length, device=hidden_states.device, dtype=torch.float32)
    inverse_frequencies = 1.0 / (
        10000
        ** (torch.arange(0, head_dim, 2, device=hidden_states.device, dtype=torch.float32) / head_dim)
    )
    phase = torch.outer(positions, inverse_frequencies)
    cosine = torch.repeat_interleave(torch.cos(phase), 2, dim=-1).unsqueeze(0).unsqueeze(2)
    sine = torch.repeat_interleave(torch.sin(phase), 2, dim=-1).unsqueeze(0).unsqueeze(2)
    cosine = cosine.to(dtype=hidden_states.dtype)
    sine = sine.to(dtype=hidden_states.dtype)
    return hidden_states * cosine + _rotate_half(hidden_states) * sine


class GrokSelfAttention(nn.Module):
    def __init__(self, model_config: ModelConfig) -> None:
        super().__init__()
        if model_config.hidden_dim % model_config.num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.hidden_dim = model_config.hidden_dim
        self.num_heads = model_config.num_heads
        self.head_dim = model_config.hidden_dim // model_config.num_heads
        if self.head_dim % 2 != 0:
            raise ValueError("per-head hidden dimension must be even")
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.query_projection = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.key_projection = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.value_projection = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.output_projection = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.attention_dropout = nn.Dropout(model_config.attention_dropout)
        self.output_dropout = nn.Dropout(model_config.dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        allowed_attention: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, sequence_length, _ = hidden_states.shape
        query = self.query_projection(hidden_states).view(batch_size, sequence_length, self.num_heads, self.head_dim)
        key = self.key_projection(hidden_states).view(batch_size, sequence_length, self.num_heads, self.head_dim)
        value = self.value_projection(hidden_states).view(batch_size, sequence_length, self.num_heads, self.head_dim)

        query = _apply_rotary_embedding(query)
        key = _apply_rotary_embedding(key)

        attention_logits = torch.einsum("bthd,bshd->bhts", query, key).float() * self.scale
        attention_logits = 30.0 * torch.tanh(attention_logits / 30.0)
        attention_logits = attention_logits.masked_fill(
            ~allowed_attention.unsqueeze(0).unsqueeze(0),
            -1.0e4,
        )
        attention_logits = attention_logits.masked_fill(
            padding_mask.unsqueeze(1).unsqueeze(2),
            -1.0e4,
        )

        attention_weights = torch.softmax(attention_logits, dim=-1).to(dtype=query.dtype)
        attention_weights = self.attention_dropout(attention_weights)
        attended = torch.einsum("bhts,bshd->bthd", attention_weights, value)
        attended = attended.reshape(batch_size, sequence_length, self.hidden_dim)
        attended = self.output_projection(self.output_dropout(attended))
        return attended.masked_fill(padding_mask.unsqueeze(-1), 0.0)


class GrokFeedForward(nn.Module):
    def __init__(self, model_config: ModelConfig) -> None:
        super().__init__()
        feedforward_dim = _ffn_size(model_config.hidden_dim, model_config.ffn_multiplier)
        self.value_projection = nn.Linear(model_config.hidden_dim, feedforward_dim, bias=False)
        self.gate_projection = nn.Linear(model_config.hidden_dim, feedforward_dim, bias=False)
        self.output_projection = nn.Linear(feedforward_dim, model_config.hidden_dim, bias=False)
        self.dropout = nn.Dropout(model_config.dropout)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gated = torch.nn.functional.gelu(self.gate_projection(hidden_states)) * self.value_projection(hidden_states)
        return self.output_projection(self.dropout(gated))


class GrokBlock(nn.Module):
    def __init__(self, model_config: ModelConfig) -> None:
        super().__init__()
        self.pre_attention_norm = RMSNorm(model_config.hidden_dim)
        self.post_attention_norm = RMSNorm(model_config.hidden_dim)
        self.pre_ffn_norm = RMSNorm(model_config.hidden_dim)
        self.post_ffn_norm = RMSNorm(model_config.hidden_dim)
        self.attention = GrokSelfAttention(model_config)
        self.feed_forward = GrokFeedForward(model_config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        allowed_attention: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        attention_output = self.attention(self.pre_attention_norm(hidden_states), allowed_attention, padding_mask)
        hidden_states = hidden_states + self.post_attention_norm(attention_output)
        feed_forward_output = self.feed_forward(self.pre_ffn_norm(hidden_states))
        hidden_states = hidden_states + self.post_ffn_norm(feed_forward_output)
        return hidden_states.masked_fill(padding_mask.unsqueeze(-1), 0.0)


class GrokBaselineModel(nn.Module):
    def __init__(self, data_config: DataConfig, model_config: ModelConfig, dense_dim: int) -> None:
        super().__init__()
        self.history_capacity = len(data_config.sequence_names) * data_config.max_seq_len
        self.hidden_dim = model_config.hidden_dim
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
        self.time_gap_embedding = nn.Embedding(TIME_GAP_BUCKET_COUNT + 1, model_config.hidden_dim, padding_idx=0)
        self.history_group_embedding = nn.Embedding(len(data_config.sequence_names) + 1, model_config.hidden_dim, padding_idx=0)
        self.segment_embedding = nn.Embedding(3, model_config.hidden_dim)
        self.dense_projection = nn.Sequential(
            nn.Linear(dense_dim, model_config.hidden_dim),
            nn.LayerNorm(model_config.hidden_dim),
            nn.GELU(),
        )
        self.user_reduce = nn.Sequential(
            nn.Linear(model_config.hidden_dim * 4, model_config.hidden_dim),
            nn.LayerNorm(model_config.hidden_dim),
            nn.GELU(),
        )
        self.candidate_reduce = nn.Sequential(
            nn.Linear(model_config.hidden_dim * 4, model_config.hidden_dim),
            nn.LayerNorm(model_config.hidden_dim),
            nn.GELU(),
        )
        self.history_reduce = nn.Sequential(
            nn.Linear(model_config.hidden_dim * 6, model_config.hidden_dim),
            nn.LayerNorm(model_config.hidden_dim),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList([GrokBlock(model_config) for _ in range(model_config.num_layers)])
        self.final_norm = RMSNorm(model_config.hidden_dim)

        head_hidden_dim = model_config.head_hidden_dim or model_config.hidden_dim * 2
        self.readout_pool = TargetAwarePool(
            model_config.hidden_dim,
            scorer_hidden_dim=head_hidden_dim,
            activation="gelu",
            dropout=0.0,
            include_difference=False,
            include_absolute_difference=True,
            include_product=True,
        )
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

    def _require_sequence_features(self, batch: BatchTensors):
        if batch.sequence_features is None:
            raise RuntimeError("Batch is missing required TorchRec sparse feature tensor: sequence_features")
        return batch.sequence_features

    def _embed_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.token_projection(self.token_embedding(tokens))

    def _dense_sequence_tokens(self, sequence_by_key, name: str) -> tuple[torch.Tensor, torch.Tensor]:
        jagged = sequence_by_key[name]
        tokens = jagged.to_padded_dense(desired_length=self.history_capacity, padding_value=0).to(dtype=torch.long)
        lengths = jagged.lengths().to(device=tokens.device)
        positions = torch.arange(self.history_capacity, device=tokens.device).unsqueeze(0)
        mask = positions < lengths.unsqueeze(1)
        return tokens, mask

    def _sparse_feature_bundle(self, batch: BatchTensors) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        sparse_features = self._require_sparse_features(batch)
        pooled_sparse = self.sparse_embedding.forward_dict(sparse_features)
        sparse_by_key = sparse_features.to_dict()
        summaries = {
            name: self.token_projection(pooled_sparse[name])
            for name in SPARSE_TABLE_NAMES
        }
        valid = {
            name: sparse_by_key[name].lengths().gt(0).unsqueeze(1)
            for name in ("user_tokens", "candidate_post_tokens")
        }
        return summaries, valid

    def _target_aware_readout(
        self,
        candidate_query: torch.Tensor,
        memory_tokens: torch.Tensor,
        memory_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.readout_pool(candidate_query, memory_tokens, memory_mask)

    def forward(self, batch: BatchTensors) -> torch.Tensor:
        sequence_by_key = self._require_sequence_features(batch).to_dict()
        missing_keys = [name for name in SEQUENCE_FEATURE_KEYS if name not in sequence_by_key]
        if missing_keys:
            missing = ", ".join(missing_keys)
            raise RuntimeError(f"Batch sequence_features is missing required keys: {missing}")

        history_tokens, history_mask = self._dense_sequence_tokens(sequence_by_key, "history_tokens")
        history_post_tokens, _ = self._dense_sequence_tokens(sequence_by_key, "history_post_tokens")
        history_author_tokens, _ = self._dense_sequence_tokens(sequence_by_key, "history_author_tokens")
        history_action_tokens, _ = self._dense_sequence_tokens(sequence_by_key, "history_action_tokens")
        history_time_gap, _ = self._dense_sequence_tokens(sequence_by_key, "history_time_gap")
        history_group_ids, _ = self._dense_sequence_tokens(sequence_by_key, "history_group_ids")
        sparse_summaries, sparse_valid = self._sparse_feature_bundle(batch)

        dense_summary = self.dense_projection(batch.dense_features)
        user_representation = self.user_reduce(
            torch.cat(
                [
                    sparse_summaries["user_tokens"],
                    sparse_summaries["context_tokens"],
                    dense_summary,
                    sparse_summaries["user_tokens"] * sparse_summaries["context_tokens"],
                ],
                dim=-1,
            )
        )

        candidate_representation = self.candidate_reduce(
            torch.cat(
                [
                    sparse_summaries["candidate_post_tokens"],
                    sparse_summaries["candidate_author_tokens"],
                    sparse_summaries["candidate_tokens"],
                    sparse_summaries["candidate_post_tokens"] * sparse_summaries["candidate_author_tokens"],
                ],
                dim=-1,
            )
        )

        history_representation = self.history_reduce(
            torch.cat(
                [
                    self._embed_tokens(history_post_tokens),
                    self._embed_tokens(history_author_tokens),
                    self._embed_tokens(history_action_tokens),
                    self.time_gap_embedding(history_time_gap.clamp_max(TIME_GAP_BUCKET_COUNT)),
                    self._embed_tokens(history_tokens),
                    self.history_group_embedding(history_group_ids.clamp_max(self.history_group_embedding.num_embeddings - 1)),
                ],
                dim=-1,
            )
        )
        history_representation = history_representation * history_mask.unsqueeze(-1).float()

        batch_size = batch.labels.shape[0]
        user_valid = sparse_valid["user_tokens"]
        candidate_valid = sparse_valid["candidate_post_tokens"]
        user_segment = self.segment_embedding(
            torch.zeros((batch_size, 1), dtype=torch.long, device=batch.labels.device)
        )
        history_segment = self.segment_embedding(
            torch.ones((batch_size, history_representation.shape[1]), dtype=torch.long, device=batch.labels.device)
        )
        candidate_segment = self.segment_embedding(
            torch.full((batch_size, 1), 2, dtype=torch.long, device=batch.labels.device)
        )

        hidden_states = torch.cat(
            [
                user_representation.unsqueeze(1) + user_segment,
                history_representation + history_segment,
                candidate_representation.unsqueeze(1) + candidate_segment,
            ],
            dim=1,
        )
        padding_mask = torch.cat([~user_valid, ~history_mask, ~candidate_valid], dim=1)
        attention_mask = make_grok_attention_mask(
            user_tokens=1,
            history_tokens=history_representation.shape[1],
            candidate_tokens=1,
            device=hidden_states.device,
        )

        for block in self.blocks:
            hidden_states = block(hidden_states, attention_mask, padding_mask)
        hidden_states = self.final_norm(hidden_states)

        user_output = hidden_states[:, 0]
        history_output = hidden_states[:, 1 : 1 + history_representation.shape[1]]
        candidate_output = hidden_states[:, -1]
        target_context = self._target_aware_readout(
            candidate_output,
            torch.cat([user_output.unsqueeze(1), history_output], dim=1),
            torch.cat([user_valid, history_mask], dim=1),
        )
        history_summary = masked_mean(history_output, history_mask)
        static_context = 0.5 * (sparse_summaries["context_tokens"] + dense_summary)

        fused = torch.cat(
            [
                candidate_output,
                target_context,
                user_output,
                history_summary,
                static_context,
                candidate_output * target_context,
                torch.abs(candidate_output - user_output),
            ],
            dim=-1,
        )
        return self.output(fused).squeeze(-1)


def build_model_component(data_config: DataConfig, model_config: ModelConfig, dense_dim: int) -> GrokBaselineModel:
    return GrokBaselineModel(data_config=data_config, model_config=model_config, dense_dim=dense_dim)