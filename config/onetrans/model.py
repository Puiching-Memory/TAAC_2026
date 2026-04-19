from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn

from taac2026.domain.config import DataConfig, ModelConfig
from taac2026.domain.features import build_default_feature_schema
from taac2026.domain.types import BatchTensors
from taac2026.infrastructure.nn.embedding import TorchRecEmbeddingBagAdapter
from taac2026.infrastructure.nn.heads import ClassificationHead
from taac2026.infrastructure.nn.pooling import masked_mean
from taac2026.infrastructure.nn.transformer import TaacMixedCausalBlock

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



class AutoSplitNSTokenizer(nn.Module):
    def __init__(self, dense_dim: int, hidden_dim: int, ns_token_count: int, dropout: float) -> None:
        super().__init__()
        self.ns_token_count = ns_token_count
        self.hidden_dim = hidden_dim
        self.group_position_embedding = nn.Embedding(ns_token_count, hidden_dim)
        self.dense_projection = nn.Sequential(
            nn.Linear(dense_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        input_dim = hidden_dim * 6
        self.auto_split = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim * 4),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, ns_token_count * hidden_dim),
        )

    def forward(self, batch: BatchTensors, sparse_summaries: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        dense_summary = self.dense_projection(batch.dense_features)

        fused = torch.cat(
            [
                dense_summary,
                sparse_summaries["user_tokens"],
                sparse_summaries["context_tokens"],
                sparse_summaries["candidate_post_tokens"],
                sparse_summaries["candidate_author_tokens"],
                sparse_summaries["candidate_tokens"],
            ],
            dim=-1,
        )
        tokens = self.auto_split(fused).view(batch.batch_size, self.ns_token_count, self.hidden_dim)
        positions = torch.arange(self.ns_token_count, device=tokens.device)
        tokens = tokens + self.group_position_embedding(positions).unsqueeze(0)
        mask = torch.ones(batch.batch_size, self.ns_token_count, dtype=torch.bool, device=tokens.device)
        return tokens, mask


class UnifiedSequentialTokenizer(nn.Module):
    def __init__(
        self,
        max_sequence_tokens: int,
        history_capacity: int,
        hidden_dim: int,
        sequence_group_count: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.max_sequence_tokens = max_sequence_tokens
        self.history_capacity = history_capacity
        self.hidden_dim = hidden_dim
        self.event_projection = nn.Sequential(
            nn.LayerNorm(hidden_dim * 6),
            nn.Linear(hidden_dim * 6, hidden_dim * 4),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.time_gap_embedding = nn.Embedding(TIME_GAP_BUCKET_COUNT + 1, hidden_dim, padding_idx=0)
        self.sequence_group_embedding = nn.Embedding(sequence_group_count + 1, hidden_dim, padding_idx=0)
        self.sequence_position_embedding = nn.Embedding(max_sequence_tokens, hidden_dim)
        self.sep_token = nn.Parameter(torch.randn(hidden_dim) * 0.02)

    def _require_sequence_features(self, batch: BatchTensors):
        if batch.sequence_features is None:
            raise RuntimeError("Batch is missing required TorchRec sparse feature tensor: sequence_features")
        return batch.sequence_features

    def _dense_sequence_tokens(self, sequence_by_key, name: str) -> tuple[torch.Tensor, torch.Tensor]:
        jagged = sequence_by_key[name]
        tokens = jagged.to_padded_dense(desired_length=self.history_capacity, padding_value=0).to(dtype=torch.long)
        lengths = jagged.lengths().to(device=tokens.device)
        positions = torch.arange(self.history_capacity, device=tokens.device).unsqueeze(0)
        mask = positions < lengths.unsqueeze(1)
        return tokens, mask

    def forward(
        self,
        batch: BatchTensors,
        token_embedding: Callable[[torch.Tensor], torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
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

        history_hidden = token_embedding(history_tokens)
        post_hidden = token_embedding(history_post_tokens)
        author_hidden = token_embedding(history_author_tokens)
        action_hidden = token_embedding(history_action_tokens)
        time_hidden = self.time_gap_embedding(history_time_gap.clamp(min=0, max=TIME_GAP_BUCKET_COUNT))
        group_hidden = self.sequence_group_embedding(history_group_ids)
        event_inputs = torch.cat(
            [
                history_hidden,
                post_hidden,
                author_hidden,
                action_hidden,
                time_hidden,
                group_hidden,
            ],
            dim=-1,
        )
        event_tokens = self.event_projection(event_inputs)
        return self.merge_with_sep(event_tokens, history_group_ids, history_mask)

    def merge_with_sep(
        self,
        event_tokens: torch.Tensor,
        history_group_ids: torch.Tensor,
        history_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = event_tokens.shape[0]
        device = event_tokens.device
        merged_tokens = event_tokens.new_zeros(batch_size, self.max_sequence_tokens, self.hidden_dim)
        merged_mask = torch.zeros(batch_size, self.max_sequence_tokens, dtype=torch.bool, device=device)

        for batch_index in range(batch_size):
            valid_positions = torch.nonzero(history_mask[batch_index], as_tuple=False).squeeze(-1).tolist()
            if not valid_positions:
                continue
            pieces: list[torch.Tensor] = []
            for offset, position in enumerate(valid_positions):
                pieces.append(event_tokens[batch_index, position])
                if offset + 1 >= len(valid_positions):
                    continue
                next_position = valid_positions[offset + 1]
                if int(history_group_ids[batch_index, position]) != int(history_group_ids[batch_index, next_position]):
                    pieces.append(self.sep_token)

            merged_length = min(len(pieces), self.max_sequence_tokens)
            start = self.max_sequence_tokens - merged_length
            merged_tokens[batch_index, start:] = torch.stack(pieces[-merged_length:], dim=0)
            merged_mask[batch_index, start:] = True

        positions = torch.arange(self.max_sequence_tokens, device=device)
        merged_tokens = merged_tokens + self.sequence_position_embedding(positions).unsqueeze(0)
        merged_tokens = merged_tokens * merged_mask.unsqueeze(-1).float()
        return merged_tokens, merged_mask


class OneTransModel(nn.Module):
    def __init__(self, data_config: DataConfig, model_config: ModelConfig, dense_dim: int) -> None:
        super().__init__()
        self.sequence_group_count = len(data_config.sequence_names)
        self.history_capacity = self.sequence_group_count * data_config.max_seq_len
        self.max_sequence_tokens = max(1, self.history_capacity * 2 - 1)
        self.ns_token_count = max(1, model_config.segment_count)
        self.hidden_dim = model_config.hidden_dim
        self.ffn_dim = int(model_config.hidden_dim * model_config.ffn_multiplier)
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
        self.sequential_tokenizer = UnifiedSequentialTokenizer(
            max_sequence_tokens=self.max_sequence_tokens,
            history_capacity=self.history_capacity,
            hidden_dim=model_config.hidden_dim,
            sequence_group_count=self.sequence_group_count,
            dropout=model_config.dropout,
        )
        self.ns_tokenizer = AutoSplitNSTokenizer(
            dense_dim=dense_dim,
            hidden_dim=model_config.hidden_dim,
            ns_token_count=self.ns_token_count,
            dropout=model_config.dropout,
        )
        self.blocks = nn.ModuleList(
            [
                TaacMixedCausalBlock(
                    hidden_dim=model_config.hidden_dim,
                    num_heads=model_config.num_heads,
                    ffn_dim=self.ffn_dim,
                    ns_token_count=self.ns_token_count,
                    dropout=model_config.dropout,
                    attention_dropout=model_config.attention_dropout,
                )
                for _ in range(model_config.num_layers)
            ]
        )
        head_hidden_dim = model_config.head_hidden_dim or model_config.hidden_dim * 4
        self.output_head = ClassificationHead(
            input_dim=(self.ns_token_count * 2) * model_config.hidden_dim,
            hidden_dims=[head_hidden_dim, model_config.hidden_dim * 2],
            activation="silu",
            dropout=[model_config.dropout, model_config.dropout],
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

    def make_pyramid_schedule(self, layer_count: int) -> list[int]:
        if layer_count <= 0:
            return []
        schedule: list[int] = []
        for layer_index in range(1, layer_count + 1):
            ratio = layer_index / layer_count
            keep = round(self.max_sequence_tokens + ratio * (self.ns_token_count - self.max_sequence_tokens))
            keep = max(self.ns_token_count, min(self.max_sequence_tokens, keep))
            if schedule:
                keep = min(keep, schedule[-1])
            schedule.append(keep)
        schedule[-1] = self.ns_token_count
        return schedule

    def forward(self, batch: BatchTensors) -> torch.Tensor:
        sequence_tokens, sequence_mask = self.sequential_tokenizer(batch, self.embed_tokens)
        ns_tokens, ns_mask = self.ns_tokenizer(batch, self._pooled_sparse_summaries(batch))
        pyramid_schedule = self.make_pyramid_schedule(len(self.blocks))

        for block, next_sequence_length in zip(self.blocks, pyramid_schedule, strict=True):
            sequence_tokens, sequence_mask, ns_tokens, ns_mask = block(
                sequence_tokens,
                sequence_mask,
                ns_tokens,
                ns_mask,
                next_sequence_length=next_sequence_length,
            )

        fused = torch.cat([sequence_tokens, ns_tokens], dim=1).reshape(batch.batch_size, -1)
        logits = self.output_head(fused)
        return logits.squeeze(-1)


def build_model_component(data_config: DataConfig, model_config: ModelConfig, dense_dim: int) -> OneTransModel:
    return OneTransModel(data_config=data_config, model_config=model_config, dense_dim=dense_dim)