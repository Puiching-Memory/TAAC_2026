from __future__ import annotations

import math

import torch
from torch import nn

from taac2026.domain.config import DataConfig, ModelConfig
from taac2026.domain.features import build_default_feature_schema
from taac2026.domain.types import BatchTensors
from taac2026.infrastructure.nn.embedding import TorchRecEmbeddingCollectionAdapter
from taac2026.infrastructure.nn.heads import ClassificationHead
from taac2026.infrastructure.nn.triton_attention import TritonAttention

from .data import AUTHOR_TOKEN_COUNT


SPARSE_FEATURE_WIDTHS = {
    "user_tokens": "max_feature_tokens",
    "context_tokens": "max_feature_tokens",
    "candidate_post_tokens": "max_event_features",
    "candidate_author_tokens": "author_token_count",
    "candidate_tokens": "candidate_token_count",
}

SEQUENCE_FEATURE_KEYS = (
    "history_tokens",
    "history_post_tokens",
    "history_author_tokens",
    "history_action_tokens",
    "history_time_gap",
    "history_group_ids",
)


def _gather_recent_tokens(
    sequence_tokens: torch.Tensor,
    sequence_mask: torch.Tensor,
    count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, _, hidden_dim = sequence_tokens.shape
    if count <= 0:
        empty_tokens = sequence_tokens.new_zeros(batch_size, 0, hidden_dim)
        empty_mask = torch.zeros(batch_size, 0, dtype=torch.bool, device=sequence_tokens.device)
        return empty_tokens, empty_mask
    recent_tokens = sequence_tokens.new_zeros(batch_size, count, hidden_dim)
    recent_mask = torch.zeros(batch_size, count, dtype=torch.bool, device=sequence_tokens.device)
    for batch_index in range(batch_size):
        valid_positions = torch.nonzero(sequence_mask[batch_index], as_tuple=False).squeeze(-1)
        if valid_positions.numel() == 0:
            continue
        selected_positions = valid_positions[-count:]
        target_offset = count - selected_positions.numel()
        recent_tokens[batch_index, target_offset:] = sequence_tokens[batch_index, selected_positions]
        recent_mask[batch_index, target_offset:] = True
    return recent_tokens, recent_mask


class SelfGating(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.value = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.gate = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, hidden_states: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        gated = torch.sigmoid(self.gate(hidden_states)) * self.value(hidden_states)
        if mask is not None:
            gated = gated * mask.unsqueeze(-1).float()
        return gated


class StandardSelfAttention(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.attention = TritonAttention(hidden_dim, num_heads, dropout=0.0)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        attended = self.attention(
            hidden_states,
            hidden_states,
            hidden_states,
            key_mask=valid_mask,
            query_mask=valid_mask,
        )
        attended = self.dropout(attended)
        return attended * valid_mask.unsqueeze(-1).float()


class LinearCompressedEmbedding(nn.Module):
    def __init__(self, input_tokens: int, output_tokens: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.compression = nn.Parameter(torch.empty(input_tokens, output_tokens))
        nn.init.xavier_uniform_(self.compression)
        self.gating = SelfGating(hidden_dim, dropout)
        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        compressed = torch.einsum("bnd,nm->bmd", hidden_states, self.compression)
        return self.output_norm(self.gating(compressed))


class SequencePreprocessor(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        input_dim = hidden_dim * 6
        self.mask_network = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, input_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim, input_dim),
        )
        self.value_network = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim * 4),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.gating = SelfGating(hidden_dim, dropout)
        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        history_hidden: torch.Tensor,
        post_hidden: torch.Tensor,
        author_hidden: torch.Tensor,
        action_hidden: torch.Tensor,
        time_gap_hidden: torch.Tensor,
        group_hidden: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        valid_weights = valid_mask.unsqueeze(-1).float()
        merged = torch.cat(
            [
                history_hidden,
                post_hidden,
                author_hidden,
                action_hidden,
                time_gap_hidden,
                group_hidden,
            ],
            dim=-1,
        )
        merged = merged * valid_weights
        masked = merged * torch.sigmoid(self.mask_network(merged))
        compressed = self.value_network(masked)
        fused = history_hidden + self.gating(compressed, valid_mask)
        return self.output_norm(fused) * valid_weights, valid_mask


class NonSequenceSummary(nn.Module):
    def __init__(self, input_tokens: int, summary_tokens: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.lce = LinearCompressedEmbedding(input_tokens, summary_tokens, hidden_dim, dropout)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lce(hidden_states)


class PoolingMultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, num_queries: int, dropout: float) -> None:
        super().__init__()
        self.num_queries = num_queries
        self.queries = nn.Parameter(torch.randn(num_queries, hidden_dim) * 0.02)
        self.attention = TritonAttention(hidden_dim, num_heads, dropout=dropout)
        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(self, hidden_states: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        if self.num_queries <= 0:
            return hidden_states.new_zeros(hidden_states.shape[0], 0, hidden_states.shape[-1])
        queries = self.queries.unsqueeze(0).expand(hidden_states.shape[0], -1, -1)
        pooled = self.attention(
            queries,
            hidden_states,
            hidden_states,
            key_mask=valid_mask,
        )
        return self.output_norm(queries + pooled)


class SequenceSummary(nn.Module):
    def __init__(
        self,
        cls_token_count: int,
        pma_token_count: int,
        recent_token_count: int,
        hidden_dim: int,
        num_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.cls_token_count = cls_token_count
        self.pma_token_count = pma_token_count
        self.recent_token_count = recent_token_count
        self.pma = PoolingMultiHeadAttention(hidden_dim, num_heads, pma_token_count, dropout)
        self.gating = SelfGating(hidden_dim, dropout)

    def forward(self, hidden_states: torch.Tensor, valid_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        cls_tokens = hidden_states[:, : self.cls_token_count]
        cls_mask = valid_mask[:, : self.cls_token_count]

        pma_tokens = self.pma(hidden_states, valid_mask)
        pma_mask = torch.ones(
            hidden_states.shape[0],
            self.pma_token_count,
            dtype=torch.bool,
            device=hidden_states.device,
        )

        sequence_tokens = hidden_states[:, self.cls_token_count :]
        sequence_mask = valid_mask[:, self.cls_token_count :]
        recent_tokens, recent_mask = _gather_recent_tokens(sequence_tokens, sequence_mask, self.recent_token_count)

        summary_tokens = torch.cat([cls_tokens, pma_tokens, recent_tokens], dim=1)
        summary_mask = torch.cat([cls_mask, pma_mask, recent_mask], dim=1)
        return self.gating(summary_tokens, summary_mask), summary_mask


class BehaviorAwareInteractionArch(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float, attention_dropout: float) -> None:
        super().__init__()
        self.self_attention = TritonAttention(hidden_dim, num_heads, dropout=attention_dropout)
        self.cross_attention = TritonAttention(hidden_dim, num_heads, dropout=attention_dropout)
        self.value_projection = nn.Sequential(
            nn.LayerNorm(hidden_dim * 4),
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.gate_projection = nn.Sequential(
            nn.LayerNorm(hidden_dim * 4),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Sigmoid(),
        )
        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        valid_mask: torch.Tensor,
        sequence_summary: torch.Tensor,
        sequence_summary_mask: torch.Tensor,
    ) -> torch.Tensor:
        self_context = self.self_attention(
            hidden_states,
            hidden_states,
            hidden_states,
            key_mask=valid_mask,
            query_mask=valid_mask,
        )
        sequence_context = self.cross_attention(
            hidden_states,
            sequence_summary,
            sequence_summary,
            key_mask=sequence_summary_mask,
            query_mask=valid_mask,
        )
        fused_inputs = torch.cat(
            [
                hidden_states,
                self_context,
                sequence_context,
                self_context * sequence_context,
            ],
            dim=-1,
        )
        delta = self.gate_projection(fused_inputs) * self.value_projection(fused_inputs)
        updated = self.output_norm(hidden_states + delta)
        return updated * valid_mask.unsqueeze(-1).float()


class PersonalizedFFN(nn.Module):
    def __init__(self, cls_token_count: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        input_dim = cls_token_count * hidden_dim
        self.weight_generator = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim * hidden_dim),
        )
        self.bias_generator = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(self, hidden_states: torch.Tensor, non_sequence_summary: torch.Tensor) -> torch.Tensor:
        batch_size, _, hidden_dim = hidden_states.shape
        summary_context = non_sequence_summary.reshape(batch_size, -1)
        projection_weight = self.weight_generator(summary_context).view(batch_size, hidden_dim, hidden_dim)
        projection_weight = projection_weight / math.sqrt(hidden_dim)
        projection_bias = self.bias_generator(summary_context).unsqueeze(1)
        projected = torch.einsum("bij,btj->bti", projection_weight, hidden_states)
        return self.output_norm(hidden_states + projected + projection_bias)


class ContextAwareSequenceArch(nn.Module):
    def __init__(
        self,
        cls_token_count: int,
        hidden_dim: int,
        num_heads: int,
        dropout: float,
        attention_dropout: float,
    ) -> None:
        super().__init__()
        self.pffn = PersonalizedFFN(cls_token_count, hidden_dim, dropout)
        self.attention_norm = nn.LayerNorm(hidden_dim)
        self.attention = StandardSelfAttention(hidden_dim, num_heads, attention_dropout)
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        valid_mask: torch.Tensor,
        non_sequence_summary: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.pffn(hidden_states, non_sequence_summary)
        attention_output = self.attention(self.attention_norm(hidden_states), valid_mask)
        updated = self.output_norm(hidden_states + self.dropout(attention_output))
        return updated * valid_mask.unsqueeze(-1).float()


class InterFormerLayer(nn.Module):
    def __init__(
        self,
        cls_token_count: int,
        hidden_dim: int,
        num_heads: int,
        dropout: float,
        attention_dropout: float,
    ) -> None:
        super().__init__()
        self.interaction_arch = BehaviorAwareInteractionArch(hidden_dim, num_heads, dropout, attention_dropout)
        self.sequence_arch = ContextAwareSequenceArch(cls_token_count, hidden_dim, num_heads, dropout, attention_dropout)

    def forward(
        self,
        non_sequence_hidden: torch.Tensor,
        non_sequence_mask: torch.Tensor,
        sequence_hidden: torch.Tensor,
        sequence_mask: torch.Tensor,
        non_sequence_summary: torch.Tensor,
        sequence_summary: torch.Tensor,
        sequence_summary_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        updated_non_sequence = self.interaction_arch(
            non_sequence_hidden,
            non_sequence_mask,
            sequence_summary,
            sequence_summary_mask,
        )
        updated_sequence = self.sequence_arch(sequence_hidden, sequence_mask, non_sequence_summary)
        return updated_non_sequence, updated_sequence


class InterFormerModel(nn.Module):
    def __init__(self, data_config: DataConfig, model_config: ModelConfig, dense_dim: int) -> None:
        super().__init__()
        self.feature_schema = build_default_feature_schema(data_config, model_config)
        self.cls_token_count = max(1, model_config.memory_slots)
        self.pma_token_count = max(0, model_config.num_queries)
        self.recent_token_count = max(0, model_config.recent_seq_len)
        self.sequence_count = len(data_config.sequence_names)
        self.history_capacity = self.sequence_count * data_config.max_seq_len
        self.hidden_dim = model_config.hidden_dim
        self.max_feature_tokens = data_config.max_feature_tokens
        self.max_event_features = max(1, data_config.max_event_features)
        self.author_token_count = AUTHOR_TOKEN_COUNT
        self.candidate_token_count = 1
        self.non_sequence_token_count = (
            1
            + data_config.max_feature_tokens
            + data_config.max_feature_tokens
            + max(1, data_config.max_event_features)
            + AUTHOR_TOKEN_COUNT
            + 1
        )

        self.sparse_embedding = TorchRecEmbeddingCollectionAdapter(
            self.feature_schema,
            table_names=tuple(SPARSE_FEATURE_WIDTHS),
        )
        self.sequence_embedding = TorchRecEmbeddingCollectionAdapter(
            self.feature_schema,
            table_names=SEQUENCE_FEATURE_KEYS,
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
        self.non_sequence_group_embeddings = nn.Parameter(torch.empty(5, model_config.hidden_dim))
        nn.init.normal_(self.non_sequence_group_embeddings, mean=0.0, std=0.02)
        self.sequence_preprocessor = SequencePreprocessor(model_config.hidden_dim, model_config.dropout)
        self.non_sequence_summary = NonSequenceSummary(
            input_tokens=self.non_sequence_token_count,
            summary_tokens=self.cls_token_count,
            hidden_dim=model_config.hidden_dim,
            dropout=model_config.dropout,
        )
        self.sequence_summary = SequenceSummary(
            cls_token_count=self.cls_token_count,
            pma_token_count=self.pma_token_count,
            recent_token_count=self.recent_token_count,
            hidden_dim=model_config.hidden_dim,
            num_heads=model_config.num_heads,
            dropout=model_config.dropout,
        )
        self.layers = nn.ModuleList(
            [
                InterFormerLayer(
                    cls_token_count=self.cls_token_count,
                    hidden_dim=model_config.hidden_dim,
                    num_heads=model_config.num_heads,
                    dropout=model_config.dropout,
                    attention_dropout=model_config.attention_dropout,
                )
                for _ in range(model_config.num_layers)
            ]
        )

        summary_input_dim = model_config.hidden_dim * (
            self.cls_token_count + self.cls_token_count + self.pma_token_count + self.recent_token_count
        )
        head_hidden_dim = model_config.head_hidden_dim or model_config.hidden_dim * 4
        self.classifier = ClassificationHead(
            input_dim=summary_input_dim,
            hidden_dims=[head_hidden_dim, model_config.hidden_dim * 2],
            activation="silu",
            dropout=[model_config.dropout, model_config.dropout],
        )

    def _require_sparse_features(self, batch: BatchTensors):
        if batch.sparse_features is None:
            raise RuntimeError("Batch is missing required TorchRec sparse feature tensor: sparse_features")
        return batch.sparse_features

    def _require_sequence_features(self, batch: BatchTensors):
        if batch.sequence_features is None:
            raise RuntimeError("Batch is missing required TorchRec sparse feature tensor: sequence_features")
        return batch.sequence_features

    def _dense_feature_hidden(self, feature_dict, name: str, desired_length: int) -> tuple[torch.Tensor, torch.Tensor]:
        jagged = feature_dict[name]
        hidden_states = jagged.to_padded_dense(desired_length=desired_length, padding_value=0.0)
        lengths = jagged.lengths().to(device=hidden_states.device)
        positions = torch.arange(desired_length, device=hidden_states.device).unsqueeze(0)
        mask = positions < lengths.unsqueeze(1)
        return self.token_projection(hidden_states), mask

    def _dense_sparse_hidden(self, sparse_by_key, name: str) -> tuple[torch.Tensor, torch.Tensor]:
        return self._dense_feature_hidden(sparse_by_key, name, getattr(self, SPARSE_FEATURE_WIDTHS[name]))

    def _dense_sequence_hidden(self, sequence_by_key, name: str) -> tuple[torch.Tensor, torch.Tensor]:
        return self._dense_feature_hidden(sequence_by_key, name, self.history_capacity)

    def _add_non_sequence_group(self, hidden_states: torch.Tensor, group_index: int) -> torch.Tensor:
        group_embedding = self.non_sequence_group_embeddings[group_index].view(1, 1, -1)
        return hidden_states + group_embedding

    def _build_non_sequence_inputs(self, batch: BatchTensors) -> tuple[torch.Tensor, torch.Tensor]:
        sparse_by_key = self.sparse_embedding.forward_dict(self._require_sparse_features(batch))
        user_features, user_mask = self._dense_sparse_hidden(sparse_by_key, "user_tokens")
        context_features, context_mask = self._dense_sparse_hidden(sparse_by_key, "context_tokens")
        candidate_post_features, candidate_post_mask = self._dense_sparse_hidden(sparse_by_key, "candidate_post_tokens")
        candidate_author_features, candidate_author_mask = self._dense_sparse_hidden(sparse_by_key, "candidate_author_tokens")
        candidate_features, candidate_mask = self._dense_sparse_hidden(sparse_by_key, "candidate_tokens")

        batch_size = batch.labels.shape[0]
        dense_token = self.dense_projection(batch.dense_features).unsqueeze(1)
        dense_mask = torch.ones(batch_size, 1, dtype=torch.bool, device=batch.labels.device)

        user_features = self._add_non_sequence_group(user_features, 0)
        context_features = self._add_non_sequence_group(context_features, 1)
        candidate_post_features = self._add_non_sequence_group(candidate_post_features, 2)
        candidate_author_features = self._add_non_sequence_group(candidate_author_features, 3)
        candidate_features = self._add_non_sequence_group(candidate_features, 4)

        hidden_states = torch.cat(
            [
                dense_token,
                user_features,
                context_features,
                candidate_post_features,
                candidate_author_features,
                candidate_features,
            ],
            dim=1,
        )
        valid_mask = torch.cat(
            [
                dense_mask,
                user_mask,
                context_mask,
                candidate_post_mask,
                candidate_author_mask,
                candidate_mask,
            ],
            dim=1,
        )
        return hidden_states * valid_mask.unsqueeze(-1).float(), valid_mask

    def _build_sequence_inputs(self, batch: BatchTensors) -> tuple[torch.Tensor, torch.Tensor]:
        sequence_by_key = self.sequence_embedding.forward_dict(self._require_sequence_features(batch))
        missing_keys = [name for name in SEQUENCE_FEATURE_KEYS if name not in sequence_by_key]
        if missing_keys:
            missing = ", ".join(missing_keys)
            raise RuntimeError(f"Batch sequence_features is missing required keys: {missing}")

        history_hidden, history_mask = self._dense_sequence_hidden(sequence_by_key, "history_tokens")
        post_hidden, _ = self._dense_sequence_hidden(sequence_by_key, "history_post_tokens")
        author_hidden, _ = self._dense_sequence_hidden(sequence_by_key, "history_author_tokens")
        action_hidden, _ = self._dense_sequence_hidden(sequence_by_key, "history_action_tokens")
        time_gap_hidden, _ = self._dense_sequence_hidden(sequence_by_key, "history_time_gap")
        group_hidden, _ = self._dense_sequence_hidden(sequence_by_key, "history_group_ids")

        return self.sequence_preprocessor(
            history_hidden=history_hidden,
            post_hidden=post_hidden,
            author_hidden=author_hidden,
            action_hidden=action_hidden,
            time_gap_hidden=time_gap_hidden,
            group_hidden=group_hidden,
            valid_mask=history_mask,
        )

    def forward(self, batch: BatchTensors) -> torch.Tensor:
        non_sequence_hidden, non_sequence_mask = self._build_non_sequence_inputs(batch)
        sequence_hidden, sequence_mask = self._build_sequence_inputs(batch)

        initial_non_sequence_summary = self.non_sequence_summary(non_sequence_hidden)
        initial_cls_mask = torch.ones(
            batch.labels.shape[0],
            self.cls_token_count,
            dtype=torch.bool,
            device=batch.labels.device,
        )
        sequence_hidden = torch.cat([initial_non_sequence_summary, sequence_hidden], dim=1)
        sequence_mask = torch.cat([initial_cls_mask, sequence_mask], dim=1)

        for layer in self.layers:
            non_sequence_summary = self.non_sequence_summary(non_sequence_hidden)
            sequence_summary, sequence_summary_mask = self.sequence_summary(sequence_hidden, sequence_mask)
            non_sequence_hidden, sequence_hidden = layer(
                non_sequence_hidden=non_sequence_hidden,
                non_sequence_mask=non_sequence_mask,
                sequence_hidden=sequence_hidden,
                sequence_mask=sequence_mask,
                non_sequence_summary=non_sequence_summary,
                sequence_summary=sequence_summary,
                sequence_summary_mask=sequence_summary_mask,
            )

        final_non_sequence_summary = self.non_sequence_summary(non_sequence_hidden)
        final_sequence_summary, _ = self.sequence_summary(sequence_hidden, sequence_mask)
        logits = self.classifier(
            torch.cat(
                [
                    final_non_sequence_summary.reshape(batch.labels.shape[0], -1),
                    final_sequence_summary.reshape(batch.labels.shape[0], -1),
                ],
                dim=-1,
            )
        )
        return logits.squeeze(-1)


def build_model_component(data_config: DataConfig, model_config: ModelConfig, dense_dim: int) -> InterFormerModel:
    return InterFormerModel(data_config=data_config, model_config=model_config, dense_dim=dense_dim)