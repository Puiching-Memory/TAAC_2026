from __future__ import annotations

import torch
from torch import nn

from taac2026.domain.config import DataConfig, ModelConfig
from taac2026.domain.types import BatchTensors

from .utils import masked_mean


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


class TargetAwareHistoryPool(nn.Module):
    """A readable DIN-style pooling block for behavior history."""

    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, query: torch.Tensor, keys: torch.Tensor, key_mask: torch.Tensor) -> torch.Tensor:
        expanded_query = query.unsqueeze(1).expand_as(keys)
        attention_inputs = torch.cat(
            [
                expanded_query,
                keys,
                expanded_query - keys,
                expanded_query * keys,
            ],
            dim=-1,
        )
        attention_scores = self.scorer(attention_inputs).squeeze(-1)
        attention_scores = attention_scores.masked_fill(~key_mask, -1.0e4)
        attention_weights = torch.softmax(attention_scores, dim=-1)
        attention_weights = attention_weights * key_mask.float()
        attention_weights = attention_weights / attention_weights.sum(dim=-1, keepdim=True).clamp_min(1.0)
        return torch.bmm(attention_weights.unsqueeze(1), keys).squeeze(1)


class ReferenceBaselineModel(nn.Module):
    """Starter model intended to be easy to read, copy, and extend.

    The design deliberately favors explicit submodules over clever abstractions:
    users can replace the history block, widen the fusion head, or add new
    feature branches without first untangling a large unified backbone.
    """

    def __init__(self, data_config: DataConfig, model_config: ModelConfig, dense_dim: int) -> None:
        super().__init__()
        del data_config
        self.hidden_dim = model_config.hidden_dim
        self.recent_seq_len = max(0, model_config.recent_seq_len)

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
        self.history_pool = TargetAwareHistoryPool(model_config.hidden_dim, model_config.dropout)

        head_hidden_dim = model_config.head_hidden_dim or model_config.hidden_dim * 2
        self.output = nn.Sequential(
            nn.LayerNorm(model_config.hidden_dim * 7),
            nn.Linear(model_config.hidden_dim * 7, head_hidden_dim),
            nn.GELU(),
            nn.Dropout(model_config.dropout),
            nn.Linear(head_hidden_dim, 1),
        )

    def _require(self, tensor: torch.Tensor | None, name: str) -> torch.Tensor:
        if tensor is None:
            raise RuntimeError(f"Batch is missing required tensor: {name}")
        return tensor

    def _embed_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.token_projection(self.token_embedding(tokens))

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
        user_tokens = self._require(batch.user_tokens, "user_tokens")
        user_mask = self._require(batch.user_mask, "user_mask")
        candidate_post_tokens = self._require(batch.candidate_post_tokens, "candidate_post_tokens")
        candidate_post_mask = self._require(batch.candidate_post_mask, "candidate_post_mask")
        candidate_author_tokens = self._require(batch.candidate_author_tokens, "candidate_author_tokens")
        candidate_author_mask = self._require(batch.candidate_author_mask, "candidate_author_mask")
        history_post_tokens = self._require(batch.history_post_tokens, "history_post_tokens")
        history_author_tokens = self._require(batch.history_author_tokens, "history_author_tokens")
        history_action_tokens = self._require(batch.history_action_tokens, "history_action_tokens")

        history_mask = self._slice_recent(batch.history_mask)
        history_tokens = self._slice_recent(batch.history_tokens)
        history_post_tokens = self._slice_recent(history_post_tokens)
        history_author_tokens = self._slice_recent(history_author_tokens)
        history_action_tokens = self._slice_recent(history_action_tokens)

        user_summary = masked_mean(self._embed_tokens(user_tokens), user_mask)
        context_summary = masked_mean(self._embed_tokens(batch.context_tokens), batch.context_mask)
        user_representation = self.user_encoder(torch.cat([user_summary, context_summary], dim=-1))

        dense_representation = self.dense_encoder(batch.dense_features)

        candidate_post_summary = masked_mean(self._embed_tokens(candidate_post_tokens), candidate_post_mask)
        candidate_author_summary = masked_mean(self._embed_tokens(candidate_author_tokens), candidate_author_mask)
        candidate_legacy_summary = masked_mean(self._embed_tokens(batch.candidate_tokens), batch.candidate_mask)
        candidate_representation = self.candidate_encoder(
            torch.cat(
                [candidate_post_summary, candidate_author_summary, candidate_legacy_summary],
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
