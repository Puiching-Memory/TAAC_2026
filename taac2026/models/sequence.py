from __future__ import annotations

import torch
from torch import nn

from ..config import ModelConfig
from .common import (
    DINActivationUnit,
    GrokAttentionBlock,
    build_causal_attention_mask,
    build_decomposed_history_embeddings,
    build_pooled_memory,
    masked_attention_pool,
    masked_mean,
)


class SequenceModelBase(nn.Module):
    def __init__(self, config: ModelConfig, dense_dim: int, max_seq_len: int) -> None:
        super().__init__()
        self.recent_seq_len = min(max(config.recent_seq_len, 1), max_seq_len)
        self.memory_slots = max(config.memory_slots, 1)
        self.token_embedding = nn.Embedding(config.vocab_size, config.embedding_dim, padding_idx=0)
        self.position_embedding = nn.Embedding(max_seq_len + 1, config.embedding_dim)
        self.sequence_group_embedding = nn.Embedding(4, config.embedding_dim, padding_idx=0)
        self.time_projection = nn.Sequential(
            nn.Linear(1, config.embedding_dim),
            nn.SiLU(),
            nn.Linear(config.embedding_dim, config.embedding_dim),
        )
        self.component_projection = nn.Sequential(
            nn.Linear(config.embedding_dim * 2, config.embedding_dim),
            nn.LayerNorm(config.embedding_dim),
            nn.SiLU(),
        )
        self.context_projection = nn.Sequential(
            nn.Linear(config.embedding_dim * 2, config.embedding_dim),
            nn.LayerNorm(config.embedding_dim),
            nn.SiLU(),
        )
        self.dense_projection = nn.Sequential(
            nn.Linear(dense_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.SiLU(),
            nn.Dropout(config.dropout),
        )
        self.blocks = nn.ModuleList(
            [
                GrokAttentionBlock(
                    hidden_dim=config.embedding_dim,
                    num_heads=config.num_heads,
                    dropout=config.dropout,
                    ffn_multiplier=config.ffn_multiplier,
                )
                for _ in range(config.num_layers)
            ]
        )
        self.dropout = nn.Dropout(config.dropout)

    def encode_inputs(
        self,
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        candidate_embeddings = self.token_embedding(batch["candidate_tokens"])
        context_embeddings = self.token_embedding(batch["context_tokens"])
        history_embeddings, component_summary = build_decomposed_history_embeddings(
            token_embedding=self.token_embedding,
            position_embedding=self.position_embedding,
            sequence_group_embedding=self.sequence_group_embedding,
            time_projection=self.time_projection,
            component_projection=self.component_projection,
            batch=batch,
        )
        history_embeddings = self.dropout(history_embeddings)

        candidate_summary = masked_mean(candidate_embeddings, batch["candidate_mask"])
        context_summary = masked_mean(context_embeddings, batch["context_mask"])
        component_history_summary = masked_mean(component_summary, batch["history_mask"])
        dense_summary = self.dense_projection(batch["dense_features"])
        return (
            candidate_summary,
            context_summary,
            history_embeddings,
            component_history_summary,
            dense_summary,
            self.context_projection(torch.cat([context_summary, component_history_summary], dim=-1)),
        )

    def encode_history(self, history_embeddings: torch.Tensor, history_mask: torch.Tensor) -> torch.Tensor:
        if history_embeddings.size(1) == 0:
            return history_embeddings
        attn_mask = build_causal_attention_mask(history_embeddings.size(1), history_embeddings.device)
        key_padding_mask = ~history_mask
        encoded = history_embeddings
        for block in self.blocks:
            encoded = block(encoded, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        return encoded


class TencentSASRecAdapter(SequenceModelBase):
    def __init__(self, config: ModelConfig, dense_dim: int, max_seq_len: int) -> None:
        super().__init__(config=config, dense_dim=dense_dim, max_seq_len=max_seq_len)
        self.output = nn.Sequential(
            nn.Linear(config.embedding_dim * 7 + config.hidden_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, 1),
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        (
            candidate_summary,
            context_summary,
            history_embeddings,
            component_history_summary,
            dense_summary,
            _,
        ) = self.encode_inputs(batch)
        encoded_history = self.encode_history(history_embeddings, batch["history_mask"])
        history_summary = masked_attention_pool(encoded_history, batch["history_mask"], candidate_summary)
        recent_history = encoded_history[:, : self.recent_seq_len]
        recent_mask = batch["history_mask"][:, : self.recent_seq_len]
        recent_summary = masked_mean(recent_history, recent_mask) if recent_history.size(1) > 0 else torch.zeros_like(candidate_summary)
        interaction = candidate_summary * history_summary
        difference = torch.abs(candidate_summary - history_summary)

        fused = torch.cat(
            [
                candidate_summary,
                context_summary,
                history_summary,
                recent_summary,
                component_history_summary,
                interaction,
                difference,
                dense_summary,
            ],
            dim=-1,
        )
        return self.output(fused).squeeze(-1)


class RetrievalStyleAdapter(SequenceModelBase):
    def __init__(self, config: ModelConfig, dense_dim: int, max_seq_len: int, variant: str) -> None:
        super().__init__(config=config, dense_dim=dense_dim, max_seq_len=max_seq_len)
        self.variant = variant
        self.global_attention = DINActivationUnit(config.embedding_dim)
        self.action_attention = DINActivationUnit(config.embedding_dim)
        self.content_attention = DINActivationUnit(config.embedding_dim)
        self.item_attention = DINActivationUnit(config.embedding_dim)
        self.route_gate = nn.Sequential(
            nn.Linear(config.embedding_dim * 2 + config.hidden_dim, config.embedding_dim),
            nn.LayerNorm(config.embedding_dim),
            nn.SiLU(),
            nn.Linear(config.embedding_dim, 3),
        )
        hidden_scale = 2 if variant == "omnigenrec_adapter" else 1
        self.output = nn.Sequential(
            nn.Linear(config.embedding_dim * 12 + config.hidden_dim, config.hidden_dim * hidden_scale),
            nn.LayerNorm(config.hidden_dim * hidden_scale),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim * hidden_scale, config.hidden_dim),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, 1),
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        (
            candidate_summary,
            context_summary,
            history_embeddings,
            component_history_summary,
            dense_summary,
            context_enhanced,
        ) = self.encode_inputs(batch)
        encoded_history = self.encode_history(history_embeddings, batch["history_mask"])

        global_summary = self.global_attention(candidate_summary, encoded_history, batch["history_mask"])
        local_history = encoded_history[:, : self.recent_seq_len]
        local_mask = batch["history_mask"][:, : self.recent_seq_len]
        local_summary = self.global_attention(candidate_summary, local_history, local_mask)
        memory_tokens, memory_mask = build_pooled_memory(
            history_embeddings=encoded_history,
            history_mask=batch["history_mask"],
            recent_seq_len=self.recent_seq_len,
            memory_slots=self.memory_slots,
        )
        memory_summary = masked_attention_pool(memory_tokens, memory_mask, candidate_summary)

        action_mask = batch["history_mask"] & (batch["history_group_ids"] == 1)
        content_mask = batch["history_mask"] & (batch["history_group_ids"] == 2)
        item_mask = batch["history_mask"] & (batch["history_group_ids"] == 3)
        action_summary = self.action_attention(candidate_summary, encoded_history, action_mask)
        content_summary = self.content_attention(candidate_summary, encoded_history, content_mask)
        item_summary = self.item_attention(candidate_summary, encoded_history, item_mask)

        route_logits = self.route_gate(torch.cat([candidate_summary, context_enhanced, dense_summary], dim=-1))
        route_weights = torch.softmax(route_logits, dim=-1)
        route_stack = torch.stack([action_summary, content_summary, item_summary], dim=1)
        grouped_summary = (route_stack * route_weights.unsqueeze(-1)).sum(dim=1)
        route_spread = route_stack.std(dim=1, correction=0)

        if self.variant == "omnigenrec_adapter":
            grouped_summary = grouped_summary + 0.5 * local_summary

        interaction_global = candidate_summary * global_summary
        interaction_grouped = candidate_summary * grouped_summary
        local_memory_gap = torch.abs(local_summary - memory_summary)

        fused = torch.cat(
            [
                candidate_summary,
                context_summary,
                context_enhanced,
                global_summary,
                local_summary,
                memory_summary,
                grouped_summary,
                component_history_summary,
                interaction_global,
                interaction_grouped,
                local_memory_gap,
                route_spread,
                dense_summary,
            ],
            dim=-1,
        )
        return self.output(fused).squeeze(-1)


__all__ = ["RetrievalStyleAdapter", "TencentSASRecAdapter"]