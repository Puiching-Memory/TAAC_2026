from __future__ import annotations

import torch
from torch import nn

from ..config import ModelConfig
from .common import (
    DINActivationUnit,
    GrokAttentionBlock,
    build_history_embeddings,
    build_pooled_memory,
    build_position_encoding,
    make_recsys_attn_mask,
    masked_mean,
)


class UnifiedModelBase(nn.Module):
    def __init__(self, config: ModelConfig, dense_dim: int) -> None:
        super().__init__()
        self.hidden_dim = config.hidden_dim
        self.recent_seq_len = max(config.recent_seq_len, 1)
        self.memory_slots = max(config.memory_slots, 1)
        self.embedding = nn.Embedding(config.vocab_size, config.embedding_dim, padding_idx=0)
        self.source_embedding = nn.Embedding(4, config.hidden_dim)
        self.sequence_group_embedding = nn.Embedding(4, config.hidden_dim, padding_idx=0)
        self.context_projection = nn.Sequential(
            nn.Linear(config.embedding_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.SiLU(),
        )
        self.time_projection = nn.Sequential(
            nn.Linear(1, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
        )
        self.history_projection = nn.Sequential(
            nn.Linear(config.embedding_dim * 2, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.SiLU(),
        )
        self.dense_projection = nn.Sequential(
            nn.Linear(dense_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.SiLU(),
            nn.Dropout(config.dropout),
        )
        self.candidate_projection = nn.Sequential(
            nn.Linear(config.embedding_dim + config.hidden_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.SiLU(),
        )
        self.embedding_dropout = nn.Dropout(config.dropout)

    def build_views(
        self,
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        dense_summary = self.dense_projection(batch["dense_features"])
        dense_token = dense_summary.unsqueeze(1) + self.source_embedding.weight[0].view(1, 1, -1)
        context_tokens = self.context_projection(self.embedding(batch["context_tokens"]))
        context_tokens = context_tokens + self.source_embedding.weight[1].view(1, 1, -1)
        history_tokens = build_history_embeddings(
            token_embedding=self.embedding,
            sequence_group_embedding=self.sequence_group_embedding,
            time_projection=self.time_projection,
            history_projection=self.history_projection,
            batch=batch,
        )
        history_tokens = history_tokens + self.source_embedding.weight[2].view(1, 1, -1)
        candidate_embeddings = self.embedding(batch["candidate_tokens"])
        candidate_summary = masked_mean(candidate_embeddings, batch["candidate_mask"])
        candidate_token = self.candidate_projection(torch.cat([candidate_summary, dense_summary], dim=-1))
        candidate_token = candidate_token + self.source_embedding.weight[3].view(1, -1)
        return dense_summary, dense_token, context_tokens, history_tokens, candidate_token


class DeepContextNet(UnifiedModelBase):
    def __init__(self, config: ModelConfig, dense_dim: int, max_seq_len: int) -> None:
        super().__init__(config=config, dense_dim=dense_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_dim))
        self.blocks = nn.ModuleList(
            [
                GrokAttentionBlock(
                    hidden_dim=config.hidden_dim,
                    num_heads=config.num_heads,
                    dropout=config.dropout,
                    ffn_multiplier=config.ffn_multiplier,
                )
                for _ in range(config.num_layers)
            ]
        )
        self.output = nn.Sequential(
            nn.LayerNorm(config.hidden_dim * 5),
            nn.Linear(config.hidden_dim * 5, config.hidden_dim * 2),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, 1),
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        batch_size = batch["labels"].size(0)
        dense_summary, dense_token, context_tokens, history_tokens, candidate_token = self.build_views(batch)
        cls_token = self.cls_token.expand(batch_size, -1, -1)
        candidate_token = candidate_token.unsqueeze(1)
        tokens = torch.cat([cls_token, dense_token, context_tokens, candidate_token, history_tokens], dim=1)
        tokens = tokens + build_position_encoding(tokens.size(1), tokens.size(-1), tokens.device, tokens.dtype)
        tokens = self.embedding_dropout(tokens)

        candidate_index = 2 + context_tokens.size(1)
        mask = torch.cat(
            [
                torch.ones((batch_size, 2), dtype=torch.bool, device=tokens.device),
                batch["context_mask"],
                torch.ones((batch_size, 1), dtype=torch.bool, device=tokens.device),
                batch["history_mask"],
            ],
            dim=1,
        )
        key_padding_mask = ~mask
        for block in self.blocks:
            tokens = block(tokens, key_padding_mask=key_padding_mask, attn_mask=None)

        cls_output = tokens[:, 0, :]
        candidate_output = tokens[:, candidate_index, :]
        history_output = tokens[:, candidate_index + 1 :, :]
        history_summary = masked_mean(history_output, batch["history_mask"])
        interaction = cls_output * candidate_output
        difference = torch.abs(candidate_output - history_summary)
        fused = torch.cat([cls_output, candidate_output, history_summary, interaction, difference], dim=-1)
        return self.output(fused).squeeze(-1)


class UniRecBackboneBase(UnifiedModelBase):
    def __init__(self, config: ModelConfig, dense_dim: int, max_seq_len: int) -> None:
        super().__init__(config=config, dense_dim=dense_dim)
        self.feature_cross_layers = nn.ModuleList(
            [
                GrokAttentionBlock(
                    hidden_dim=config.hidden_dim,
                    num_heads=config.num_heads,
                    dropout=config.dropout,
                    ffn_multiplier=config.ffn_multiplier,
                )
                for _ in range(max(config.feature_cross_layers, 1))
            ]
        )
        self.interest_attention = DINActivationUnit(config.hidden_dim)
        self.blocks = nn.ModuleList(
            [
                GrokAttentionBlock(
                    hidden_dim=config.hidden_dim,
                    num_heads=config.num_heads,
                    dropout=config.dropout,
                    ffn_multiplier=config.ffn_multiplier,
                )
                for _ in range(config.num_layers)
            ]
        )

    def _build_output_head(self, input_dim: int, dropout: float) -> nn.Sequential:
        return nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, self.hidden_dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, 1),
        )

    def _encode_backbone(
        self,
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, int, int]:
        batch_size = batch["labels"].size(0)
        dense_summary, dense_token, context_tokens, history_tokens, candidate_seed = self.build_views(batch)
        context_mask = batch["context_mask"]
        for layer in self.feature_cross_layers:
            context_tokens = layer(context_tokens, key_padding_mask=~context_mask, attn_mask=None)

        interest_token = self.interest_attention(candidate_seed, history_tokens, batch["history_mask"]).unsqueeze(1)
        candidate_token = candidate_seed.unsqueeze(1)
        prefix_tokens = torch.cat([dense_token, context_tokens, interest_token, history_tokens], dim=1)
        static_prefix_len = 1 + context_tokens.size(1) + 1
        candidate_start_offset = prefix_tokens.size(1)
        tokens = torch.cat([prefix_tokens, candidate_token], dim=1)
        tokens = tokens + build_position_encoding(tokens.size(1), tokens.size(-1), tokens.device, tokens.dtype)
        tokens = self.embedding_dropout(tokens)

        prefix_mask = torch.cat(
            [
                torch.ones((batch_size, 1), dtype=torch.bool, device=tokens.device),
                context_mask,
                torch.ones((batch_size, 1), dtype=torch.bool, device=tokens.device),
                batch["history_mask"],
            ],
            dim=1,
        )
        sequence_mask = torch.cat(
            [
                prefix_mask,
                torch.ones((batch_size, 1), dtype=torch.bool, device=tokens.device),
            ],
            dim=1,
        )
        attn_mask = make_recsys_attn_mask(
            seq_len=tokens.size(1),
            static_prefix_len=static_prefix_len,
            candidate_start_offset=candidate_start_offset,
            device=tokens.device,
        )
        key_padding_mask = ~sequence_mask
        for block in self.blocks:
            tokens = block(tokens, key_padding_mask=key_padding_mask, attn_mask=attn_mask)

        return tokens, dense_summary, static_prefix_len, candidate_start_offset


class UniRecModel(UniRecBackboneBase):
    def __init__(self, config: ModelConfig, dense_dim: int, max_seq_len: int) -> None:
        super().__init__(config=config, dense_dim=dense_dim, max_seq_len=max_seq_len)
        self.output = self._build_output_head(config.hidden_dim * 6, config.dropout)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        tokens, dense_summary, static_prefix_len, candidate_start_offset = self._encode_backbone(batch)

        candidate_output = tokens[:, candidate_start_offset, :]
        interest_output = tokens[:, static_prefix_len - 1, :]
        history_summary = masked_mean(tokens[:, static_prefix_len:candidate_start_offset, :], batch["history_mask"])
        route = candidate_output * interest_output
        difference = torch.abs(candidate_output - history_summary)
        fused = torch.cat([candidate_output, interest_output, history_summary, dense_summary, route, difference], dim=-1)
        return self.output(fused).squeeze(-1)


class UniRecDINReadoutModel(UniRecBackboneBase):
    def __init__(self, config: ModelConfig, dense_dim: int, max_seq_len: int) -> None:
        super().__init__(config=config, dense_dim=dense_dim, max_seq_len=max_seq_len)
        self.readout_attention = DINActivationUnit(config.hidden_dim)
        self.output = self._build_output_head(config.hidden_dim * 8, config.dropout)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        tokens, dense_summary, static_prefix_len, candidate_start_offset = self._encode_backbone(batch)

        candidate_output = tokens[:, candidate_start_offset, :]
        interest_output = tokens[:, static_prefix_len - 1, :]
        history_outputs = tokens[:, static_prefix_len:candidate_start_offset, :]
        history_summary = masked_mean(history_outputs, batch["history_mask"])
        readout_summary = self.readout_attention(candidate_output, history_outputs, batch["history_mask"])
        route = candidate_output * interest_output
        readout_route = candidate_output * readout_summary
        difference = torch.abs(candidate_output - readout_summary)
        fused = torch.cat(
            [
                candidate_output,
                interest_output,
                history_summary,
                readout_summary,
                dense_summary,
                route,
                readout_route,
                difference,
            ],
            dim=-1,
        )
        return self.output(fused).squeeze(-1)


class UniScaleFormer(UnifiedModelBase):
    def __init__(self, config: ModelConfig, dense_dim: int, max_seq_len: int) -> None:
        super().__init__(config=config, dense_dim=dense_dim)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=config.hidden_dim,
            num_heads=config.num_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.blocks = nn.ModuleList(
            [
                GrokAttentionBlock(
                    hidden_dim=config.hidden_dim,
                    num_heads=config.num_heads,
                    dropout=config.dropout,
                    ffn_multiplier=config.ffn_multiplier,
                )
                for _ in range(max(config.num_layers - 1, 1))
            ]
        )
        self.output = nn.Sequential(
            nn.LayerNorm(config.hidden_dim * 6),
            nn.Linear(config.hidden_dim * 6, config.hidden_dim * 2),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, 1),
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        batch_size = batch["labels"].size(0)
        dense_summary, dense_token, context_tokens, history_tokens, candidate_seed = self.build_views(batch)
        recent_len = min(self.recent_seq_len, history_tokens.size(1))
        local_history = history_tokens[:, :recent_len]
        local_mask = batch["history_mask"][:, :recent_len]
        memory_tokens, memory_mask = build_pooled_memory(
            history_embeddings=history_tokens,
            history_mask=batch["history_mask"],
            recent_seq_len=recent_len,
            memory_slots=self.memory_slots,
        )
        memory_context = torch.cat([local_history, memory_tokens], dim=1)
        memory_context_mask = torch.cat([local_mask, memory_mask], dim=1)

        candidate_query = candidate_seed.unsqueeze(1)
        if memory_context.size(1) > 0:
            attended, _ = self.cross_attention(
                query=candidate_query,
                key=memory_context,
                value=memory_context,
                key_padding_mask=~memory_context_mask,
                need_weights=False,
            )
        else:
            attended = torch.zeros_like(candidate_query)
        candidate_token = candidate_query + attended

        prefix_tokens = torch.cat([dense_token, context_tokens, memory_context], dim=1)
        candidate_start_offset = prefix_tokens.size(1)
        tokens = torch.cat([prefix_tokens, candidate_token], dim=1)
        tokens = tokens + build_position_encoding(tokens.size(1), tokens.size(-1), tokens.device, tokens.dtype)
        tokens = self.embedding_dropout(tokens)

        sequence_mask = torch.cat(
            [
                torch.ones((batch_size, 1), dtype=torch.bool, device=tokens.device),
                batch["context_mask"],
                memory_context_mask,
                torch.ones((batch_size, 1), dtype=torch.bool, device=tokens.device),
            ],
            dim=1,
        )
        key_padding_mask = ~sequence_mask
        for block in self.blocks:
            tokens = block(tokens, key_padding_mask=key_padding_mask, attn_mask=None)

        candidate_output = tokens[:, candidate_start_offset, :]
        context_summary = masked_mean(tokens[:, 1 : 1 + context_tokens.size(1), :], batch["context_mask"])
        memory_summary = masked_mean(tokens[:, 1 + context_tokens.size(1) : candidate_start_offset, :], memory_context_mask)
        interaction = candidate_output * memory_summary
        difference = torch.abs(candidate_output - candidate_seed)
        fused = torch.cat([candidate_output, candidate_seed, context_summary, memory_summary, interaction, difference], dim=-1)
        return self.output(fused).squeeze(-1)


__all__ = ["DeepContextNet", "UniRecDINReadoutModel", "UniRecModel", "UniScaleFormer"]