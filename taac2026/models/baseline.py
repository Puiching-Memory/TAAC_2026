from __future__ import annotations

import torch
from torch import nn

from ..config import ModelConfig
from .common import (
    DINActivationUnit,
    GrokAttentionBlock,
    build_history_embeddings,
    build_position_encoding,
    make_recsys_attn_mask,
    masked_mean,
)


class GrokUnifiedBaseline(nn.Module):
    def __init__(self, config: ModelConfig, dense_dim: int, max_seq_len: int) -> None:
        super().__init__()
        self.hidden_dim = config.hidden_dim
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
        self.output = self._build_output_head(config.hidden_dim * 4, config.dropout)

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

    def _build_context_tokens(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        context_embeddings = self.embedding(batch["context_tokens"])
        return self.context_projection(context_embeddings) + self.source_embedding.weight[1].view(1, 1, -1)

    def _build_history_tokens(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        history_tokens = build_history_embeddings(
            token_embedding=self.embedding,
            sequence_group_embedding=self.sequence_group_embedding,
            time_projection=self.time_projection,
            history_projection=self.history_projection,
            batch=batch,
        )
        return history_tokens + self.source_embedding.weight[2].view(1, 1, -1)

    def _build_candidate_token(
        self,
        batch: dict[str, torch.Tensor],
        dense_summary: torch.Tensor,
    ) -> torch.Tensor:
        candidate_embeddings = self.embedding(batch["candidate_tokens"])
        candidate_summary = masked_mean(candidate_embeddings, batch["candidate_mask"])
        candidate_token = self.candidate_projection(torch.cat([candidate_summary, dense_summary], dim=-1))
        return candidate_token + self.source_embedding.weight[3].view(1, -1)

    def _encode_backbone(
        self,
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, int, int, torch.Tensor]:
        batch_size = batch["labels"].size(0)
        dense_summary = self.dense_projection(batch["dense_features"])
        dense_token = dense_summary.unsqueeze(1) + self.source_embedding.weight[0].view(1, 1, -1)
        context_tokens = self._build_context_tokens(batch)
        history_tokens = self._build_history_tokens(batch)
        candidate_seed = self._build_candidate_token(batch, dense_summary)
        candidate_token = candidate_seed.unsqueeze(1)

        prefix_tokens = torch.cat([dense_token, context_tokens, history_tokens], dim=1)
        static_prefix_len = 1 + context_tokens.size(1)
        candidate_start_offset = prefix_tokens.size(1)
        tokens = torch.cat([prefix_tokens, candidate_token], dim=1)
        tokens = tokens + build_position_encoding(
            length=tokens.size(1),
            hidden_dim=tokens.size(-1),
            device=tokens.device,
            dtype=tokens.dtype,
        )
        tokens = self.embedding_dropout(tokens)

        prefix_mask = torch.cat(
            [
                torch.ones((batch_size, 1), dtype=torch.bool, device=tokens.device),
                batch["context_mask"],
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

        return tokens, prefix_mask, static_prefix_len, candidate_start_offset, candidate_seed

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        tokens, prefix_mask, _, candidate_start_offset, candidate_seed = self._encode_backbone(batch)

        candidate_output = tokens[:, candidate_start_offset, :]
        prefix_summary = masked_mean(tokens[:, :candidate_start_offset, :], prefix_mask)
        interaction = candidate_output * prefix_summary
        difference = torch.abs(candidate_output - candidate_seed)

        fused = torch.cat(
            [
                candidate_output,
                prefix_summary,
                interaction,
                difference,
            ],
            dim=-1,
        )
        return self.output(fused).squeeze(-1)


class GrokDINReadoutBaseline(GrokUnifiedBaseline):
    def __init__(self, config: ModelConfig, dense_dim: int, max_seq_len: int) -> None:
        super().__init__(config=config, dense_dim=dense_dim, max_seq_len=max_seq_len)
        self.interest_attention = DINActivationUnit(config.hidden_dim)
        self.output = self._build_output_head(config.hidden_dim * 7, config.dropout)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        tokens, prefix_mask, static_prefix_len, candidate_start_offset, candidate_seed = self._encode_backbone(batch)

        candidate_output = tokens[:, candidate_start_offset, :]
        prefix_summary = masked_mean(tokens[:, :candidate_start_offset, :], prefix_mask)
        history_outputs = tokens[:, static_prefix_len:candidate_start_offset, :]
        history_summary = masked_mean(history_outputs, batch["history_mask"])
        interest_summary = self.interest_attention(candidate_output, history_outputs, batch["history_mask"])
        interest_interaction = candidate_output * interest_summary
        prefix_interaction = candidate_output * prefix_summary
        candidate_gap = torch.abs(candidate_output - candidate_seed)

        fused = torch.cat(
            [
                candidate_output,
                prefix_summary,
                history_summary,
                interest_summary,
                interest_interaction,
                prefix_interaction,
                candidate_gap,
            ],
            dim=-1,
        )
        return self.output(fused).squeeze(-1)


__all__ = [
    "GrokDINReadoutBaseline",
    "GrokUnifiedBaseline",
    "GrokAttentionBlock",
    "make_recsys_attn_mask",
]