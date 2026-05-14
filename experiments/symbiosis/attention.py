"""Metadata-driven attention masks for Symbiosis V2."""

from __future__ import annotations

import torch
import torch.nn as nn

from taac2026.api import safe_key_padding_mask

try:
    from .tokenization import ROLE_CANDIDATE, ROLE_CLS, ROLE_DENSE, ROLE_ITEM, ROLE_MISSING, ROLE_SEQUENCE, ROLE_STATS, ROLE_USER, UnifiedTokenBatch
except ImportError:  # pragma: no cover - direct file loading in contract tests
    from tokenization import ROLE_CANDIDATE, ROLE_CLS, ROLE_DENSE, ROLE_ITEM, ROLE_MISSING, ROLE_SEQUENCE, ROLE_STATS, ROLE_USER, UnifiedTokenBatch


class MetadataAttentionMask(nn.Module):
    def __init__(self, *, enabled: bool = True) -> None:
        super().__init__()
        self.enabled = bool(enabled)

    def forward(self, batch: UnifiedTokenBatch) -> torch.Tensor:
        token_count = int(batch.tokens.shape[1])
        key_valid = ~safe_key_padding_mask(batch.padding_mask)
        if not self.enabled:
            structural = torch.ones(token_count, token_count, dtype=torch.bool, device=batch.tokens.device)
        else:
            roles = batch.role_ids
            key_roles = roles.unsqueeze(0)
            query_roles = roles.unsqueeze(1)
            readout_query = (query_roles == ROLE_CLS) | (query_roles == ROLE_CANDIDATE)
            field_query = (query_roles == ROLE_USER) | (query_roles == ROLE_ITEM) | (query_roles == ROLE_DENSE) | (query_roles == ROLE_MISSING)
            sequence_query = query_roles == ROLE_SEQUENCE
            stats_query = query_roles == ROLE_STATS

            readout_keys = torch.ones_like(key_roles, dtype=torch.bool)
            field_keys = (
                (key_roles == ROLE_CLS)
                | (key_roles == ROLE_CANDIDATE)
                | (key_roles == ROLE_USER)
                | (key_roles == ROLE_ITEM)
                | (key_roles == ROLE_DENSE)
                | (key_roles == ROLE_MISSING)
                | (key_roles == ROLE_STATS)
            )
            sequence_keys = (
                (key_roles == ROLE_CLS)
                | (key_roles == ROLE_CANDIDATE)
                | (key_roles == ROLE_ITEM)
                | (key_roles == ROLE_SEQUENCE)
                | (key_roles == ROLE_MISSING)
                | (key_roles == ROLE_STATS)
            )
            stats_keys = readout_keys
            structural = torch.zeros(token_count, token_count, dtype=torch.bool, device=batch.tokens.device)
            structural = torch.where(readout_query, readout_keys.expand_as(structural), structural)
            structural = torch.where(field_query, field_keys.expand_as(structural), structural)
            structural = torch.where(sequence_query, sequence_keys.expand_as(structural), structural)
            structural = torch.where(stats_query, stats_keys.expand_as(structural), structural)
            structural = structural | torch.eye(token_count, dtype=torch.bool, device=batch.tokens.device)
        return structural.unsqueeze(0).unsqueeze(0) & key_valid.unsqueeze(1).unsqueeze(2)


__all__ = ["MetadataAttentionMask"]