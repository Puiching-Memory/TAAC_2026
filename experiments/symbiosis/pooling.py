"""Readout pooling for Symbiosis V2."""

from __future__ import annotations

import torch
import torch.nn as nn

from taac2026.api import RMSNorm, masked_mean

try:
    from .tokenization import UnifiedTokenBatch
except ImportError:  # pragma: no cover - direct file loading in contract tests
    from tokenization import UnifiedTokenBatch


class CandidateClsPooler(nn.Module):
    def __init__(self, d_model: int, *, use_candidate_readout: bool) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.use_candidate_readout = bool(use_candidate_readout)
        self.output_dim = self.d_model * 3
        self.norm = RMSNorm(self.output_dim)

    def forward(self, tokens: torch.Tensor, batch: UnifiedTokenBatch) -> torch.Tensor:
        cls_summary = tokens[:, batch.cls_index, :]
        if self.use_candidate_readout:
            candidate_summary = tokens[:, batch.candidate_index, :]
        else:
            candidate_summary = torch.zeros_like(cls_summary)
        global_summary = masked_mean(tokens, batch.padding_mask)
        return self.norm(torch.cat([cls_summary, candidate_summary, global_summary], dim=-1))


__all__ = ["CandidateClsPooler"]