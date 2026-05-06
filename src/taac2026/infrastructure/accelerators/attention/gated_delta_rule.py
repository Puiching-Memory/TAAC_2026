"""Chunked gated-delta-rule attention operator boundary."""

from __future__ import annotations

from taac2026.infrastructure.accelerators.attention.kernels.gated_delta_rule import (
    chunk_gated_delta_rule,
    chunk_gated_delta_rule_available,
    chunk_gated_delta_rule_bwd,
    chunk_gated_delta_rule_fwd,
)


__all__ = [
    "chunk_gated_delta_rule",
    "chunk_gated_delta_rule_available",
    "chunk_gated_delta_rule_bwd",
    "chunk_gated_delta_rule_fwd",
]
