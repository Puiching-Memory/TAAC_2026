"""TileLang kernels for the chunked gated-delta-rule attention operator."""

from __future__ import annotations

from taac2026.infrastructure.accelerators.attention.kernels.gated_delta_rule.chunk import (
    chunk_gated_delta_rule,
    chunk_gated_delta_rule_bwd,
    chunk_gated_delta_rule_fwd,
)
from taac2026.infrastructure.accelerators.attention.gated_delta_rule_capabilities import (
    chunk_gated_delta_rule_available,
)


__all__ = [
    "chunk_gated_delta_rule",
    "chunk_gated_delta_rule_available",
    "chunk_gated_delta_rule_bwd",
    "chunk_gated_delta_rule_fwd",
]
