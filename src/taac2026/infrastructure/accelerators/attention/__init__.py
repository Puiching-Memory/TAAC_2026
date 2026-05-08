"""Attention accelerator operator boundaries."""

from __future__ import annotations

from taac2026.infrastructure.accelerators.attention.flash_attention import (
	FlashAttentionBackend,
	FlashAttentionKernel,
	FlashAttentionKernelKey,
	FlashAttentionMaskPlan,
	clear_flash_attention_kernel_cache,
	compile_flash_attention_kernel,
	flash_attention,
	register_flash_attention_kernel,
	resolved_flash_attention_backend,
)
from taac2026.infrastructure.accelerators.attention.gated_delta_rule import (
	chunk_gated_delta_rule,
	chunk_gated_delta_rule_available,
	chunk_gated_delta_rule_bwd,
	chunk_gated_delta_rule_fwd,
)
from taac2026.infrastructure.accelerators.attention.mla import multi_latent_attention

__all__ = [
	"FlashAttentionBackend",
	"FlashAttentionKernel",
	"FlashAttentionKernelKey",
	"FlashAttentionMaskPlan",
	"chunk_gated_delta_rule",
	"chunk_gated_delta_rule_available",
	"chunk_gated_delta_rule_bwd",
	"chunk_gated_delta_rule_fwd",
	"clear_flash_attention_kernel_cache",
	"compile_flash_attention_kernel",
	"flash_attention",
	"multi_latent_attention",
	"register_flash_attention_kernel",
	"resolved_flash_attention_backend",
]
