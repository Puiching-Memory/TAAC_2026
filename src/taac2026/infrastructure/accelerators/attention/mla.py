"""Multi-latent attention operator boundary."""

from __future__ import annotations

from typing import Literal

import torch

from taac2026.infrastructure.accelerators.attention.flash_attention import flash_attention


def multi_latent_attention(
	queries: torch.Tensor,
	latent_keys: torch.Tensor,
	latent_values: torch.Tensor,
	*,
	backend: Literal["torch", "tilelang"] = "torch",
	attn_mask: torch.Tensor | None = None,
	dropout_p: float = 0.0,
	training: bool = False,
	is_causal: bool = False,
) -> torch.Tensor:
	"""Apply attention from query tokens to a latent key/value memory.

	Shapes follow the same convention as the lower-level flash-attention
	boundary: ``(batch, heads, tokens, head_dim)``. The function intentionally
	keeps MLA as an operator-level primitive; projection into latent key/value
	spaces remains model code.
	"""

	if queries.ndim != 4 or latent_keys.ndim != 4 or latent_values.ndim != 4:
		raise ValueError("multi_latent_attention expects 4D query/key/value tensors")
	if latent_keys.shape[2] != latent_values.shape[2]:
		raise ValueError("latent key and value memories must have matching token counts")
	return flash_attention(
		queries,
		latent_keys,
		latent_values,
		backend=backend,
		attn_mask=attn_mask,
		dropout_p=dropout_p,
		training=training,
		is_causal=is_causal,
	)


__all__ = ["multi_latent_attention"]