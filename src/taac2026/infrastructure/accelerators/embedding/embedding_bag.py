"""Embedding fused operator boundary."""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn.functional as F


EmbeddingBagMeanKernel = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]

_embedding_bag_mean_kernel: EmbeddingBagMeanKernel | None = None


def register_embedding_bag_mean_kernel(kernel: EmbeddingBagMeanKernel) -> None:
	global _embedding_bag_mean_kernel
	_embedding_bag_mean_kernel = kernel


def embedding_bag_mean(embedding_weight: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
	if _embedding_bag_mean_kernel is not None:
		return _embedding_bag_mean_kernel(embedding_weight, values)
	embedded = F.embedding(values, embedding_weight, padding_idx=0)
	valid = values.ne(0).to(embedded.dtype).unsqueeze(-1)
	return (embedded * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)


__all__ = ["EmbeddingBagMeanKernel", "embedding_bag_mean", "register_embedding_bag_mean_kernel"]