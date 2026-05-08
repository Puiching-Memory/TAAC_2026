"""Embedding accelerator operator boundaries."""

from __future__ import annotations

from taac2026.infrastructure.accelerators.embedding.embedding_bag import (
	EmbeddingBagMeanBackend,
	EmbeddingBagMeanKernel,
	EmbeddingBagMeanKernelKey,
	clear_embedding_bag_mean_kernel_cache,
	compile_embedding_bag_mean_kernel,
	embedding_bag_mean,
	register_embedding_bag_mean_kernel,
	resolved_embedding_bag_mean_backend,
)

__all__ = [
	"EmbeddingBagMeanBackend",
	"EmbeddingBagMeanKernel",
	"EmbeddingBagMeanKernelKey",
	"clear_embedding_bag_mean_kernel_cache",
	"compile_embedding_bag_mean_kernel",
	"embedding_bag_mean",
	"register_embedding_bag_mean_kernel",
	"resolved_embedding_bag_mean_backend",
]
