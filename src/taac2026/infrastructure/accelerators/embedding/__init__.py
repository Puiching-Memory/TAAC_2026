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
from taac2026.infrastructure.accelerators.embedding.cuembed_runtime import (
	CuEmbedEmbeddingBagMeanKernel,
	compile_cuembed_embedding_bag_mean_kernel,
	cuembed_available,
)

__all__ = [
	"CuEmbedEmbeddingBagMeanKernel",
	"EmbeddingBagMeanBackend",
	"EmbeddingBagMeanKernel",
	"EmbeddingBagMeanKernelKey",
	"clear_embedding_bag_mean_kernel_cache",
	"compile_cuembed_embedding_bag_mean_kernel",
	"compile_embedding_bag_mean_kernel",
	"cuembed_available",
	"embedding_bag_mean",
	"register_embedding_bag_mean_kernel",
	"resolved_embedding_bag_mean_backend",
]
