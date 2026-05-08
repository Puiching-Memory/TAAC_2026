"""Kernel source boundaries for embedding operators."""

from __future__ import annotations

from taac2026.infrastructure.accelerators.embedding.kernels.tilelang import (
    build_embedding_bag_mean_backward_kernel,
    build_embedding_bag_mean_forward_kernel,
)

__all__ = [
    "build_embedding_bag_mean_backward_kernel",
    "build_embedding_bag_mean_forward_kernel",
]
