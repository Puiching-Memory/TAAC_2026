"""Kernel source boundaries for embedding operators."""

from __future__ import annotations

from taac2026.infrastructure.accelerators.embedding.kernels.tilelang import (
    build_embedding_bag_mean_backward_kernel,
    build_embedding_bag_mean_forward_kernel,
)
from taac2026.infrastructure.accelerators.embedding.kernels.triton import (
    build_embedding_bag_mean_backward_kernel as build_triton_embedding_bag_mean_backward_kernel,
    build_embedding_bag_mean_forward_kernel as build_triton_embedding_bag_mean_forward_kernel,
)

__all__ = [
    "build_embedding_bag_mean_backward_kernel",
    "build_embedding_bag_mean_forward_kernel",
    "build_triton_embedding_bag_mean_backward_kernel",
    "build_triton_embedding_bag_mean_forward_kernel",
]
