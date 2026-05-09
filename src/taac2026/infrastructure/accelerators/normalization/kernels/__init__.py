"""Normalization accelerator kernels."""

from __future__ import annotations

from taac2026.infrastructure.accelerators.normalization.kernels.tilelang import (
    build_rms_norm_backward_kernel,
    build_rms_norm_forward_kernel,
)
from taac2026.infrastructure.accelerators.normalization.kernels.triton import (
    build_rms_norm_backward_kernel as build_triton_rms_norm_backward_kernel,
    build_rms_norm_forward_kernel as build_triton_rms_norm_forward_kernel,
)

__all__ = [
    "build_rms_norm_backward_kernel",
    "build_rms_norm_forward_kernel",
    "build_triton_rms_norm_backward_kernel",
    "build_triton_rms_norm_forward_kernel",
]
