"""TileLang attention kernels."""

from __future__ import annotations

from taac2026.infrastructure.accelerators.kernels import build_flash_attention_forward_kernel

__all__ = ["build_flash_attention_forward_kernel"]