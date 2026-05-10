"""Normalization model primitives."""

from __future__ import annotations

from contextvars import ContextVar

import torch
import torch.nn as nn

from taac2026.infrastructure.accelerators.normalization.rms_norm import rms_norm


RMS_NORM_BACKEND = "torch"
RMS_NORM_BLOCK_ROWS = 1
_RMS_NORM_BACKEND: ContextVar[str] = ContextVar("taac2026_rms_norm_backend", default="torch")
_RMS_NORM_BLOCK_ROWS: ContextVar[int] = ContextVar("taac2026_rms_norm_block_rows", default=1)


def configure_rms_norm_runtime(*, backend: str, block_rows: int) -> None:
    if backend not in {"torch", "tilelang", "triton"}:
        raise ValueError(f"unsupported rms_norm backend: {backend}")
    resolved_block_rows = int(block_rows)
    if resolved_block_rows < 1:
        raise ValueError("rms_norm block_rows must be positive")
    _RMS_NORM_BACKEND.set(backend)
    _RMS_NORM_BLOCK_ROWS.set(resolved_block_rows)


def rms_norm_runtime_state() -> tuple[str, int]:
    return _RMS_NORM_BACKEND.get(), _RMS_NORM_BLOCK_ROWS.get()


class RMSNorm(nn.Module):
	def __init__(self, dim: int, eps: float = 1e-6, *, backend: str | None = None, block_rows: int | None = None) -> None:
		super().__init__()
		runtime_backend, runtime_block_rows = rms_norm_runtime_state()
		self.backend = runtime_backend if backend is None else backend
		self.block_rows = runtime_block_rows if block_rows is None else int(block_rows)
		if self.backend not in {"torch", "tilelang", "triton"}:
			raise ValueError(f"unsupported rms_norm backend: {self.backend}")
		if self.block_rows < 1:
			raise ValueError("rms_norm block_rows must be positive")
		self.weight = nn.Parameter(torch.ones(dim))
		self.eps = eps

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return rms_norm(
			x,
			self.weight,
			self.eps,
			backend=self.backend,
			block_rows=self.block_rows,
		)


__all__ = [
    "RMS_NORM_BACKEND",
    "RMS_NORM_BLOCK_ROWS",
    "RMSNorm",
    "configure_rms_norm_runtime",
    "rms_norm_runtime_state",
]
