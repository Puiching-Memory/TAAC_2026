"""Normalization model primitives."""

from __future__ import annotations

import torch
import torch.nn as nn

from taac2026.infrastructure.accelerators.normalization.rms_norm import rms_norm


RMS_NORM_BACKEND = "torch"
RMS_NORM_BLOCK_ROWS = 1


def configure_rms_norm_runtime(*, backend: str, block_rows: int) -> None:
	global RMS_NORM_BACKEND, RMS_NORM_BLOCK_ROWS
	if backend not in {"torch", "tilelang"}:
		raise ValueError(f"unsupported rms_norm backend: {backend}")
	resolved_block_rows = int(block_rows)
	if resolved_block_rows < 1:
		raise ValueError("rms_norm block_rows must be positive")
	RMS_NORM_BACKEND = backend
	RMS_NORM_BLOCK_ROWS = resolved_block_rows


def rms_norm_runtime_state() -> tuple[str, int]:
	return RMS_NORM_BACKEND, RMS_NORM_BLOCK_ROWS


class RMSNorm(nn.Module):
	def __init__(self, dim: int, eps: float = 1e-6) -> None:
		super().__init__()
		self.weight = nn.Parameter(torch.ones(dim))
		self.eps = eps

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return rms_norm(
			x,
			self.weight,
			self.eps,
			backend=RMS_NORM_BACKEND,
			block_rows=RMS_NORM_BLOCK_ROWS,
		)


__all__ = [
	"RMS_NORM_BACKEND",
	"RMS_NORM_BLOCK_ROWS",
	"RMSNorm",
	"configure_rms_norm_runtime",
	"rms_norm_runtime_state",
]