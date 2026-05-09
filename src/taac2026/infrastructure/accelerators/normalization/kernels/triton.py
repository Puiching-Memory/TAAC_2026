"""Triton RMSNorm kernel builders."""

from __future__ import annotations

import torch

from taac2026.infrastructure.accelerators.triton_runtime import (
    tl,
    triton,
    triton_available,
    triton_next_power_of_2,
    triton_num_warps,
)


def _ensure_triton() -> None:
    if not triton_available():
        raise RuntimeError("triton is not installed")


def build_rms_norm_forward_kernel(
    rows: int,
    cols: int,
    block_rows: int,
    eps: float,
):
    _ensure_triton()
    block_cols = triton_next_power_of_2(cols)
    num_warps = triton_num_warps(block_cols)

    @triton.jit
    def rms_norm_forward_kernel(
        x,
        weight,
        out,
        inv_rms,
        ROWS: tl.constexpr,
        COLS: tl.constexpr,
        BLOCK_ROWS: tl.constexpr,
        BLOCK_COLS: tl.constexpr,
        EPS: tl.constexpr,
    ):
        row_offsets = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
        col_offsets = tl.arange(0, BLOCK_COLS)
        mask = (row_offsets[:, None] < ROWS) & (col_offsets[None, :] < COLS)
        x_values = tl.load(x + row_offsets[:, None] * COLS + col_offsets[None, :], mask=mask, other=0.0).to(tl.float32)
        weight_values = tl.load(weight + col_offsets, mask=col_offsets < COLS, other=0.0).to(tl.float32)
        row_scale = tl.rsqrt(tl.sum(x_values * x_values, axis=1) / COLS + EPS)
        out_values = x_values * row_scale[:, None] * weight_values[None, :]
        tl.store(out + row_offsets[:, None] * COLS + col_offsets[None, :], out_values, mask=mask)
        tl.store(inv_rms + row_offsets, row_scale, mask=row_offsets < ROWS)

    def runner(x: torch.Tensor, weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = torch.empty_like(x)
        inv_rms = torch.empty((rows,), dtype=torch.float32, device=x.device)
        grid = (triton.cdiv(rows, block_rows),)
        rms_norm_forward_kernel[grid](
            x,
            weight,
            out,
            inv_rms,
            rows,
            cols,
            block_rows,
            block_cols,
            float(eps),
            num_warps=num_warps,
        )
        return out, inv_rms

    return runner


def build_rms_norm_backward_kernel(
    rows: int,
    cols: int,
    block_rows: int,
):
    _ensure_triton()
    block_cols = triton_next_power_of_2(cols)
    num_warps = triton_num_warps(block_cols)

    @triton.jit
    def rms_norm_backward_kernel(
        x,
        weight,
        inv_rms,
        grad_out,
        grad_x,
        grad_weight_partial,
        ROWS: tl.constexpr,
        COLS: tl.constexpr,
        BLOCK_ROWS: tl.constexpr,
        BLOCK_COLS: tl.constexpr,
    ):
        block_id = tl.program_id(0)
        row_offsets = block_id * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
        col_offsets = tl.arange(0, BLOCK_COLS)
        mask = (row_offsets[:, None] < ROWS) & (col_offsets[None, :] < COLS)
        x_values = tl.load(x + row_offsets[:, None] * COLS + col_offsets[None, :], mask=mask, other=0.0).to(tl.float32)
        grad_values = tl.load(
            grad_out + row_offsets[:, None] * COLS + col_offsets[None, :],
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        weight_values = tl.load(weight + col_offsets, mask=col_offsets < COLS, other=0.0).to(tl.float32)
        inv_values = tl.load(inv_rms + row_offsets, mask=row_offsets < ROWS, other=0.0).to(tl.float32)

        weighted_grad = grad_values * weight_values[None, :]
        row_dot = tl.sum(weighted_grad * x_values, axis=1)
        inv_cubed = inv_values * inv_values * inv_values
        grad_x_values = weighted_grad * inv_values[:, None] - x_values * row_dot[:, None] * inv_cubed[:, None] / COLS
        grad_weight_values = grad_values * x_values * inv_values[:, None]
        grad_weight_block = tl.sum(grad_weight_values, axis=0)

        tl.store(grad_x + row_offsets[:, None] * COLS + col_offsets[None, :], grad_x_values, mask=mask)
        tl.store(grad_weight_partial + block_id * COLS + col_offsets, grad_weight_block, mask=col_offsets < COLS)

    def runner(
        x: torch.Tensor,
        weight: torch.Tensor,
        inv_rms: torch.Tensor,
        grad_out: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        grad_x = torch.empty_like(x)
        grad_weight_partial = torch.empty(
            (triton.cdiv(rows, block_rows), cols),
            dtype=torch.float32,
            device=x.device,
        )
        grid = (triton.cdiv(rows, block_rows),)
        rms_norm_backward_kernel[grid](
            x,
            weight,
            inv_rms,
            grad_out,
            grad_x,
            grad_weight_partial,
            rows,
            cols,
            block_rows,
            block_cols,
            num_warps=num_warps,
        )
        return grad_x, grad_weight_partial

    return runner


__all__ = [
    "build_rms_norm_backward_kernel",
    "build_rms_norm_forward_kernel",
]
