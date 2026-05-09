"""Triton embedding-bag kernel builders."""

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


def build_embedding_bag_mean_forward_kernel(
    batch: int,
    bag_size: int,
    num_embeddings: int,
    emb_dim: int,
    *,
    block_rows: int,
    block_cols: int,
):
    _ensure_triton()
    resolved_block_cols = triton_next_power_of_2(block_cols)
    num_warps = triton_num_warps(resolved_block_cols)

    @triton.jit
    def embedding_bag_mean_forward_kernel(
        weight,
        values,
        out,
        BATCH: tl.constexpr,
        BAG_SIZE: tl.constexpr,
        NUM_EMBEDDINGS: tl.constexpr,
        EMB_DIM: tl.constexpr,
        BLOCK_ROWS: tl.constexpr,
        BLOCK_COLS: tl.constexpr,
    ):
        row_offsets = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
        col_offsets = tl.program_id(1) * BLOCK_COLS + tl.arange(0, BLOCK_COLS)
        col_mask = col_offsets < EMB_DIM
        accum = tl.zeros((BLOCK_ROWS, BLOCK_COLS), dtype=tl.float32)
        valid_counts = tl.zeros((BLOCK_ROWS,), dtype=tl.int32)

        for position in range(0, BAG_SIZE):
            token_ids = tl.load(
                values + row_offsets * BAG_SIZE + position,
                mask=row_offsets < BATCH,
                other=0,
            )
            valid = (token_ids > 0) & (token_ids < NUM_EMBEDDINGS) & (row_offsets < BATCH)
            valid_counts += tl.where(valid, 1, 0)
            weight_offsets = token_ids[:, None] * EMB_DIM + col_offsets[None, :]
            weight_values = tl.load(weight + weight_offsets, mask=valid[:, None] & col_mask[None, :], other=0.0)
            accum += weight_values.to(tl.float32)

        denominator = tl.maximum(valid_counts, 1).to(tl.float32)
        output = accum / denominator[:, None]
        tl.store(
            out + row_offsets[:, None] * EMB_DIM + col_offsets[None, :],
            output,
            mask=(row_offsets[:, None] < BATCH) & col_mask[None, :],
        )

    def runner(embedding_weight: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        out = torch.empty((batch, emb_dim), dtype=embedding_weight.dtype, device=embedding_weight.device)
        grid = (triton.cdiv(batch, block_rows), triton.cdiv(emb_dim, resolved_block_cols))
        embedding_bag_mean_forward_kernel[grid](
            embedding_weight,
            values,
            out,
            batch,
            bag_size,
            num_embeddings,
            emb_dim,
            block_rows,
            resolved_block_cols,
            num_warps=num_warps,
        )
        return out

    return runner


def build_embedding_bag_mean_backward_kernel(
    batch: int,
    bag_size: int,
    num_embeddings: int,
    emb_dim: int,
    *,
    block_rows: int,
    block_cols: int,
):
    _ensure_triton()
    resolved_block_cols = triton_next_power_of_2(block_cols)
    num_warps = triton_num_warps(resolved_block_cols)

    @triton.jit
    def embedding_bag_mean_backward_kernel(
        values,
        grad_out,
        grad_weight,
        BATCH: tl.constexpr,
        BAG_SIZE: tl.constexpr,
        NUM_EMBEDDINGS: tl.constexpr,
        EMB_DIM: tl.constexpr,
        BLOCK_ROWS: tl.constexpr,
        BLOCK_COLS: tl.constexpr,
    ):
        row_offsets = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
        col_offsets = tl.program_id(1) * BLOCK_COLS + tl.arange(0, BLOCK_COLS)
        col_mask = col_offsets < EMB_DIM
        valid_counts = tl.zeros((BLOCK_ROWS,), dtype=tl.int32)

        for position in range(0, BAG_SIZE):
            token_ids = tl.load(
                values + row_offsets * BAG_SIZE + position,
                mask=row_offsets < BATCH,
                other=0,
            )
            valid = (token_ids > 0) & (token_ids < NUM_EMBEDDINGS) & (row_offsets < BATCH)
            valid_counts += tl.where(valid, 1, 0)

        denominator = tl.maximum(valid_counts, 1).to(tl.float32)
        grad_values = tl.load(
            grad_out + row_offsets[:, None] * EMB_DIM + col_offsets[None, :],
            mask=(row_offsets[:, None] < BATCH) & col_mask[None, :],
            other=0.0,
        ).to(tl.float32)

        for position in range(0, BAG_SIZE):
            token_ids = tl.load(
                values + row_offsets * BAG_SIZE + position,
                mask=row_offsets < BATCH,
                other=0,
            )
            valid = (token_ids > 0) & (token_ids < NUM_EMBEDDINGS) & (row_offsets < BATCH)
            grad_update = grad_values / denominator[:, None]
            tl.atomic_add(
                grad_weight + token_ids[:, None] * EMB_DIM + col_offsets[None, :],
                grad_update,
                sem="relaxed",
                mask=valid[:, None] & col_mask[None, :],
            )

    def runner(values: torch.Tensor, grad_out: torch.Tensor) -> torch.Tensor:
        grad_weight = torch.zeros((num_embeddings, emb_dim), dtype=torch.float32, device=grad_out.device)
        grid = (triton.cdiv(batch, block_rows), triton.cdiv(emb_dim, resolved_block_cols))
        embedding_bag_mean_backward_kernel[grid](
            values,
            grad_out,
            grad_weight,
            batch,
            bag_size,
            num_embeddings,
            emb_dim,
            block_rows,
            resolved_block_cols,
            num_warps=num_warps,
        )
        return grad_weight

    return runner



__all__ = [
    "build_embedding_bag_mean_backward_kernel",
    "build_embedding_bag_mean_forward_kernel",
]
