from __future__ import annotations

import torch
from torch import nn
import triton
import triton.language as tl


@triton.jit
def _rms_norm_forward_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    input_row_stride,
    output_row_stride,
    hidden_dim,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row_index = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < hidden_dim
    row_ptr = input_ptr + row_index * input_row_stride
    output_row_ptr = output_ptr + row_index * output_row_stride

    values = tl.load(row_ptr + offsets, mask=mask, other=0.0)
    weights = tl.load(weight_ptr + offsets, mask=mask, other=1.0)
    variance = tl.sum(values * values, axis=0) / hidden_dim
    normalized = values * tl.rsqrt(variance + eps)
    tl.store(output_row_ptr + offsets, normalized * weights, mask=mask)


def _select_num_warps(block_size: int) -> int:
    if block_size <= 256:
        return 4
    if block_size <= 1024:
        return 8
    return 16


def triton_rms_norm(hidden_states: torch.Tensor, weight: torch.Tensor, eps: float = 1.0e-6) -> torch.Tensor:
    if hidden_states.device.type != "cuda":
        raise ValueError("triton_rms_norm requires CUDA tensors")
    if weight.device != hidden_states.device:
        raise ValueError("weight must be on the same device as hidden_states")
    if hidden_states.shape[-1] != weight.shape[0]:
        raise ValueError("weight shape must match the last hidden dimension")

    hidden_dim = hidden_states.shape[-1]
    flattened = hidden_states.contiguous().view(-1, hidden_dim)
    output = torch.empty_like(flattened)

    block_size = min(65536, triton.next_power_of_2(hidden_dim))
    num_warps = _select_num_warps(block_size)
    _rms_norm_forward_kernel[(flattened.shape[0],)](
        flattened,
        weight,
        output,
        flattened.stride(0),
        output.stride(0),
        hidden_dim,
        eps,
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
    )
    return output.view_as(hidden_states)


class TritonRMSNorm(nn.Module):
    def __init__(self, hidden_dim: int, eps: float = 1.0e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_dim))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return triton_rms_norm(hidden_states, self.weight, eps=self.eps)


__all__ = ["TritonRMSNorm", "triton_rms_norm"]