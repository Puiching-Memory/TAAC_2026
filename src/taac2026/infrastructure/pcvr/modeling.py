"""Reusable PCVR model building blocks for experiment packages."""

from __future__ import annotations

import math
from typing import Literal, NamedTuple

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from taac2026.infrastructure.pcvr.tilelang_ops import flash_attention as run_flash_attention


class ModelInput(NamedTuple):
    user_int_feats: torch.Tensor
    item_int_feats: torch.Tensor
    user_dense_feats: torch.Tensor
    item_dense_feats: torch.Tensor
    seq_data: dict[str, torch.Tensor]
    seq_lens: dict[str, torch.Tensor]
    seq_time_buckets: dict[str, torch.Tensor]


def make_padding_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    positions = torch.arange(max_len, device=lengths.device).unsqueeze(0)
    return positions >= lengths.unsqueeze(1)


def safe_key_padding_mask(mask: torch.Tensor) -> torch.Tensor:
    if mask.numel() == 0:
        return mask
    all_masked = mask.all(dim=1, keepdim=True)
    if mask.shape[1] == 0:
        return mask
    first_column = torch.zeros_like(mask)
    first_column[:, :1] = True
    return torch.where(all_masked, mask & ~first_column, mask)


def maybe_gradient_checkpoint(function, *args, enabled: bool = False, **kwargs):
    if not enabled or not torch.is_grad_enabled():
        return function(*args, **kwargs)
    return checkpoint(function, *args, use_reentrant=False, **kwargs)


def masked_mean(tokens: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
    if tokens.shape[1] == 0:
        return tokens.new_zeros(tokens.shape[0], tokens.shape[-1])
    if padding_mask is None:
        return tokens.mean(dim=1)
    valid = (~padding_mask).to(tokens.dtype).unsqueeze(-1)
    return (tokens * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)


def masked_last(tokens: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    if tokens.shape[1] == 0:
        return tokens.new_zeros(tokens.shape[0], tokens.shape[-1])
    indices = lengths.clamp_min(1).clamp_max(tokens.shape[1]).to(torch.long) - 1
    batch_indices = torch.arange(tokens.shape[0], device=tokens.device)
    return tokens[batch_indices, indices]


def choose_num_heads(d_model: int, requested_heads: int) -> int:
    requested_heads = max(1, requested_heads)
    if d_model % requested_heads == 0:
        return requested_heads
    for heads in range(min(requested_heads, d_model), 0, -1):
        if d_model % heads == 0:
            return heads
    return 1


def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    num_heads: int,
    attn_mask: torch.Tensor | None,
    dropout_p: float,
    training: bool,
    backend: Literal["torch", "tilelang"] = "torch",
    is_causal: bool = False,
    block_m: int = 64,
    block_n: int = 64,
    num_stages: int = 1,
    threads: int = 128,
) -> torch.Tensor:
    batch_size, query_len, d_model = q.shape
    head_dim = d_model // num_heads
    q = q.view(batch_size, query_len, num_heads, head_dim).transpose(1, 2)
    k = k.view(batch_size, k.shape[1], num_heads, head_dim).transpose(1, 2)
    v = v.view(batch_size, v.shape[1], num_heads, head_dim).transpose(1, 2)
    output = run_flash_attention(
        q,
        k,
        v,
        backend=backend,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        training=training,
        is_causal=is_causal,
        block_m=block_m,
        block_n=block_n,
        num_stages=num_stages,
        threads=threads,
    )
    return output.transpose(1, 2).contiguous().view(batch_size, query_len, d_model)


def causal_valid_attention_mask(padding_mask: torch.Tensor, num_heads: int) -> torch.Tensor:
    batch_size, token_count = padding_mask.shape
    causal = torch.ones(token_count, token_count, dtype=torch.bool, device=padding_mask.device).tril()
    key_valid = ~padding_mask
    mask = causal.unsqueeze(0) & key_valid.unsqueeze(1)
    query_invalid = padding_mask.unsqueeze(-1)
    fallback = torch.eye(token_count, dtype=torch.bool, device=padding_mask.device).unsqueeze(0)
    mask = torch.where(query_invalid, fallback, mask)
    return mask.unsqueeze(1).expand(batch_size, num_heads, token_count, token_count)


def sinusoidal_positions(length: int, dim: int, device: torch.device) -> torch.Tensor:
    if length == 0:
        return torch.empty(0, dim, device=device)
    positions = torch.arange(length, dtype=torch.float32, device=device).unsqueeze(1)
    frequencies = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32, device=device) * (-math.log(10000.0) / dim))
    values = torch.zeros(length, dim, device=device)
    values[:, 0::2] = torch.sin(positions * frequencies)
    values[:, 1::2] = torch.cos(positions * frequencies[: values[:, 1::2].shape[1]])
    return values