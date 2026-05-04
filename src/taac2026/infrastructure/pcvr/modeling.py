"""Reusable PCVR model building blocks for experiment packages."""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Literal, NamedTuple

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from taac2026.infrastructure.pcvr.tilelang_ops import embedding_bag_mean, flash_attention, rms_norm


class ModelInput(NamedTuple):
    user_int_feats: torch.Tensor
    item_int_feats: torch.Tensor
    user_dense_feats: torch.Tensor
    item_dense_feats: torch.Tensor
    seq_data: dict[str, torch.Tensor]
    seq_lens: dict[str, torch.Tensor]
    seq_time_buckets: dict[str, torch.Tensor]


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


class FeatureEmbeddingBank(nn.Module):
    def __init__(self, feature_specs: list[tuple[int, int, int]], emb_dim: int, emb_skip_threshold: int = 0) -> None:
        super().__init__()
        self.feature_specs = list(feature_specs)
        self.emb_dim = emb_dim
        self.embeddings = nn.ModuleList()
        self._embedding_index: list[int] = []
        for vocab_size, _offset, _length in self.feature_specs:
            should_skip = int(vocab_size) <= 0 or (emb_skip_threshold > 0 and int(vocab_size) > emb_skip_threshold)
            if should_skip:
                self._embedding_index.append(-1)
            else:
                self._embedding_index.append(len(self.embeddings))
                self.embeddings.append(nn.Embedding(int(vocab_size) + 1, emb_dim, padding_idx=0))
        self.reset_parameters()

    @property
    def output_dim(self) -> int:
        return self.emb_dim

    def reset_parameters(self) -> None:
        for embedding in self.embeddings:
            nn.init.xavier_normal_(embedding.weight)
            embedding.weight.data[0].zero_()

    def forward(self, int_feats: torch.Tensor) -> torch.Tensor:
        batch_size = int_feats.shape[0]
        if not self.feature_specs:
            return int_feats.new_zeros(batch_size, 0, self.emb_dim, dtype=torch.float32)
        tokens: list[torch.Tensor] = []
        for feature_index, (vocab_size, offset, length) in enumerate(self.feature_specs):
            embedding_index = self._embedding_index[feature_index]
            if embedding_index < 0:
                tokens.append(int_feats.new_zeros(batch_size, self.emb_dim, dtype=torch.float32))
                continue
            values = int_feats[:, offset : offset + length].to(torch.long).clamp(min=0, max=int(vocab_size))
            tokens.append(embedding_bag_mean(self.embeddings[embedding_index].weight, values))
        return torch.stack(tokens, dim=1)


class NonSequentialTokenizer(nn.Module):
    def __init__(
        self,
        feature_specs: list[tuple[int, int, int]],
        groups: list[list[int]],
        emb_dim: int,
        d_model: int,
        num_tokens: int = 0,
        emb_skip_threshold: int = 0,
        force_auto_split: bool = False,
    ) -> None:
        super().__init__()
        self.bank = FeatureEmbeddingBank(feature_specs, emb_dim, emb_skip_threshold)
        self.groups = [list(group) for group in groups] or [[index] for index in range(len(feature_specs))]
        self.feature_count = len(feature_specs)
        self.num_tokens = int(num_tokens) if num_tokens > 0 else len(self.groups)
        self.auto_split = force_auto_split or self.num_tokens != len(self.groups)
        if self.auto_split:
            input_dim = max(1, self.feature_count * emb_dim)
            self.project = nn.Sequential(
                nn.Linear(input_dim, self.num_tokens * d_model),
                nn.SiLU(),
                nn.LayerNorm(self.num_tokens * d_model),
            )
        else:
            self.project = nn.Sequential(nn.Linear(emb_dim, d_model), nn.LayerNorm(d_model))
        self.d_model = d_model

    @property
    def embeddings(self) -> Iterable[nn.Embedding]:
        return self.bank.embeddings

    def forward(self, int_feats: torch.Tensor) -> torch.Tensor:
        batch_size = int_feats.shape[0]
        feature_tokens = self.bank(int_feats)
        if self.num_tokens <= 0:
            return int_feats.new_zeros(batch_size, 0, self.d_model, dtype=torch.float32)
        if self.auto_split:
            if feature_tokens.shape[1] == 0:
                flat = int_feats.new_zeros(batch_size, 1, dtype=torch.float32)
            else:
                flat = feature_tokens.reshape(batch_size, -1)
            return self.project(flat).view(batch_size, self.num_tokens, self.d_model)
        grouped_tokens: list[torch.Tensor] = []
        for group in self.groups:
            valid_indices = [index for index in group if 0 <= index < feature_tokens.shape[1]]
            if valid_indices:
                grouped_tokens.append(feature_tokens[:, valid_indices, :].mean(dim=1))
            else:
                grouped_tokens.append(int_feats.new_zeros(batch_size, self.bank.output_dim, dtype=torch.float32))
        return self.project(torch.stack(grouped_tokens, dim=1))


class DenseTokenProjector(nn.Module):
    def __init__(self, input_dim: int, d_model: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        if input_dim > 0:
            self.project = nn.Sequential(nn.Linear(input_dim, d_model), nn.SiLU(), nn.LayerNorm(d_model))
        else:
            self.project = None

    def forward(self, features: torch.Tensor) -> torch.Tensor | None:
        if self.project is None:
            return None
        return self.project(features).unsqueeze(1)


class SequenceTokenizer(nn.Module):
    def __init__(
        self,
        vocab_sizes: list[int],
        emb_dim: int,
        d_model: int,
        num_time_buckets: int = 0,
        emb_skip_threshold: int = 0,
    ) -> None:
        super().__init__()
        self.vocab_sizes = [int(value) for value in vocab_sizes]
        self.emb_dim = emb_dim
        self.embeddings = nn.ModuleList()
        self._embedding_index: list[int] = []
        for vocab_size in self.vocab_sizes:
            should_skip = vocab_size <= 0 or (emb_skip_threshold > 0 and vocab_size > emb_skip_threshold)
            if should_skip:
                self._embedding_index.append(-1)
            else:
                self._embedding_index.append(len(self.embeddings))
                self.embeddings.append(nn.Embedding(vocab_size + 1, emb_dim, padding_idx=0))
        input_dim = max(1, len(self.vocab_sizes) * emb_dim)
        self.project = nn.Sequential(nn.Linear(input_dim, d_model), nn.SiLU(), nn.LayerNorm(d_model))
        self.time_embedding = nn.Embedding(num_time_buckets, d_model, padding_idx=0) if num_time_buckets > 0 else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for embedding in self.embeddings:
            nn.init.xavier_normal_(embedding.weight)
            embedding.weight.data[0].zero_()
        if self.time_embedding is not None:
            nn.init.xavier_normal_(self.time_embedding.weight)
            self.time_embedding.weight.data[0].zero_()

    def forward(self, sequence: torch.Tensor, time_buckets: torch.Tensor | None = None) -> torch.Tensor:
        batch_size, feature_count, seq_len = sequence.shape
        pieces: list[torch.Tensor] = []
        for feature_index in range(feature_count):
            embedding_index = self._embedding_index[feature_index] if feature_index < len(self._embedding_index) else -1
            if embedding_index < 0:
                pieces.append(sequence.new_zeros(batch_size, seq_len, self.emb_dim, dtype=torch.float32))
                continue
            vocab_size = self.vocab_sizes[feature_index]
            values = sequence[:, feature_index, :].to(torch.long).clamp(min=0, max=vocab_size)
            pieces.append(self.embeddings[embedding_index](values))
        if pieces:
            token_input = torch.cat(pieces, dim=-1)
        else:
            token_input = sequence.new_zeros(batch_size, seq_len, 1, dtype=torch.float32)
        tokens = self.project(token_input)
        if self.time_embedding is not None and time_buckets is not None:
            time_values = time_buckets.to(torch.long).clamp(min=0, max=self.time_embedding.num_embeddings - 1)
            tokens = tokens + self.time_embedding(time_values)
        return tokens


class EmbeddingParameterMixin:
    def get_sparse_params(self) -> list[nn.Parameter]:
        sparse_ptrs = {module.weight.data_ptr() for module in self.modules() if isinstance(module, nn.Embedding)}
        return [parameter for parameter in self.parameters() if parameter.data_ptr() in sparse_ptrs]

    def get_dense_params(self) -> list[nn.Parameter]:
        sparse_ptrs = {parameter.data_ptr() for parameter in self.get_sparse_params()}
        return [parameter for parameter in self.parameters() if parameter.data_ptr() not in sparse_ptrs]

    def reinit_high_cardinality_params(self, cardinality_threshold: int = 10000) -> set[int]:
        reinitialized: set[int] = set()
        for module in self.modules():
            if not isinstance(module, nn.Embedding):
                continue
            if module.num_embeddings - 1 <= cardinality_threshold:
                continue
            nn.init.xavier_normal_(module.weight)
            module.weight.data[0].zero_()
            reinitialized.add(module.weight.data_ptr())
        return reinitialized


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
    output = flash_attention(
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