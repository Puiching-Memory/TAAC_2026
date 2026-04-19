from __future__ import annotations

import math
from functools import lru_cache

import torch
import torch.nn.functional as F
from torch import nn

from .norms import RMSNorm
from .triton_attention import triton_attention


def masked_last(tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    positions = torch.arange(mask.shape[1], device=mask.device).unsqueeze(0).expand_as(mask)
    last_indices = (positions * mask.long()).max(dim=1).values
    batch_indices = torch.arange(tokens.shape[0], device=tokens.device)
    return tokens[batch_indices, last_indices]


@lru_cache(maxsize=8)
def build_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    row_index = torch.arange(seq_len, device=device).unsqueeze(1)
    column_index = torch.arange(seq_len, device=device).unsqueeze(0)
    return (column_index <= row_index).unsqueeze(0).unsqueeze(0)


@lru_cache(maxsize=16)
def build_local_window_mask(seq_len: int, local_window: int, device: torch.device) -> torch.Tensor:
    row_index = torch.arange(seq_len, device=device).unsqueeze(1)
    column_index = torch.arange(seq_len, device=device).unsqueeze(0)
    return ((column_index <= row_index) & ((row_index - column_index) < max(1, local_window))).unsqueeze(0).unsqueeze(0)


@lru_cache(maxsize=8)
def build_unified_attention_mask(
    seq_len: int,
    n_feature_tokens: int,
    n_special_tokens: int,
    global_window: int,
    local_window: int,
    device: torch.device,
) -> torch.Tensor:
    row_index = torch.arange(seq_len, device=device).unsqueeze(1)
    column_index = torch.arange(seq_len, device=device).unsqueeze(0)
    causal = column_index <= row_index
    global_mask = column_index < global_window
    local_mask = (row_index - column_index) < max(1, local_window)
    mask = causal & (global_mask | local_mask)

    if n_feature_tokens > 0:
        mask[:n_feature_tokens, :n_feature_tokens] = True
        mask[n_feature_tokens:, :n_feature_tokens] = True

    if n_special_tokens > 0:
        mask[-n_special_tokens:, :] = True

    return mask.unsqueeze(0).unsqueeze(0)


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, rope_base: float = 10000.0, rope_fraction: float = 1.0) -> None:
        super().__init__()
        rotary_dim = int(head_dim * float(rope_fraction))
        rotary_dim = max(2, rotary_dim - (rotary_dim % 2))
        self.head_dim = head_dim
        self.rotary_dim = rotary_dim
        inv_freq = 1.0 / (rope_base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _build_cos_sin(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        positions = torch.arange(seq_len, device=device, dtype=dtype)
        freqs = torch.einsum("s,f->sf", positions, self.inv_freq.to(device=device, dtype=dtype))
        cos = torch.cos(freqs)
        sin = torch.sin(freqs)
        cos = torch.stack([cos, cos], dim=-1).reshape(seq_len, -1).unsqueeze(0).unsqueeze(0)
        sin = torch.stack([sin, sin], dim=-1).reshape(seq_len, -1).unsqueeze(0).unsqueeze(0)
        return cos, sin

    @staticmethod
    def _rotate_half(hidden_states: torch.Tensor) -> torch.Tensor:
        first_half = hidden_states[..., ::2]
        second_half = hidden_states[..., 1::2]
        rotated = torch.stack((-second_half, first_half), dim=-1)
        return rotated.flatten(-2)

    def apply_rotary(self, query_states: torch.Tensor, key_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _, _, seq_len, head_dim = query_states.shape
        if head_dim != self.head_dim or self.rotary_dim == 0:
            return query_states, key_states
        cos, sin = self._build_cos_sin(seq_len, query_states.device, query_states.dtype)

        def apply_one(hidden_states: torch.Tensor) -> torch.Tensor:
            rotary_part = hidden_states[..., : self.rotary_dim]
            pass_part = hidden_states[..., self.rotary_dim :]
            rotated = rotary_part * cos + self._rotate_half(rotary_part) * sin
            return torch.cat([rotated, pass_part], dim=-1)

        return apply_one(query_states), apply_one(key_states)


class RelativeTimeBias(nn.Module):
    def __init__(self, num_buckets: int) -> None:
        super().__init__()
        self.num_buckets = num_buckets
        self.bucket_weights = nn.Parameter(torch.empty(num_buckets + 1).normal_(mean=0.0, std=0.02))

    def bucketize(self, deltas: torch.Tensor) -> torch.Tensor:
        bucket_ids = (torch.log2(deltas.clamp(min=1.0)).floor() + 1).long()
        return bucket_ids.clamp(min=1, max=self.num_buckets)

    def forward(self, timestamps: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        seq_len = timestamps.shape[1]
        time_i = timestamps.unsqueeze(2)
        time_j = timestamps.unsqueeze(1)
        deltas = torch.clamp(time_i - time_j, min=0.0)
        bucket_ids = self.bucketize(deltas)

        pair_valid = valid_mask.unsqueeze(2) & valid_mask.unsqueeze(1)
        tril_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=timestamps.device))
        diag_mask = torch.eye(seq_len, dtype=torch.bool, device=timestamps.device)
        pair_valid = pair_valid & tril_mask.unsqueeze(0) & (~diag_mask.unsqueeze(0))
        bucket_ids = torch.where(pair_valid, bucket_ids, torch.zeros_like(bucket_ids))
        return self.bucket_weights[bucket_ids]


class FourierTimeEncoding(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_frequencies: int = 8,
        min_period: float = 1.0,
        max_period: float = 256.0,
    ) -> None:
        super().__init__()
        periods = torch.logspace(math.log10(min_period), math.log10(max_period), steps=num_frequencies)
        self.register_buffer("periods", periods, persistent=False)
        self.register_buffer("two_pi", torch.tensor(2.0 * math.pi), persistent=False)
        self.projection = nn.Linear(2 * num_frequencies, hidden_dim, bias=False)

    def forward(self, timestamps: torch.Tensor) -> torch.Tensor:
        angles = (timestamps.unsqueeze(-1) / self.periods) * self.two_pi
        features = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        return self.projection(features)


class SiLUAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkuv_proj = nn.Linear(dim, 4 * dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.norm_attn = nn.LayerNorm(self.head_dim)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _resolve_attention_mask(attn_mask: torch.Tensor | None) -> torch.Tensor | None:
        if attn_mask is None:
            return None
        if attn_mask.ndim == 4:
            return attn_mask[:, 0]
        if attn_mask.ndim == 3:
            return attn_mask
        raise ValueError("attn_mask must have shape [batch, query, key] or [batch, 1, query, key]")

    def forward(self, hidden_states: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        batch_size, token_count, hidden_dim = hidden_states.shape
        qkuv = F.silu(self.qkuv_proj(hidden_states))
        query, key, value, gate = qkuv.chunk(4, dim=-1)

        query = query.view(batch_size, token_count, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, token_count, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, token_count, self.num_heads, self.head_dim).transpose(1, 2)
        gate = gate.view(batch_size, token_count, self.num_heads, self.head_dim).transpose(1, 2)

        resolved_mask = self._resolve_attention_mask(attn_mask)
        use_triton_path = (not self.training) and (not hidden_states.requires_grad) and hidden_states.device.type == "cuda"
        if use_triton_path:
            attended = triton_attention(
                query,
                key,
                value,
                attention_mask=resolved_mask,
                mode="silu",
                backend="triton",
            )
            attended = self.dropout(self.norm_attn(attended) * gate)
        else:
            scores = torch.matmul(query, key.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
            weights = F.silu(scores)
            if resolved_mask is not None:
                weights = weights * resolved_mask.unsqueeze(1).to(dtype=weights.dtype)
            weights = self.dropout(weights)
            attended = torch.matmul(weights, value)
            attended = self.norm_attn(attended)
            attended = attended * gate

        attended = attended.transpose(1, 2).reshape(batch_size, token_count, hidden_dim)
        return self.out_proj(attended)


class HSTUFeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w2(F.silu(self.w1(hidden_states)) * self.w3(hidden_states)))


class HSTUBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.attn = SiLUAttention(hidden_dim, num_heads, dropout)
        self.ffn = HSTUFeedForward(hidden_dim, ffn_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        hidden_states = hidden_states + self.dropout(self.attn(self.norm1(hidden_states), attn_mask))
        hidden_states = hidden_states + self.dropout(self.ffn(self.norm2(hidden_states)))
        return hidden_states


class ULTRAHSTUBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float = 0.1,
        local_window: int = 16,
    ) -> None:
        super().__init__()
        self.local_window = local_window
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.local_attn = SiLUAttention(hidden_dim, num_heads, dropout)
        self.global_attn = SiLUAttention(hidden_dim, num_heads, dropout)
        self.mix_gate = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid(),
        )
        self.ffn = HSTUFeedForward(hidden_dim, ffn_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        normalized_states = self.norm1(hidden_states)
        batch_size, seq_len, _ = normalized_states.shape
        local_mask = build_local_window_mask(seq_len, self.local_window, normalized_states.device).expand(batch_size, -1, -1, -1)
        if attn_mask is not None:
            resolved_global_mask = attn_mask if attn_mask.ndim == 4 else attn_mask.unsqueeze(1)
            local_mask = local_mask & resolved_global_mask
        else:
            resolved_global_mask = None

        local_output = self.local_attn(normalized_states, local_mask)
        global_output = self.global_attn(normalized_states, resolved_global_mask)
        mix_gate = self.mix_gate(torch.cat([local_output, global_output], dim=-1))
        mixed_output = mix_gate * local_output + (1.0 - mix_gate) * global_output
        hidden_states = hidden_states + self.dropout(mixed_output)
        hidden_states = hidden_states + self.dropout(self.ffn(self.norm2(hidden_states)))
        return hidden_states


class GatedFusion(nn.Module):
    def __init__(self, dim: int, num_branches: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(dim, num_branches, bias=False)
        self.value_projs = nn.ModuleList([nn.Linear(dim, dim, bias=False) for _ in range(num_branches)])

    def forward(self, branch_outputs: list[torch.Tensor]) -> torch.Tensor:
        stacked = torch.stack(branch_outputs, dim=1)
        gate_input = stacked.mean(dim=1)
        gates = F.softmax(self.gate_proj(gate_input), dim=-1).unsqueeze(-1)
        values = torch.stack(
            [proj(branch_output) for proj, branch_output in zip(self.value_projs, branch_outputs, strict=True)],
            dim=1,
        )
        return (gates * values).sum(dim=1)


class BranchTransducer(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_layers: int, dropout: float, ffn_dim: int | None = None) -> None:
        super().__init__()
        resolved_ffn_dim = dim * 4 if ffn_dim is None else ffn_dim
        self.layers = nn.ModuleList(
            [HSTUBlock(hidden_dim=dim, num_heads=num_heads, ffn_dim=resolved_ffn_dim, dropout=dropout) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, hidden_states: torch.Tensor, token_mask: torch.Tensor) -> torch.Tensor:
        attn_mask = build_causal_mask(hidden_states.shape[1], hidden_states.device)
        attn_mask = attn_mask & token_mask.unsqueeze(1).unsqueeze(-1) & token_mask.unsqueeze(1).unsqueeze(2)
        for layer in self.layers:
            hidden_states = layer(hidden_states, attn_mask)
            hidden_states = hidden_states * token_mask.unsqueeze(-1).float()
        hidden_states = self.norm(hidden_states)
        return masked_last(hidden_states, token_mask)


class MixtureOfTransducers(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_layers: int, dropout: float, num_branches: int) -> None:
        super().__init__()
        self.branches = nn.ModuleList([BranchTransducer(dim, num_heads, num_layers, dropout) for _ in range(num_branches)])
        self.fusion = GatedFusion(dim, num_branches)
        self._cuda_streams: list[torch.cuda.Stream] | None = None
        self._streams_device: torch.device | None = None

    def _get_cuda_streams(self, device: torch.device) -> list[torch.cuda.Stream]:
        if self._cuda_streams is None or self._streams_device != device:
            self._cuda_streams = [torch.cuda.Stream(device=device) for _ in range(len(self.branches) - 1)]
            self._streams_device = device
        return self._cuda_streams

    def forward(
        self,
        branch_tokens_list: list[torch.Tensor],
        branch_mask_list: list[torch.Tensor],
    ) -> torch.Tensor:
        if branch_tokens_list[0].is_cuda and len(self.branches) >= 3:
            streams = self._get_cuda_streams(branch_tokens_list[0].device)
            outputs: list[torch.Tensor | None] = [None] * len(self.branches)
            for index, stream in enumerate(streams):
                with torch.cuda.stream(stream):
                    outputs[index] = self.branches[index](branch_tokens_list[index], branch_mask_list[index])
            outputs[-1] = self.branches[-1](branch_tokens_list[-1], branch_mask_list[-1])
            for stream in streams:
                torch.cuda.current_stream().wait_stream(stream)
            return self.fusion(outputs)  # type: ignore[arg-type]

        outputs = [
            branch(tokens, mask)
            for branch, tokens, mask in zip(self.branches, branch_tokens_list, branch_mask_list, strict=True)
        ]
        return self.fusion(outputs)


class BlockAttnRes(nn.Module):
    def __init__(self, dim: int, total_layers: int) -> None:
        super().__init__()
        self.dim = dim
        self.pseudo_queries = nn.ParameterList([nn.Parameter(torch.zeros(dim)) for _ in range(total_layers)])
        self.res_norm = RMSNorm(dim)

    def forward(
        self,
        layer_idx: int,
        layer_output: torch.Tensor,
        block_summaries: list[torch.Tensor],
        partial_sum: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        current_last = layer_output[:, -1, :]
        partial_sum = partial_sum + current_last

        if not block_summaries:
            return layer_output, partial_sum, current_last

        query = self.pseudo_queries[layer_idx]
        keys = self.res_norm(torch.stack(block_summaries, dim=1))
        scores = torch.matmul(keys, query) * (1.0 / math.sqrt(self.dim))
        weights = F.softmax(scores, dim=-1)

        inter_block = torch.matmul(weights.unsqueeze(1), keys).squeeze(1)
        combined = layer_output + inter_block.unsqueeze(1)
        current_last = combined[:, -1, :]
        partial_sum = partial_sum - layer_output[:, -1, :] + current_last
        return combined, partial_sum, current_last


class TimeAwareHSTU(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float,
        rope_fraction: float = 1.0,
        rope_base: float = 10000.0,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.qkvu_linear = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 6, bias=False),
            nn.SiLU(),
        )
        self.out_linear = nn.Linear(hidden_dim * 3, hidden_dim)
        self.output_dropout = nn.Dropout(dropout)
        self.rms_norm = RMSNorm(hidden_dim)
        self.rope = RotaryEmbedding(self.head_dim, rope_base=rope_base, rope_fraction=rope_fraction)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        rel_ts_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        fused = self.qkvu_linear(hidden_states)
        gated_states, value_states, query_states, key_states = torch.split(
            fused,
            [self.hidden_dim * 3, self.hidden_dim, self.hidden_dim, self.hidden_dim],
            dim=-1,
        )

        query_states = query_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        query_rope, key_rope = self.rope.apply_rotary(query_states, key_states)

        qk_attn = torch.matmul(query_states, key_states.transpose(-2, -1))
        qk_attn = F.relu(qk_attn) / max(seq_len, 1)

        qk_attn_rope = torch.matmul(query_rope, key_rope.transpose(-2, -1))
        qk_attn_rope = F.relu(qk_attn_rope) / max(seq_len, 1)

        if rel_ts_bias is None:
            rel_ts_bias = torch.zeros(batch_size, seq_len, seq_len, device=hidden_states.device, dtype=value_states.dtype)

        if attention_mask is not None:
            masked_positions = attention_mask.logical_not()
            qk_attn = qk_attn.masked_fill(masked_positions.unsqueeze(1), 0.0)
            qk_attn_rope = qk_attn_rope.masked_fill(masked_positions.unsqueeze(1), 0.0)
            rel_ts_bias = rel_ts_bias.masked_fill(masked_positions, 0.0)

        ts_output = torch.einsum("bnm,bhmd->bnhd", rel_ts_bias, value_states)
        rope_output = torch.einsum("bhnm,bhmd->bnhd", qk_attn_rope, value_states)
        plain_output = torch.einsum("bhnm,bhmd->bnhd", qk_attn, value_states)
        combined_output = torch.cat([rope_output, ts_output, plain_output], dim=-1).contiguous()
        combined_output = combined_output.view(batch_size, seq_len, self.hidden_dim * 3)
        next_hidden_states = self.out_linear(combined_output * gated_states)
        next_hidden_states = self.output_dropout(next_hidden_states)
        return self.rms_norm(next_hidden_states + hidden_states)


__all__ = [
    "BranchTransducer",
    "BlockAttnRes",
    "FourierTimeEncoding",
    "GatedFusion",
    "HSTUBlock",
    "HSTUFeedForward",
    "MixtureOfTransducers",
    "RelativeTimeBias",
    "RotaryEmbedding",
    "SiLUAttention",
    "TimeAwareHSTU",
    "build_causal_mask",
    "build_local_window_mask",
    "build_unified_attention_mask",
    "masked_last",
    "ULTRAHSTUBlock",
]