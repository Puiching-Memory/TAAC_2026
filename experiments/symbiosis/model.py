"""Symbiosis v3: unified token-stream PCVR model.

Explicit role tokens (recent/memory/global) replace the collapsed free-latent
pooler.  SwiGLU FFN, gated pooling, cross/global interaction tokens, and
random-chunk sparse tokenization are the default.
"""

from __future__ import annotations

import math
import random

import torch
import torch.nn as nn

from taac2026.api import (
    DenseTokenProjector,
    EmbeddingParameterMixin,
    FeatureEmbeddingBank,
    ModelInput,
    RMSNorm,
    SequenceTokenizer,
    choose_num_heads,
    configure_rms_norm_runtime as _configure_rms_norm_runtime,
    make_padding_mask,
    masked_mean,
    maybe_gradient_checkpoint,
    safe_key_padding_mask,
    scaled_dot_product_attention,
    sinusoidal_positions,
)


SEQUENCE_STATS_DIM = 6


def _feature_width(feature_specs: list[tuple[int, int, int]]) -> int:
    return max((offset + length for _vocab_size, offset, length in feature_specs), default=0)


def configure_rms_norm_runtime(*, rms_norm_backend: str, rms_norm_block_rows: int) -> None:
    _configure_rms_norm_runtime(
        backend=rms_norm_backend,
        block_rows=rms_norm_block_rows,
    )


def _torch_is_compiling() -> bool:
    compiler = getattr(torch, "compiler", None)
    is_compiling = getattr(compiler, "is_compiling", None)
    if callable(is_compiling) and is_compiling():
        return True
    dynamo = getattr(torch, "_dynamo", None)
    dynamo_is_compiling = getattr(dynamo, "is_compiling", None)
    return bool(callable(dynamo_is_compiling) and dynamo_is_compiling())


class RandomChunkNonSequentialTokenizer(nn.Module):
    def __init__(
        self,
        feature_specs: list[tuple[int, int, int]],
        emb_dim: int,
        d_model: int,
        num_tokens: int,
        emb_skip_threshold: int,
        compress_high_cardinality: bool,
        seed: int,
    ) -> None:
        super().__init__()
        self.bank = FeatureEmbeddingBank(
            feature_specs,
            emb_dim,
            emb_skip_threshold,
            compress_high_cardinality=compress_high_cardinality,
        )
        self.feature_count = len(feature_specs)
        self.num_tokens = max(1, int(num_tokens)) if self.feature_count > 0 else 0
        self.d_model = int(d_model)
        indices = list(range(self.feature_count))
        random.Random(int(seed)).shuffle(indices)
        self.chunks = [indices[index :: self.num_tokens] for index in range(self.num_tokens)] if self.num_tokens > 0 else []
        self.projects = nn.ModuleList(
            nn.Sequential(
                nn.Linear(emb_dim, d_model),
                nn.SiLU(),
                nn.LayerNorm(d_model),
            )
            for _ in self.chunks
        )

    @property
    def embeddings(self):  # type: ignore[no-untyped-def]
        return self.bank.embeddings

    def forward(self, int_feats: torch.Tensor) -> torch.Tensor:
        batch_size = int_feats.shape[0]
        if self.num_tokens <= 0:
            return int_feats.new_zeros(batch_size, 0, self.d_model, dtype=torch.float32)
        field_tokens = self.bank(int_feats)
        pieces: list[torch.Tensor] = []
        for chunk, project in zip(self.chunks, self.projects, strict=True):
            if chunk:
                chunk_tokens = field_tokens[:, chunk, :].mean(dim=1)
            else:
                chunk_tokens = int_feats.new_zeros(batch_size, self.bank.output_dim, dtype=torch.float32)
            pieces.append(project(chunk_tokens))
        return torch.stack(pieces, dim=1)


class SemanticNonSequentialTokenizer(nn.Module):
    def __init__(
        self,
        feature_specs: list[tuple[int, int, int]],
        groups: list[list[int]],
        emb_dim: int,
        d_model: int,
        num_tokens: int,
        emb_skip_threshold: int,
        compress_high_cardinality: bool,
        mode: str,
        seed: int,
    ) -> None:
        super().__init__()
        self.mode = str(mode).strip().lower() or "group"
        if self.mode not in {"random_chunk", "group", "group_compressed"}:
            raise ValueError(f"unknown symbiosis_ns_tokenizer_mode: {mode}")
        self.bank = FeatureEmbeddingBank(
            feature_specs,
            emb_dim,
            emb_skip_threshold,
            compress_high_cardinality=compress_high_cardinality,
        )
        self.feature_count = len(feature_specs)
        self.d_model = int(d_model)
        clean_groups = [list(group) for group in groups if group]
        self.groups = clean_groups or [[index] for index in range(self.feature_count)]
        if self.feature_count <= 0:
            self.num_tokens = 0
            self.chunks: list[list[int]] = []
        elif self.mode == "random_chunk":
            self.num_tokens = max(1, int(num_tokens))
            indices = list(range(self.feature_count))
            random.Random(int(seed)).shuffle(indices)
            self.chunks = [indices[index :: self.num_tokens] for index in range(self.num_tokens)]
        elif self.mode == "group":
            self.num_tokens = len(self.groups)
            self.chunks = self.groups
        else:
            self.num_tokens = max(1, int(num_tokens))
            self.chunks = self.groups
        self.group_project = nn.Sequential(
            nn.Linear(emb_dim, d_model),
            nn.SiLU(),
            nn.LayerNorm(d_model),
        )
        if self.mode == "group_compressed" and self.feature_count > 0:
            self.compress_project = nn.Sequential(
                RMSNorm(d_model * len(self.groups)),
                nn.Linear(d_model * len(self.groups), self.num_tokens * d_model),
                nn.SiLU(),
                nn.LayerNorm(self.num_tokens * d_model),
            )
        else:
            self.compress_project = None

    @property
    def embeddings(self):  # type: ignore[no-untyped-def]
        return self.bank.embeddings

    def forward(self, int_feats: torch.Tensor) -> torch.Tensor:
        batch_size = int_feats.shape[0]
        if self.num_tokens <= 0:
            return int_feats.new_zeros(batch_size, 0, self.d_model, dtype=torch.float32)
        field_tokens = self.bank(int_feats)
        pieces: list[torch.Tensor] = []
        for chunk in self.chunks:
            valid_indices = [index for index in chunk if 0 <= index < field_tokens.shape[1]]
            if valid_indices:
                pieces.append(field_tokens[:, valid_indices, :].mean(dim=1))
            else:
                pieces.append(int_feats.new_zeros(batch_size, self.bank.output_dim, dtype=torch.float32))
        group_tokens = self.group_project(torch.stack(pieces, dim=1))
        if self.compress_project is None:
            return group_tokens
        return self.compress_project(group_tokens.reshape(batch_size, -1)).view(batch_size, self.num_tokens, self.d_model)


class DensePacketTokenizer(nn.Module):
    """Turns dense vectors into a few normalized packet tokens."""

    def __init__(self, input_dim: int, d_model: int, *, max_packets: int) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.d_model = int(d_model)
        self.num_tokens = 0
        self.chunk_dim = 0
        self.pad_size = 0
        if self.input_dim <= 0:
            self.projects = nn.ModuleList()
            return

        self.num_tokens = max(1, min(int(max_packets), self.input_dim))
        self.chunk_dim = (self.input_dim + self.num_tokens - 1) // self.num_tokens
        self.pad_size = self.chunk_dim * self.num_tokens - self.input_dim
        self.projects = nn.ModuleList(
            nn.Sequential(
                nn.Linear(self.chunk_dim, d_model),
                nn.SiLU(),
                nn.LayerNorm(d_model),
            )
            for _ in range(self.num_tokens)
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if self.num_tokens <= 0:
            return features.new_zeros(features.shape[0], 0, self.d_model, dtype=torch.float32)
        normalized = torch.nan_to_num(features.float(), nan=0.0, posinf=0.0, neginf=0.0)
        normalized = torch.sign(normalized) * torch.log1p(normalized.abs())
        if self.pad_size > 0:
            normalized = torch.cat([normalized, normalized.new_zeros(normalized.shape[0], self.pad_size)], dim=-1)
        chunks = normalized.view(normalized.shape[0], self.num_tokens, self.chunk_dim)
        return torch.stack(
            [project(chunks[:, token_index, :]) for token_index, project in enumerate(self.projects)],
            dim=1,
        )


class SequenceStatsTokenizer(nn.Module):
    def __init__(self, domain_count: int, d_model: int) -> None:
        super().__init__()
        self.input_dim = int(domain_count) * SEQUENCE_STATS_DIM
        self.d_model = int(d_model)
        if self.input_dim > 0:
            self.project = nn.Sequential(
                RMSNorm(self.input_dim),
                nn.Linear(self.input_dim, d_model),
                nn.SiLU(),
                nn.LayerNorm(d_model),
            )
        else:
            self.project = None

    @property
    def num_tokens(self) -> int:
        return int(self.project is not None)

    def forward(
        self,
        stats_by_domain: dict[str, torch.Tensor] | None,
        domains: list[str],
        reference: torch.Tensor,
    ) -> torch.Tensor:
        if self.project is None:
            return reference.new_zeros(reference.shape[0], 0, self.d_model, dtype=torch.float32)
        pieces: list[torch.Tensor] = []
        batch_size = int(reference.shape[0])
        for domain in domains:
            stats = stats_by_domain.get(domain) if stats_by_domain is not None else None
            if stats is None:
                stats = reference.new_zeros(batch_size, SEQUENCE_STATS_DIM, dtype=torch.float32)
            pieces.append(self._normalize(stats.to(reference.device)))
        return self.project(torch.cat(pieces, dim=-1)).unsqueeze(1)

    def _normalize(self, stats: torch.Tensor) -> torch.Tensor:
        normalized = torch.nan_to_num(stats.float(), nan=0.0, posinf=0.0, neginf=0.0).clone()
        if normalized.shape[-1] >= 3:
            normalized[..., :3] = torch.log1p(normalized[..., :3].clamp_min(0.0))
        if normalized.shape[-1] >= 5:
            normalized[..., 3:5] = normalized[..., 3:5].clamp(0.0, 1.0)
        if normalized.shape[-1] >= 6:
            normalized[..., 5] = torch.log1p(normalized[..., 5].clamp_min(0.0))
        return normalized


class SwiGLUFeedForward(nn.Module):
    def __init__(self, d_model: int, hidden_mult: int, dropout: float) -> None:
        super().__init__()
        hidden_dim = d_model * hidden_mult
        self.up = nn.Linear(d_model, hidden_dim * 2)
        self.dropout = nn.Dropout(dropout)
        self.down = nn.Linear(hidden_dim, d_model)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        gate, value = self.up(tokens).chunk(2, dim=-1)
        return self.down(self.dropout(nn.functional.silu(gate) * value))


class UnifiedSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.dropout = dropout
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.query_norm = RMSNorm(d_model)
        self.key_norm = RMSNorm(d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(
        self,
        tokens: torch.Tensor,
        padding_mask: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        query, key, value = self.qkv(tokens).chunk(3, dim=-1)
        query = self.query_norm(query)
        key = self.key_norm(key)
        query = query.to(dtype=value.dtype)
        key = key.to(dtype=value.dtype)
        if attention_mask is None:
            safe_mask = safe_key_padding_mask(padding_mask)
            attn_mask = (~safe_mask).view(padding_mask.shape[0], 1, 1, padding_mask.shape[1])
        else:
            attn_mask = attention_mask
        attended = scaled_dot_product_attention(
            query,
            key,
            value,
            num_heads=self.num_heads,
            attn_mask=attn_mask,
            dropout_p=self.dropout,
            training=self.training,
        )
        attended = attended * (~padding_mask).to(attended.dtype).unsqueeze(-1)
        return self.out(attended)


class UnifiedInteractionBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, hidden_mult: int, dropout: float) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(d_model)
        self.attention = UnifiedSelfAttention(d_model, num_heads, dropout)
        self.ffn_norm = RMSNorm(d_model)
        self.ffn = SwiGLUFeedForward(d_model, hidden_mult, dropout)

    def forward(
        self,
        tokens: torch.Tensor,
        padding_mask: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        attn_tokens = tokens + self.attention(self.attn_norm(tokens), padding_mask, attention_mask)
        ffn_tokens = attn_tokens + self.ffn(self.ffn_norm(attn_tokens))
        return ffn_tokens, attn_tokens


class GatedTokenPooler(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.score = nn.Linear(d_model, 1)

    def forward(self, tokens: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        if tokens.shape[1] == 0:
            return tokens.new_zeros(tokens.shape[0], tokens.shape[-1])
        scores = self.score(tokens).squeeze(-1).masked_fill(padding_mask, float("-inf"))
        all_padded = padding_mask.all(dim=1, keepdim=True)
        scores = scores.masked_fill(all_padded, 0.0)
        weights = torch.softmax(scores.float(), dim=-1).to(tokens.dtype).unsqueeze(-1)
        pooled = (tokens * weights).sum(dim=1)
        return pooled * (~all_padded).to(tokens.dtype)


class MultiHeadCrossTokenizer(nn.Module):
    def __init__(self, d_model: int, num_tokens: int) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.num_tokens = max(1, int(num_tokens))
        self.explicit_project = nn.Sequential(
            RMSNorm(d_model * 5),
            nn.Linear(d_model * 5, self.num_tokens * d_model),
            nn.SiLU(),
            nn.LayerNorm(self.num_tokens * d_model),
        )
        self.user_factor = nn.Linear(d_model, self.num_tokens * d_model)
        self.item_factor = nn.Linear(d_model, self.num_tokens * d_model)
        self.sequence_factor = nn.Linear(d_model, self.num_tokens * d_model)
        self.sequence_item_factor = nn.Linear(d_model, self.num_tokens * d_model)
        self.output_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        user_summary: torch.Tensor,
        item_summary: torch.Tensor,
        sequence_summary: torch.Tensor,
    ) -> torch.Tensor:
        explicit_input = torch.cat(
            [
                user_summary,
                item_summary,
                sequence_summary,
                user_summary * item_summary,
                (user_summary - item_summary).abs(),
            ],
            dim=-1,
        )
        batch_size = user_summary.shape[0]
        base = self.explicit_project(explicit_input).view(batch_size, self.num_tokens, self.d_model)
        profile_item = self.user_factor(user_summary).view(batch_size, self.num_tokens, self.d_model)
        profile_item = profile_item * self.item_factor(item_summary).view(batch_size, self.num_tokens, self.d_model)
        sequence_item = self.sequence_factor(sequence_summary).view(batch_size, self.num_tokens, self.d_model)
        sequence_item = sequence_item * self.sequence_item_factor(item_summary).view(batch_size, self.num_tokens, self.d_model)
        scale = math.sqrt(float(self.d_model))
        return self.output_norm(base + (profile_item + sequence_item) / scale)


class LearnedBlockCompressor(nn.Module):
    def __init__(self, d_model: int, block_size: int) -> None:
        super().__init__()
        self.block_size = max(1, int(block_size))
        self.value_projection = nn.Linear(d_model, d_model)
        self.weight_projection = nn.Linear(d_model, d_model)
        self.position_bias = nn.Parameter(torch.zeros(self.block_size, d_model))
        self.output_norm = RMSNorm(d_model)

    def forward(
        self,
        tokens: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, token_count, d_model = tokens.shape
        pad_len = (-token_count) % self.block_size
        if pad_len:
            tokens = nn.functional.pad(tokens, (0, 0, 0, pad_len))
            padding_mask = nn.functional.pad(padding_mask, (0, pad_len), value=True)
        blocks = tokens.view(batch_size, -1, self.block_size, d_model)
        block_mask = padding_mask.view(batch_size, -1, self.block_size)
        values = self.value_projection(blocks)
        logits = self.weight_projection(blocks) + self.position_bias.view(1, 1, self.block_size, d_model)
        logits = logits.masked_fill(block_mask.unsqueeze(-1), float("-inf"))
        weights = torch.softmax(logits.float(), dim=2)
        weights = weights.nan_to_num(nan=0.0).to(values.dtype)
        weights = weights * (~block_mask).to(values.dtype).unsqueeze(-1)
        weights = weights / weights.sum(dim=2, keepdim=True).clamp_min(torch.finfo(weights.dtype).eps)
        block_tokens = self.output_norm((weights * values).sum(dim=2))
        block_padding_mask = block_mask.all(dim=2)
        block_tokens = block_tokens * (~block_padding_mask).to(block_tokens.dtype).unsqueeze(-1)
        return block_tokens, block_padding_mask


class SparseBlockIndexer(nn.Module):
    def __init__(self, d_model: int, num_heads: int) -> None:
        super().__init__()
        self.num_heads = max(1, int(num_heads))
        self.index_dim = max(8, d_model // self.num_heads)
        self.query_down = nn.Linear(d_model, self.index_dim)
        self.query_up = nn.Linear(self.index_dim, self.num_heads * self.index_dim)
        self.key_projection = nn.Linear(d_model, self.index_dim)
        self.head_weight = nn.Linear(d_model, self.num_heads)
        self.query_norm = RMSNorm(self.index_dim)
        self.key_norm = RMSNorm(self.index_dim)

    def forward(
        self,
        query: torch.Tensor,
        blocks: torch.Tensor,
        block_mask: torch.Tensor,
        top_k: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if top_k <= 0 or blocks.shape[1] == 0:
            empty_blocks = blocks.new_zeros(blocks.shape[0], 0, blocks.shape[-1])
            empty_mask = torch.ones(blocks.shape[0], 0, dtype=torch.bool, device=blocks.device)
            return empty_blocks, empty_mask
        keep_count = min(int(top_k), blocks.shape[1])
        query_latent = self.query_norm(self.query_down(query))
        query_heads = self.query_up(query_latent).view(query.shape[0], self.num_heads, self.index_dim)
        keys = self.key_norm(self.key_projection(blocks))
        scores_by_head = torch.relu(torch.einsum("bhi,bni->bhn", query_heads, keys))
        head_weights = torch.softmax(self.head_weight(query), dim=-1).unsqueeze(-1)
        scores = (scores_by_head * head_weights).sum(dim=1)
        scores = scores.masked_fill(block_mask, float("-inf"))
        indices = torch.topk(scores, k=keep_count, dim=1).indices
        gather_index = indices.unsqueeze(-1).expand(-1, -1, blocks.shape[-1])
        return blocks.gather(1, gather_index), block_mask.gather(1, indices)


class SequenceMemoryEncoder(nn.Module):
    """Builds bounded sequence memory tokens from long, repetitive domains."""

    def __init__(
        self,
        seq_vocab_sizes: dict[str, list[int]],
        emb_dim: int,
        d_model: int,
        num_time_buckets: int,
        emb_skip_threshold: int,
        seq_id_threshold: int,
        compress_high_cardinality: bool,
        *,
        num_index_heads: int,
        hidden_mult: int,
        dropout: float,
        recent_tokens: int,
        recent_output_tokens: int,
        memory_block_size: int,
        memory_top_k: int,
        memory_output_tokens: int,
        use_compressed_memory: bool,
        use_temporal_encoder: bool,
    ) -> None:
        super().__init__()
        self.seq_domains = sorted(seq_vocab_sizes)
        self.d_model = int(d_model)
        self.recent_tokens = max(0, int(recent_tokens))
        self.recent_output_tokens = max(0, int(recent_output_tokens))
        self.memory_block_size = max(1, int(memory_block_size))
        self.memory_top_k = max(0, int(memory_top_k))
        self.memory_output_tokens = max(0, int(memory_output_tokens))
        self.use_compressed_memory = bool(use_compressed_memory)
        self.use_temporal_encoder = bool(use_temporal_encoder)
        self.sequence_tokenizers = nn.ModuleDict(
            {
                domain: SequenceTokenizer(
                    vocab_sizes,
                    emb_dim,
                    d_model,
                    num_time_buckets,
                    seq_id_threshold or emb_skip_threshold,
                    compress_high_cardinality=compress_high_cardinality,
                )
                for domain, vocab_sizes in seq_vocab_sizes.items()
            }
        )
        self.block_compressors = nn.ModuleDict(
            {domain: LearnedBlockCompressor(d_model, self.memory_block_size) for domain in self.seq_domains}
        )
        self.block_indexers = nn.ModuleDict({domain: SparseBlockIndexer(d_model, num_index_heads) for domain in self.seq_domains})
        self.recent_encoders = nn.ModuleDict(
            {
                domain: UnifiedInteractionBlock(d_model, num_index_heads, hidden_mult, dropout)
                for domain in self.seq_domains
            }
        )
        self.role_norm = RMSNorm(d_model)
        self._training_diagnostics_enabled = False
        self._token_health_collecting = False
        self._token_health_scalars: dict[str, float] = {}
        self._token_health_total_valid_latents = 0.0
        self._token_health_total_latent_slots = 0.0
        self._token_health_total_empty_sequences = 0.0
        self._token_health_total_sequence_observations = 0.0
        self._token_health_total_sequence_length = 0.0
        self._token_health_batch_size = 0

    @property
    def tokens_per_domain(self) -> int:
        return self.recent_output_tokens + self.memory_output_tokens + 1

    def set_tensorboard_diagnostics_enabled(self, enabled: bool) -> None:
        self.set_training_diagnostics_enabled(enabled)

    def set_training_diagnostics_enabled(self, enabled: bool) -> None:
        self._training_diagnostics_enabled = bool(enabled)
        if not enabled:
            self._token_health_collecting = False

    def consume_tensorboard_scalars(self, phase: str) -> dict[str, float]:
        return self.consume_training_scalars(phase=phase)

    def consume_training_scalars(self, phase: str) -> dict[str, float]:
        scalars = self._token_health_scalars
        self._token_health_scalars = {}
        clean_phase = str(phase).strip().replace("/", "_") or "train"
        tagged_scalars: dict[str, float] = {}
        for metric_path, value in scalars.items():
            metric_family, metric_name = metric_path.split("/", 1)
            tagged_scalars[f"Symbiosis/{metric_family}/{clean_phase}/{metric_name}"] = value
        return tagged_scalars

    def _should_collect_token_health(self) -> bool:
        return self._training_diagnostics_enabled and not _torch_is_compiling()

    def _begin_token_health_collection(self) -> None:
        self._token_health_collecting = self._should_collect_token_health()
        if not self._token_health_collecting:
            return
        self._token_health_scalars = {}
        self._token_health_total_valid_latents = 0.0
        self._token_health_total_latent_slots = 0.0
        self._token_health_total_empty_sequences = 0.0
        self._token_health_total_sequence_observations = 0.0
        self._token_health_total_sequence_length = 0.0
        self._token_health_batch_size = 0

    def _finish_token_health_collection(self) -> None:
        if not self._token_health_collecting:
            return
        if self._token_health_total_latent_slots > 0.0:
            self._put_token_health_scalar(
                "token_health/all/latent_valid_ratio",
                self._token_health_total_valid_latents / self._token_health_total_latent_slots,
            )
        if self._token_health_batch_size > 0:
            self._put_token_health_scalar(
                "token_health/all/active_latent_tokens_mean",
                self._token_health_total_valid_latents / float(self._token_health_batch_size),
            )
        if self._token_health_total_sequence_observations > 0.0:
            self._put_token_health_scalar(
                "token_health/all/empty_sequence_rate",
                self._token_health_total_empty_sequences / self._token_health_total_sequence_observations,
            )
            self._put_token_health_scalar(
                "token_health/all/seq_len_mean",
                self._token_health_total_sequence_length / self._token_health_total_sequence_observations,
            )
        self._token_health_collecting = False

    def _put_token_health_scalar(self, metric_path: str, value: float | torch.Tensor) -> None:
        if isinstance(value, torch.Tensor):
            if value.numel() == 0:
                return
            value = float(value.detach().float().mean().cpu())
        else:
            value = float(value)
        if math.isfinite(value):
            self._token_health_scalars[metric_path] = value

    def _record_token_diversity(
        self,
        domain: str,
        latents: torch.Tensor,
        latent_mask: torch.Tensor,
    ) -> None:
        nonempty_samples = ~latent_mask.detach().all(dim=1)
        if not bool(nonempty_samples.any()):
            return
        sample_latents = latents.detach().float()[nonempty_samples]
        token_count = sample_latents.shape[1]
        norms = sample_latents.norm(dim=-1)
        prefix = f"latent_diversity/{domain}"
        self._put_token_health_scalar(f"{prefix}/token_norm_mean", norms.mean())
        self._put_token_health_scalar(f"{prefix}/token_norm_std", norms.std(unbiased=False))

        if token_count <= 1:
            self._put_token_health_scalar(f"{prefix}/effective_rank", float(token_count))
            return

        normalized = nn.functional.normalize(sample_latents, dim=-1, eps=1.0e-12)
        cosine = torch.einsum("bld,bmd->blm", normalized, normalized)
        row_indices, col_indices = torch.triu_indices(token_count, token_count, offset=1, device=latents.device)
        pairwise = cosine[:, row_indices, col_indices]
        self._put_token_health_scalar(f"{prefix}/mean_pairwise_cosine", pairwise.mean())
        self._put_token_health_scalar(f"{prefix}/max_pairwise_cosine", pairwise.max())

        eigenvalues = torch.linalg.eigvalsh(cosine.float()).clamp_min(0.0)
        eigenvalue_sum = eigenvalues.sum(dim=-1, keepdim=True)
        valid_rank = eigenvalue_sum.squeeze(-1) > 1.0e-12
        if bool(valid_rank.any()):
            probabilities = eigenvalues[valid_rank] / eigenvalue_sum[valid_rank]
            entropy = -(probabilities * probabilities.clamp_min(1.0e-12).log()).sum(dim=-1)
            self._put_token_health_scalar(f"{prefix}/effective_rank", entropy.exp().mean())

    def _record_role_token_health(
        self,
        *,
        domain: str,
        seq_len: torch.Tensor,
        tokens: torch.Tensor,
        mask: torch.Tensor,
    ) -> None:
        if not self._token_health_collecting:
            return
        clean_domain = str(domain).replace("/", "_")
        valid_tokens = ~mask.detach()
        seq_len_float = seq_len.detach().float()
        self._token_health_batch_size = max(self._token_health_batch_size, int(seq_len.shape[0]))
        self._token_health_total_valid_latents += float(valid_tokens.sum().cpu())
        self._token_health_total_latent_slots += float(mask.numel())
        self._token_health_total_empty_sequences += float((seq_len <= 0).sum().cpu())
        self._token_health_total_sequence_observations += float(seq_len.numel())
        self._token_health_total_sequence_length += float(seq_len_float.sum().cpu())
        prefix = f"token_health/{clean_domain}"
        self._put_token_health_scalar(f"{prefix}/latent_valid_ratio", valid_tokens.float().mean())
        self._put_token_health_scalar(f"{prefix}/active_latent_tokens_mean", valid_tokens.sum(dim=1).float().mean())
        self._put_token_health_scalar(f"{prefix}/empty_sequence_rate", (seq_len <= 0).float().mean())
        self._put_token_health_scalar(f"{prefix}/seq_len_mean", seq_len_float.mean())
        self._put_token_health_scalar(f"{prefix}/seq_len_p90", torch.quantile(seq_len_float, 0.9))
        self._record_token_diversity(clean_domain, tokens, mask)

    def _recent_window(
        self,
        tokens: torch.Tensor,
        lengths: torch.Tensor,
        keep_count: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        keep_count = min(max(0, int(keep_count)), tokens.shape[1])
        if keep_count <= 0:
            empty_tokens = tokens.new_zeros(tokens.shape[0], 0, tokens.shape[-1])
            empty_mask = torch.ones(tokens.shape[0], 0, dtype=torch.bool, device=tokens.device)
            return empty_tokens, empty_mask
        clamped_lengths = lengths.clamp(min=0, max=tokens.shape[1]).to(torch.long)
        start = (clamped_lengths - keep_count).clamp_min(0)
        offsets = torch.arange(keep_count, device=tokens.device).unsqueeze(0)
        positions = (start.unsqueeze(1) + offsets).clamp_max(max(0, tokens.shape[1] - 1))
        recent_tokens = tokens.gather(1, positions.unsqueeze(-1).expand(-1, -1, tokens.shape[-1]))
        recent_lengths = clamped_lengths.clamp_max(keep_count)
        recent_mask = make_padding_mask(recent_lengths, keep_count)
        return recent_tokens, recent_mask

    def _prefix_window(
        self,
        tokens: torch.Tensor,
        lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        prefix_capacity = max(0, tokens.shape[1] - min(self.recent_tokens, tokens.shape[1]))
        if prefix_capacity <= 0:
            empty_tokens = tokens.new_zeros(tokens.shape[0], 0, tokens.shape[-1])
            empty_mask = torch.ones(tokens.shape[0], 0, dtype=torch.bool, device=tokens.device)
            return empty_tokens, empty_mask
        prefix_lengths = (lengths.clamp(min=0, max=tokens.shape[1]).to(torch.long) - self.recent_tokens).clamp_min(0)
        prefix_tokens = tokens[:, :prefix_capacity, :]
        prefix_mask = make_padding_mask(prefix_lengths.clamp_max(prefix_capacity), prefix_capacity)
        return prefix_tokens, prefix_mask

    def _pad_token_slots(
        self,
        tokens: torch.Tensor,
        padding_mask: torch.Tensor,
        slot_count: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        slot_count = max(0, int(slot_count))
        if tokens.shape[1] == slot_count:
            return tokens, padding_mask
        if tokens.shape[1] > slot_count:
            return tokens[:, :slot_count, :], padding_mask[:, :slot_count]
        pad_count = slot_count - tokens.shape[1]
        token_pad = tokens.new_zeros(tokens.shape[0], pad_count, tokens.shape[-1])
        mask_pad = torch.ones(tokens.shape[0], pad_count, dtype=torch.bool, device=tokens.device)
        return torch.cat([tokens, token_pad], dim=1), torch.cat([padding_mask, mask_pad], dim=1)

    def _compressed_blocks(
        self,
        domain: str,
        tokens: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.block_compressors[domain](tokens, padding_mask)

    def _select_topk_blocks(
        self,
        query: torch.Tensor,
        domain: str,
        blocks: torch.Tensor,
        block_mask: torch.Tensor,
        top_k: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.block_indexers[domain](query, blocks, block_mask, self.memory_top_k if top_k is None else top_k)

    def _add_recent_positions(self, tokens: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        if tokens.shape[1] == 0:
            return tokens
        positions = sinusoidal_positions(tokens.shape[1], self.d_model, tokens.device).unsqueeze(0)
        tokens = tokens + positions
        return tokens * (~padding_mask).to(tokens.dtype).unsqueeze(-1)

    def _add_sequence_positions(self, tokens: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        if tokens.shape[1] == 0:
            return tokens
        positions = sinusoidal_positions(tokens.shape[1], self.d_model, tokens.device).unsqueeze(0).to(tokens.dtype)
        tokens = tokens + positions
        return tokens * (~padding_mask).to(tokens.dtype).unsqueeze(-1)

    def _compact_raw_sequence(
        self,
        sequence: torch.Tensor,
        time_buckets: torch.Tensor | None,
        lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, tuple[tuple[str, int], ...]]:
        batch_size, feature_count, max_len = sequence.shape
        if max_len <= 0:
            empty_sequence = sequence.new_zeros(batch_size, feature_count, 0)
            empty_mask = torch.ones(batch_size, 0, dtype=torch.bool, device=sequence.device)
            return empty_sequence, None, empty_mask, ()

        clamped_lengths = lengths.clamp(min=0, max=max_len).to(torch.long)
        positions: list[torch.Tensor] = []
        masks: list[torch.Tensor] = []
        source_segments: list[tuple[str, int]] = []

        recent_keep = min(self.recent_tokens, max_len)
        prefix_capacity = max(0, max_len - recent_keep)
        memory_keep = min(self.memory_top_k if self.use_compressed_memory else 0, prefix_capacity)
        if memory_keep > 0:
            offsets = torch.arange(memory_keep, device=sequence.device)
            prefix_lengths = (clamped_lengths - recent_keep).clamp_min(0)
            memory_positions = (prefix_lengths.unsqueeze(1) * (offsets + 1) // (memory_keep + 1)).clamp_max(max_len - 1)
            memory_lengths = prefix_lengths.clamp_max(memory_keep)
            positions.append(memory_positions)
            masks.append(offsets.unsqueeze(0) >= memory_lengths.unsqueeze(1))
            source_segments.append(("memory", int(memory_keep)))

        if recent_keep > 0:
            offsets = torch.arange(recent_keep, device=sequence.device)
            start = (clamped_lengths - recent_keep).clamp_min(0)
            recent_positions = (start.unsqueeze(1) + offsets).clamp_max(max_len - 1)
            recent_lengths = clamped_lengths.clamp_max(recent_keep)
            positions.append(recent_positions)
            masks.append(offsets.unsqueeze(0) >= recent_lengths.unsqueeze(1))
            source_segments.append(("recent", int(recent_keep)))

        if not positions:
            empty_sequence = sequence.new_zeros(batch_size, feature_count, 0)
            empty_mask = torch.ones(batch_size, 0, dtype=torch.bool, device=sequence.device)
            return empty_sequence, None, empty_mask, ()

        gather_positions = torch.cat(positions, dim=1)
        padding_mask = torch.cat(masks, dim=1)
        gather_index = gather_positions.unsqueeze(1).expand(-1, feature_count, -1)
        compact_sequence = sequence.gather(2, gather_index)
        compact_time_buckets = None
        if time_buckets is not None:
            if time_buckets.shape[1] <= 0:
                compact_time_buckets = torch.zeros_like(gather_positions)
            else:
                time_positions = gather_positions.clamp_max(time_buckets.shape[1] - 1)
                compact_time_buckets = time_buckets.gather(1, time_positions)
        return compact_sequence, compact_time_buckets, padding_mask, tuple(source_segments)

    def _segment_tokens(
        self,
        tokens: torch.Tensor,
        padding_mask: torch.Tensor,
        source_segments: tuple[tuple[str, int], ...],
        segment_name: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        offset = 0
        for name, segment_size in source_segments:
            next_offset = offset + segment_size
            if name == segment_name:
                return tokens[:, offset:next_offset, :], padding_mask[:, offset:next_offset]
            offset = next_offset
        empty_tokens = tokens.new_zeros(tokens.shape[0], 0, tokens.shape[-1])
        empty_mask = torch.ones(tokens.shape[0], 0, dtype=torch.bool, device=tokens.device)
        return empty_tokens, empty_mask

    def _pool_role_token(
        self,
        query: torch.Tensor,
        tokens: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if tokens.shape[1] == 0:
            token = query.new_zeros(query.shape[0], 1, self.d_model)
            mask = torch.ones(query.shape[0], 1, dtype=torch.bool, device=query.device)
            return token, mask
        token = masked_mean(tokens, padding_mask).unsqueeze(1)
        mask = padding_mask.all(dim=1, keepdim=True)
        token = self.role_norm(token + query.unsqueeze(1))
        return token, mask

    def _query_condition_tokens(
        self,
        query: torch.Tensor,
        tokens: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        if tokens.shape[1] == 0:
            return tokens
        conditioned = self.role_norm(tokens + query.unsqueeze(1))
        return conditioned * (~padding_mask).to(conditioned.dtype).unsqueeze(-1)

    def _recent_role_tokens(
        self,
        query: torch.Tensor,
        domain: str,
        tokens: torch.Tensor,
        padding_mask: torch.Tensor,
        seq_len: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        keep_count = min(self.recent_output_tokens, self.recent_tokens if self.recent_tokens > 0 else self.recent_output_tokens)
        recent_tokens, recent_mask = self._recent_window(tokens, seq_len, keep_count)
        recent_tokens, recent_mask = self._pad_token_slots(recent_tokens, recent_mask, self.recent_output_tokens)
        recent_tokens = self._add_recent_positions(recent_tokens, recent_mask)
        if self.use_temporal_encoder and recent_tokens.shape[1] > 1:
            recent_tokens, _attn_tokens = self.recent_encoders[domain](recent_tokens, recent_mask)
            recent_tokens = recent_tokens * (~recent_mask).to(recent_tokens.dtype).unsqueeze(-1)
        return self._query_condition_tokens(query, recent_tokens, recent_mask), recent_mask

    def _memory_role_tokens(
        self,
        query: torch.Tensor,
        domain: str,
        tokens: torch.Tensor,
        seq_len: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.memory_output_tokens <= 0:
            empty_tokens = tokens.new_zeros(tokens.shape[0], 0, tokens.shape[-1])
            empty_mask = torch.ones(tokens.shape[0], 0, dtype=torch.bool, device=tokens.device)
            return empty_tokens, empty_mask
        prefix_tokens, prefix_mask = self._prefix_window(tokens, seq_len)
        if prefix_tokens.shape[1] == 0:
            empty_tokens = tokens.new_zeros(tokens.shape[0], self.memory_output_tokens, tokens.shape[-1])
            empty_mask = torch.ones(tokens.shape[0], self.memory_output_tokens, dtype=torch.bool, device=tokens.device)
            return empty_tokens, empty_mask
        if self.use_compressed_memory:
            blocks, block_mask = self._compressed_blocks(domain, prefix_tokens, prefix_mask)
            memory_tokens, memory_mask = self._select_topk_blocks(query, domain, blocks, block_mask, self.memory_output_tokens)
        else:
            memory_tokens, memory_mask = self._recent_window(prefix_tokens, seq_len.clamp_max(prefix_tokens.shape[1]), self.memory_output_tokens)
        memory_tokens, memory_mask = self._pad_token_slots(memory_tokens, memory_mask, self.memory_output_tokens)
        return self._query_condition_tokens(query, memory_tokens, memory_mask), memory_mask

    def _global_role_token(
        self,
        query: torch.Tensor,
        tokens: torch.Tensor,
        padding_mask: torch.Tensor,
        seq_len: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        global_token, global_mask = self._pool_role_token(query, tokens, padding_mask)
        global_mask = global_mask | (seq_len <= 0).view(-1, 1)
        global_token = global_token * (~global_mask).to(global_token.dtype).unsqueeze(-1)
        return global_token, global_mask

    def _role_tokens_from_raw(
        self,
        query: torch.Tensor,
        domain: str,
        sequence: torch.Tensor,
        time_buckets: torch.Tensor | None,
        seq_len: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _batch_size, _feature_count, max_len = sequence.shape
        if max_len <= 0 or self.tokens_per_domain <= 0:
            tokens = query.new_zeros(query.shape[0], self.tokens_per_domain, self.d_model)
            mask = torch.ones(query.shape[0], self.tokens_per_domain, dtype=torch.bool, device=query.device)
            self._record_role_token_health(domain=domain, seq_len=seq_len, tokens=tokens, mask=mask)
            return tokens, mask

        seq_len = seq_len.clamp(min=0, max=max_len).to(torch.long)
        full_mask = make_padding_mask(seq_len, max_len)
        full_tokens = self.sequence_tokenizers[domain](sequence, time_buckets)
        full_tokens = self._add_sequence_positions(full_tokens, full_mask)
        recent_tokens, recent_mask = self._recent_role_tokens(query, domain, full_tokens, full_mask, seq_len)
        memory_tokens, memory_mask = self._memory_role_tokens(query, domain, full_tokens, seq_len)
        global_token, global_token_mask = self._global_role_token(query, full_tokens, full_mask, seq_len)

        tokens = torch.cat([recent_tokens, memory_tokens, global_token], dim=1)
        mask = torch.cat([recent_mask, memory_mask, global_token_mask], dim=1)
        self._record_role_token_health(domain=domain, seq_len=seq_len, tokens=tokens, mask=mask)
        return tokens, mask

    def forward(
        self,
        inputs: ModelInput,
        query: torch.Tensor,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[int]]:
        self._begin_token_health_collection()
        pieces: list[torch.Tensor] = []
        masks: list[torch.Tensor] = []
        domain_indices: list[int] = []
        for domain_index, domain in enumerate(self.seq_domains):
            raw_sequence = inputs.seq_data[domain]
            seq_len = inputs.seq_lens[domain].to(raw_sequence.device).clamp_max(raw_sequence.shape[2])
            role_tokens, role_mask = self._role_tokens_from_raw(
                query,
                domain,
                raw_sequence,
                inputs.seq_time_buckets.get(domain),
                seq_len,
            )
            pieces.append(role_tokens)
            masks.append(role_mask)
            domain_indices.append(domain_index)
        self._finish_token_health_collection()
        return pieces, masks, domain_indices


class PCVRSymbiosis(EmbeddingParameterMixin, nn.Module):
    """Unified sequence-memory and field-interaction PCVR model."""

    _SHORTCUT_SPAN_NAMES = frozenset({"candidate", "cross", "global"})
    _CLASSIFIER_SPAN_NAMES = ("candidate", "cross", "global", "context", "item", "sequence")

    def __init__(
        self,
        user_int_feature_specs: list[tuple[int, int, int]],
        item_int_feature_specs: list[tuple[int, int, int]],
        user_dense_dim: int,
        item_dense_dim: int,
        seq_vocab_sizes: dict[str, list[int]],
        user_ns_groups: list[list[int]],
        item_ns_groups: list[list[int]],
        d_model: int = 256,
        emb_dim: int = 256,
        num_queries: int = 1,
        num_blocks: int = 4,
        num_heads: int = 8,
        seq_encoder_type: str = "transformer",
        hidden_mult: int = 4,
        dropout_rate: float = 0.01,
        seq_top_k: int = 50,
        seq_causal: bool = False,
        action_num: int = 1,
        num_time_buckets: int = 65,
        rank_mixer_mode: str = "full",
        use_rope: bool = False,
        rope_base: float = 10000.0,
        emb_skip_threshold: int = 0,
        seq_id_threshold: int = 10000,
        gradient_checkpointing: bool = False,
        ns_tokenizer_type: str = "rankmixer",
        user_ns_tokens: int = 5,
        item_ns_tokens: int = 2,
        symbiosis_use_dense_packets: bool = True,
        symbiosis_use_sequence_memory: bool = True,
        symbiosis_use_compressed_memory: bool = True,
        symbiosis_use_candidate_token: bool = True,
        symbiosis_use_item_prior: bool = True,
        symbiosis_use_domain_type: bool = True,
        symbiosis_use_cross_token: bool = True,
        symbiosis_use_global_token: bool = True,
        symbiosis_sparse_seed: int = 20260511,
        symbiosis_memory_block_size: int = 32,
        symbiosis_memory_top_k: int = 8,
        symbiosis_recent_tokens: int = 32,
        symbiosis_sequence_recent_output_tokens: int = 12,
        symbiosis_sequence_memory_output_tokens: int = 8,
        symbiosis_use_sequence_temporal_encoder: bool = True,
        symbiosis_ns_tokenizer_mode: str = "group",
        symbiosis_use_structured_fusion: bool = True,
        symbiosis_cross_token_count: int = 6,
        symbiosis_compile_fusion_core: bool = True,
        symbiosis_shortcut_dropout_rate: float = 0.0,
        symbiosis_compress_large_ids: bool = True,
        symbiosis_use_missing_signals: bool = True,
        symbiosis_use_sequence_stats: bool = True,
    ) -> None:
        super().__init__()
        del num_queries, seq_encoder_type, seq_top_k, seq_causal, rank_mixer_mode, use_rope, rope_base, ns_tokenizer_type
        num_heads = choose_num_heads(d_model, num_heads)
        self.d_model = int(d_model)
        self.action_num = int(action_num)
        self.gradient_checkpointing = bool(gradient_checkpointing)
        self.seq_domains = sorted(seq_vocab_sizes)
        self.symbiosis_use_dense_packets = bool(symbiosis_use_dense_packets)
        self.symbiosis_use_sequence_memory = bool(symbiosis_use_sequence_memory)
        self.symbiosis_use_compressed_memory = bool(symbiosis_use_compressed_memory)
        self.symbiosis_use_candidate_token = bool(symbiosis_use_candidate_token)
        self.symbiosis_use_item_prior = bool(symbiosis_use_item_prior)
        self.symbiosis_use_domain_type = bool(symbiosis_use_domain_type)
        self.symbiosis_use_cross_token = bool(symbiosis_use_cross_token)
        self.symbiosis_use_global_token = bool(symbiosis_use_global_token)
        self.symbiosis_sparse_seed = int(symbiosis_sparse_seed)
        self.symbiosis_memory_block_size = max(1, int(symbiosis_memory_block_size))
        self.symbiosis_memory_top_k = max(0, int(symbiosis_memory_top_k))
        self.symbiosis_recent_tokens = max(0, int(symbiosis_recent_tokens))
        self.symbiosis_sequence_recent_output_tokens = max(0, int(symbiosis_sequence_recent_output_tokens))
        self.symbiosis_sequence_memory_output_tokens = max(0, int(symbiosis_sequence_memory_output_tokens))
        self.symbiosis_use_sequence_temporal_encoder = bool(symbiosis_use_sequence_temporal_encoder)
        self.symbiosis_ns_tokenizer_mode = str(symbiosis_ns_tokenizer_mode).strip().lower() or "group"
        self.symbiosis_use_structured_fusion = bool(symbiosis_use_structured_fusion)
        self.symbiosis_cross_token_count = max(1, int(symbiosis_cross_token_count))
        self.symbiosis_compile_fusion_core = bool(symbiosis_compile_fusion_core)
        self.symbiosis_shortcut_dropout_rate = min(1.0, max(0.0, float(symbiosis_shortcut_dropout_rate)))
        self.symbiosis_compress_large_ids = bool(symbiosis_compress_large_ids)
        self.symbiosis_use_missing_signals = bool(symbiosis_use_missing_signals)
        self.symbiosis_use_sequence_stats = bool(symbiosis_use_sequence_stats)

        self.user_sparse = SemanticNonSequentialTokenizer(
            user_int_feature_specs,
            user_ns_groups,
            emb_dim,
            d_model,
            user_ns_tokens,
            emb_skip_threshold,
            self.symbiosis_compress_large_ids,
            self.symbiosis_ns_tokenizer_mode,
            self.symbiosis_sparse_seed,
        )
        self.item_sparse = SemanticNonSequentialTokenizer(
            item_int_feature_specs,
            item_ns_groups,
            emb_dim,
            d_model,
            item_ns_tokens,
            emb_skip_threshold,
            self.symbiosis_compress_large_ids,
            self.symbiosis_ns_tokenizer_mode,
            self.symbiosis_sparse_seed + 1,
        )

        if self.symbiosis_use_dense_packets:
            self.user_dense = DensePacketTokenizer(user_dense_dim, d_model, max_packets=2)
            self.item_dense = DensePacketTokenizer(item_dense_dim, d_model, max_packets=1)
            user_dense_tokens = self.user_dense.num_tokens
            item_dense_tokens = self.item_dense.num_tokens
        else:
            self.user_dense = DenseTokenProjector(user_dense_dim, d_model)
            self.item_dense = DenseTokenProjector(item_dense_dim, d_model)
            user_dense_tokens = int(user_dense_dim > 0)
            item_dense_tokens = int(item_dense_dim > 0)

        user_missing_dim = _feature_width(user_int_feature_specs) + int(user_dense_dim)
        item_missing_dim = _feature_width(item_int_feature_specs) + int(item_dense_dim)
        if self.symbiosis_use_missing_signals:
            self.user_missing = DensePacketTokenizer(user_missing_dim, d_model, max_packets=2)
            self.item_missing = DensePacketTokenizer(item_missing_dim, d_model, max_packets=1)
            user_missing_tokens = self.user_missing.num_tokens
            item_missing_tokens = self.item_missing.num_tokens
        else:
            self.user_missing = None
            self.item_missing = None
            user_missing_tokens = 0
            item_missing_tokens = 0

        self.sequence_stats_tokenizer = (
            SequenceStatsTokenizer(len(self.seq_domains), d_model)
            if self.symbiosis_use_sequence_stats
            else None
        )
        sequence_stats_tokens = self.sequence_stats_tokenizer.num_tokens if self.sequence_stats_tokenizer is not None else 0

        self.num_ns = self.user_sparse.num_tokens + self.item_sparse.num_tokens
        self.num_ns += user_dense_tokens + item_dense_tokens + int(self.symbiosis_use_candidate_token)
        self.num_ns += (self.symbiosis_cross_token_count if self.symbiosis_use_cross_token else 0)
        self.num_ns += int(self.symbiosis_use_global_token)
        self.num_ns += user_missing_tokens + item_missing_tokens + sequence_stats_tokens

        self.sequence_memory = (
            SequenceMemoryEncoder(
                seq_vocab_sizes,
                emb_dim,
                d_model,
                num_time_buckets,
                emb_skip_threshold,
                seq_id_threshold,
                self.symbiosis_compress_large_ids,
                num_index_heads=num_heads,
                hidden_mult=hidden_mult,
                dropout=dropout_rate,
                recent_tokens=self.symbiosis_recent_tokens,
                recent_output_tokens=self.symbiosis_sequence_recent_output_tokens,
                memory_block_size=self.symbiosis_memory_block_size,
                memory_top_k=self.symbiosis_memory_top_k,
                memory_output_tokens=self.symbiosis_sequence_memory_output_tokens,
                use_compressed_memory=self.symbiosis_use_compressed_memory,
                use_temporal_encoder=self.symbiosis_use_sequence_temporal_encoder,
            )
            if self.symbiosis_use_sequence_memory
            else None
        )
        self.num_sequence_tokens = (self.sequence_memory.tokens_per_domain * len(self.seq_domains)) if self.sequence_memory else 0

        self.sequence_type_base = 7
        next_type_id = self.sequence_type_base + len(self.seq_domains)
        self.missing_type_id = next_type_id
        if self.symbiosis_use_missing_signals:
            next_type_id += 1
        self.sequence_stats_type_id = next_type_id
        if self.symbiosis_use_sequence_stats:
            next_type_id += 1
        type_count = next_type_id
        self.type_embedding = nn.Embedding(type_count, d_model)
        self.candidate_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.sequence_query_projection = nn.Sequential(RMSNorm(d_model * 2), nn.Linear(d_model * 2, d_model), nn.SiLU())
        self.cross_project = MultiHeadCrossTokenizer(d_model, self.symbiosis_cross_token_count)
        self.global_project = nn.Sequential(RMSNorm(d_model * 4), nn.Linear(d_model * 4, d_model), nn.SiLU(), nn.LayerNorm(d_model))
        self.blocks = nn.ModuleList(
            [
                UnifiedInteractionBlock(d_model, num_heads, hidden_mult, dropout_rate)
                for _ in range(max(1, num_blocks))
            ]
        )
        self.final_norm = RMSNorm(d_model)
        self.context_pooler = GatedTokenPooler(d_model)
        self.item_pooler = GatedTokenPooler(d_model)
        self.sequence_pooler = GatedTokenPooler(d_model)
        self.cross_pooler = GatedTokenPooler(d_model)
        self.global_pooler = GatedTokenPooler(d_model)
        self.classifier = nn.Sequential(
            nn.Linear(d_model * len(self._CLASSIFIER_SPAN_NAMES), d_model),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(d_model, action_num),
        )
        self._fusion_core_compiled = False
        self._compiled_fusion_core = None
        self._training_diagnostics_enabled = False
        self._fusion_rank_scalars: dict[str, float] = {}
        self._span_usage_scalars: dict[str, float] = {}
        self._structural_mask_cache: dict[tuple, torch.Tensor] = {}

    def set_tensorboard_diagnostics_enabled(self, enabled: bool) -> None:
        self.set_training_diagnostics_enabled(enabled)

    def set_training_diagnostics_enabled(self, enabled: bool) -> None:
        self._training_diagnostics_enabled = bool(enabled)
        if self.sequence_memory is not None:
            self.sequence_memory.set_training_diagnostics_enabled(enabled)

    def consume_tensorboard_scalars(self, *, phase: str) -> dict[str, float]:
        return self.consume_training_scalars(phase=phase)

    def consume_training_scalars(self, *, phase: str) -> dict[str, float]:
        scalars = self.sequence_memory.consume_training_scalars(phase) if self.sequence_memory is not None else {}
        clean_phase = str(phase).strip().replace("/", "_") or "train"
        for metric_name, value in self._fusion_rank_scalars.items():
            scalars[f"Symbiosis/fusion_rank/{clean_phase}/{metric_name}"] = value
        self._fusion_rank_scalars = {}
        for metric_name, value in self._span_usage_scalars.items():
            scalars[f"Symbiosis/span_usage/{clean_phase}/{metric_name}"] = value
        self._span_usage_scalars = {}
        return scalars

    def _should_collect_fusion_rank(self) -> bool:
        return self._training_diagnostics_enabled and not _torch_is_compiling()

    def _record_fusion_rank(self, metric_name: str, tokens: torch.Tensor, padding_mask: torch.Tensor) -> None:
        if not self._should_collect_fusion_rank() or tokens.shape[1] <= 1:
            return
        valid_samples = ~padding_mask.detach().all(dim=1)
        if not bool(valid_samples.any()):
            return
        sample_tokens = tokens.detach().float()[valid_samples]
        sample_mask = padding_mask.detach()[valid_samples]
        sample_tokens = sample_tokens * (~sample_mask).to(sample_tokens.dtype).unsqueeze(-1)
        singular_values = torch.linalg.svdvals(sample_tokens)
        total = singular_values.sum(dim=-1, keepdim=True)
        valid = total.squeeze(-1) > 1.0e-12
        if not bool(valid.any()):
            return
        probabilities = singular_values[valid] / total[valid]
        entropy = -(probabilities * probabilities.clamp_min(1.0e-12).log()).sum(dim=-1)
        value = float(entropy.exp().mean().cpu())
        if math.isfinite(value):
            self._fusion_rank_scalars[metric_name] = value

    def _should_collect_span_usage(self) -> bool:
        return self._training_diagnostics_enabled and not _torch_is_compiling()

    def _put_span_usage_scalar(self, metric_name: str, value: float | torch.Tensor) -> None:
        if isinstance(value, torch.Tensor):
            if value.numel() == 0:
                return
            value = float(value.detach().float().mean().cpu())
        else:
            value = float(value)
        if math.isfinite(value):
            self._span_usage_scalars[metric_name] = value

    def _record_span_usage(
        self,
        span_name: str,
        tokens: torch.Tensor,
        padding_mask: torch.Tensor,
        summary: torch.Tensor,
    ) -> None:
        if not self._should_collect_span_usage():
            return
        clean_name = str(span_name).replace("/", "_")
        valid_tokens = ~padding_mask.detach()
        self._put_span_usage_scalar(f"{clean_name}/active_token_ratio", valid_tokens.float().mean())
        self._put_span_usage_scalar(f"{clean_name}/summary_norm_mean", summary.detach().float().norm(dim=-1).mean())
        if bool(valid_tokens.any()):
            token_norms = tokens.detach().float().norm(dim=-1)
            self._put_span_usage_scalar(f"{clean_name}/token_norm_mean", token_norms[valid_tokens].mean())
        else:
            self._put_span_usage_scalar(f"{clean_name}/token_norm_mean", 0.0)

    def _record_classifier_span_usage(self) -> None:
        if not self._should_collect_span_usage():
            return
        first_layer = self.classifier[0]
        if not isinstance(first_layer, nn.Linear) or first_layer.weight.shape[1] != self.d_model * len(self._CLASSIFIER_SPAN_NAMES):
            return
        block_norms: dict[str, torch.Tensor] = {}
        for span_index, span_name in enumerate(self._CLASSIFIER_SPAN_NAMES):
            start = span_index * self.d_model
            end = start + self.d_model
            block_norms[span_name] = first_layer.weight[:, start:end].detach().float().norm()
        total_norm = sum(block_norms.values()).clamp_min(1.0e-12)
        for span_name, block_norm in block_norms.items():
            self._put_span_usage_scalar(f"{span_name}/classifier_weight_norm", block_norm)
            self._put_span_usage_scalar(f"{span_name}/classifier_weight_share", block_norm / total_norm)

    def _shortcut_dropout_enabled(self) -> bool:
        return self.training and torch.is_grad_enabled() and self.symbiosis_shortcut_dropout_rate > 0.0

    def _maybe_apply_span_dropout(
        self,
        span_name: str,
        tokens: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if tokens.shape[1] == 0 or span_name not in self._SHORTCUT_SPAN_NAMES or not self._shortcut_dropout_enabled():
            return tokens, padding_mask
        keep = torch.rand(tokens.shape[0], 1, device=tokens.device) >= self.symbiosis_shortcut_dropout_rate
        tokens = tokens * keep.to(tokens.dtype).unsqueeze(-1)
        padding_mask = padding_mask | (~keep).expand(-1, tokens.shape[1])
        return tokens, padding_mask

    def _type_ids(self, batch_size: int, token_count: int, type_id: int, device: torch.device) -> torch.Tensor:
        return torch.full((batch_size, token_count), int(type_id), dtype=torch.long, device=device)

    def _add_type(self, tokens: torch.Tensor, type_id: int) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, token_count, _dim = tokens.shape
        mask = torch.zeros(batch_size, token_count, dtype=torch.bool, device=tokens.device)
        if token_count == 0 or not self.symbiosis_use_domain_type:
            return tokens, mask
        type_ids = self._type_ids(batch_size, token_count, type_id, tokens.device)
        return tokens + self.type_embedding(type_ids), mask

    def _sparse_tokens(self, inputs: ModelInput) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        user_tokens, user_mask = self._add_type(self.user_sparse(inputs.user_int_feats), 0)
        item_tokens, item_mask = self._add_type(self.item_sparse(inputs.item_int_feats), 2)
        return user_tokens, user_mask, item_tokens, item_mask

    def _dense_tokens(self, inputs: ModelInput) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        user_dense_tokens = dense_or_empty(self.user_dense(inputs.user_dense_feats), inputs.user_dense_feats, self.d_model)
        item_dense_tokens = dense_or_empty(self.item_dense(inputs.item_dense_feats), inputs.item_dense_feats, self.d_model)
        user_dense, user_dense_mask = self._add_type(user_dense_tokens, 1)
        item_dense, item_dense_mask = self._add_type(item_dense_tokens, 3)
        return user_dense, user_dense_mask, item_dense, item_dense_mask

    def _empty_aux_tokens(self, reference: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = reference.new_zeros(reference.shape[0], 0, self.d_model, dtype=torch.float32)
        mask = torch.ones(reference.shape[0], 0, dtype=torch.bool, device=reference.device)
        return tokens, mask

    def _missing_mask_or_default(self, mask: torch.Tensor | None, features: torch.Tensor, *, dense: bool) -> torch.Tensor:
        if mask is not None:
            return mask.to(features.device).bool()
        if dense:
            return ~torch.isfinite(features.float())
        return features <= 0

    def _missing_tokens(self, inputs: ModelInput) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.symbiosis_use_missing_signals or self.user_missing is None or self.item_missing is None:
            user_empty, user_empty_mask = self._empty_aux_tokens(inputs.user_int_feats)
            item_empty, item_empty_mask = self._empty_aux_tokens(inputs.item_int_feats)
            return user_empty, user_empty_mask, item_empty, item_empty_mask

        user_missing_features = torch.cat(
            [
                self._missing_mask_or_default(inputs.user_int_missing_mask, inputs.user_int_feats, dense=False).float(),
                self._missing_mask_or_default(inputs.user_dense_missing_mask, inputs.user_dense_feats, dense=True).float(),
            ],
            dim=-1,
        )
        item_missing_features = torch.cat(
            [
                self._missing_mask_or_default(inputs.item_int_missing_mask, inputs.item_int_feats, dense=False).float(),
                self._missing_mask_or_default(inputs.item_dense_missing_mask, inputs.item_dense_feats, dense=True).float(),
            ],
            dim=-1,
        )
        user_tokens = dense_or_empty(self.user_missing(user_missing_features), user_missing_features, self.d_model)
        item_tokens = dense_or_empty(self.item_missing(item_missing_features), item_missing_features, self.d_model)
        user_missing, user_missing_mask = self._add_type(user_tokens, self.missing_type_id)
        item_missing, item_missing_mask = self._add_type(item_tokens, self.missing_type_id)
        return user_missing, user_missing_mask, item_missing, item_missing_mask

    def _sequence_stats_tokens(self, inputs: ModelInput) -> tuple[torch.Tensor, torch.Tensor]:
        if self.sequence_stats_tokenizer is None:
            return self._empty_aux_tokens(inputs.user_int_feats)
        stats_tokens = self.sequence_stats_tokenizer(inputs.seq_stats, self.seq_domains, inputs.user_int_feats)
        return self._add_type(stats_tokens, self.sequence_stats_type_id)

    def _sequence_tokens(
        self,
        inputs: ModelInput,
        query: torch.Tensor,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[int]]:
        if self.sequence_memory is None:
            return [], [], []
        pieces, masks, domain_indices = self.sequence_memory(inputs, query)
        typed_pieces: list[torch.Tensor] = []
        for tokens, domain_index in zip(pieces, domain_indices, strict=True):
            if tokens.shape[1] == 0 or not self.symbiosis_use_domain_type:
                typed_pieces.append(tokens)
                continue
            type_ids = self._type_ids(tokens.shape[0], tokens.shape[1], self.sequence_type_base + domain_index, tokens.device)
            typed_pieces.append(tokens + self.type_embedding(type_ids))
        return typed_pieces, masks, domain_indices

    def _encode_tokens(self, inputs: ModelInput) -> tuple[torch.Tensor, torch.Tensor, dict[str, tuple[int, int]]]:
        user_sparse, user_sparse_mask, item_sparse, item_sparse_mask = self._sparse_tokens(inputs)
        user_dense, user_dense_mask, item_dense, item_dense_mask = self._dense_tokens(inputs)
        user_missing, user_missing_mask, item_missing, item_missing_mask = self._missing_tokens(inputs)
        sequence_stats, sequence_stats_mask = self._sequence_stats_tokens(inputs)

        user_context_tokens = torch.cat([user_sparse, user_dense, user_missing], dim=1)
        user_context_mask = torch.cat([user_sparse_mask, user_dense_mask, user_missing_mask], dim=1)
        item_context_tokens = torch.cat([item_sparse, item_dense, item_missing], dim=1)
        item_context_mask = torch.cat([item_sparse_mask, item_dense_mask, item_missing_mask], dim=1)
        user_context = masked_mean(user_context_tokens, user_context_mask)
        item_summary = masked_mean(item_context_tokens, item_context_mask)
        sequence_query = self.sequence_query_projection(torch.cat([user_context, item_summary], dim=-1))
        sequence_pieces, sequence_masks, _domain_indices = self._sequence_tokens(inputs, sequence_query)
        sequence_summary_pieces = list(sequence_pieces)
        sequence_summary_masks = list(sequence_masks)
        if sequence_stats.shape[1] > 0:
            sequence_summary_pieces.append(sequence_stats)
            sequence_summary_masks.append(sequence_stats_mask)
        if sequence_summary_pieces:
            sequence_tokens_raw = torch.cat(sequence_summary_pieces, dim=1)
            sequence_mask_raw = torch.cat(sequence_summary_masks, dim=1)
            sequence_summary = masked_mean(sequence_tokens_raw, sequence_mask_raw)
        else:
            sequence_summary = user_context.new_zeros(user_context.shape)

        pieces: list[torch.Tensor] = []
        masks: list[torch.Tensor] = []
        spans: dict[str, tuple[int, int]] = {}

        def add_piece(name: str, tokens: torch.Tensor, mask: torch.Tensor) -> None:
            if tokens.shape[1] <= 0:
                return
            tokens, mask = self._maybe_apply_span_dropout(name, tokens, mask)
            start = sum(piece.shape[1] for piece in pieces)
            pieces.append(tokens)
            masks.append(mask)
            previous = spans.get(name)
            spans[name] = (previous[0], start + tokens.shape[1]) if previous else (start, start + tokens.shape[1])

        if self.symbiosis_use_candidate_token:
            candidate = self.candidate_token.expand(inputs.user_int_feats.shape[0], -1, -1)
            if self.symbiosis_use_item_prior:
                candidate = candidate + item_summary.unsqueeze(1)
            candidate, candidate_mask = self._add_type(candidate, 4)
            add_piece("candidate", candidate, candidate_mask)

        if self.symbiosis_use_cross_token:
            cross_tokens = self.cross_project(user_context, item_summary, sequence_summary)
            cross_tokens, cross_mask = self._add_type(cross_tokens, 5)
            add_piece("cross", cross_tokens, cross_mask)

        if self.symbiosis_use_global_token:
            global_input = torch.cat([user_context, item_summary, sequence_summary, user_context * item_summary], dim=-1)
            global_token = self.global_project(global_input).unsqueeze(1)
            global_token, global_mask = self._add_type(global_token, 6)
            add_piece("global", global_token, global_mask)

        for tokens, mask in ((user_sparse, user_sparse_mask), (user_dense, user_dense_mask), (user_missing, user_missing_mask)):
            add_piece("context", tokens, mask)
        for tokens, mask in zip(sequence_pieces, sequence_masks, strict=True):
            add_piece("sequence", tokens, mask)
        add_piece("sequence", sequence_stats, sequence_stats_mask)
        for tokens, mask in ((item_sparse, item_sparse_mask), (item_dense, item_dense_mask), (item_missing, item_missing_mask)):
            add_piece("item", tokens, mask)

        tokens = torch.cat(pieces, dim=1)
        padding_mask = torch.cat(masks, dim=1)
        return tokens, padding_mask, spans

    def _span_positions(self, spans: dict[str, tuple[int, int]], token_count: int, device: torch.device) -> dict[str, torch.Tensor]:
        positions = torch.arange(token_count, device=device)
        return {name: (positions >= start) & (positions < end) for name, (start, end) in spans.items()}

    @staticmethod
    def _structural_mask_cache_key(spans: dict[str, tuple[int, int]], token_count: int, device: torch.device) -> tuple:
        """Build a hashable key from the span layout, token count, and device."""
        span_key = tuple(sorted(spans.items()))
        return ("symbiosis_structural", span_key, token_count, str(device))

    def _build_structural_mask(
        self,
        spans: dict[str, tuple[int, int]],
        token_count: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Build the structural (padding-independent) attention mask and cache it."""
        key = self._structural_mask_cache_key(spans, token_count, device)
        cached = self._structural_mask_cache.get(key)
        if cached is not None and cached.device == device:
            return cached
        span_positions = self._span_positions(spans, token_count, device)
        all_tokens = torch.ones(token_count, dtype=torch.bool, device=device)

        def keys_for(*names: str) -> torch.Tensor:
            allowed = torch.zeros(token_count, dtype=torch.bool, device=device)
            for name in names:
                allowed = allowed | span_positions.get(name, torch.zeros_like(allowed))
            return allowed

        special_keys = keys_for("candidate", "cross", "global")
        context_keys = keys_for("context", "candidate", "cross", "global")
        item_keys = keys_for("item", "sequence", "candidate", "cross", "global")
        sequence_keys = keys_for("sequence", "item", "candidate", "cross", "global")
        base = torch.zeros(token_count, token_count, dtype=torch.bool, device=device)
        for span_name in ("candidate", "cross", "global"):
            query_positions = span_positions.get(span_name)
            if query_positions is not None:
                base[query_positions, :] = all_tokens
        query_positions = span_positions.get("context")
        if query_positions is not None:
            base[query_positions, :] = context_keys | special_keys
        query_positions = span_positions.get("item")
        if query_positions is not None:
            base[query_positions, :] = item_keys | special_keys
        query_positions = span_positions.get("sequence")
        if query_positions is not None:
            base[query_positions, :] = sequence_keys | special_keys
        diagonal = torch.eye(token_count, dtype=torch.bool, device=device)
        base = base | diagonal
        self._structural_mask_cache[key] = base
        return base

    def _build_structured_attention_mask(
        self,
        padding_mask: torch.Tensor,
        spans: dict[str, tuple[int, int]],
    ) -> torch.Tensor | None:
        if not self.symbiosis_use_structured_fusion:
            return None
        token_count = padding_mask.shape[1]
        device = padding_mask.device
        structural = self._build_structural_mask(spans, token_count, device)
        key_valid = ~safe_key_padding_mask(padding_mask)
        return structural.unsqueeze(0).unsqueeze(0) & key_valid.unsqueeze(1).unsqueeze(2)

    def _run_fusion_core(
        self,
        tokens: torch.Tensor,
        padding_mask: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self._record_fusion_rank("input", tokens, padding_mask)
        for block_index, block in enumerate(self.blocks):
            tokens, _attn_tokens = maybe_gradient_checkpoint(
                block,
                tokens,
                padding_mask,
                attention_mask,
                enabled=self.gradient_checkpointing,
            )
            self._record_fusion_rank(f"block_{block_index}/attn", _attn_tokens, padding_mask)
            self._record_fusion_rank(f"block_{block_index}/ffn", tokens, padding_mask)
        tokens = self.final_norm(tokens)
        self._record_fusion_rank("output", tokens, padding_mask)
        return tokens

    def _embed(self, inputs: ModelInput) -> torch.Tensor:
        tokens, padding_mask, spans = self._encode_tokens(inputs)
        attention_mask = self._build_structured_attention_mask(padding_mask, spans)
        fusion_core = self._compiled_fusion_core if self._compiled_fusion_core is not None else self._run_fusion_core
        tokens = fusion_core(tokens, padding_mask, attention_mask)
        candidate_span = spans.get("candidate") or spans.get("context")
        cross_span = spans.get("cross") or spans.get("candidate") or spans.get("context")
        global_span = spans.get("global") or spans.get("candidate") or spans.get("context")
        context_span = spans.get("context") or spans.get("sequence") or spans.get("candidate")
        item_span = spans.get("item") or spans.get("candidate") or spans.get("context")
        sequence_span = spans.get("sequence") or spans.get("context") or spans.get("candidate")

        def span_tokens(span: tuple[int, int]) -> tuple[torch.Tensor, torch.Tensor]:
            start, end = span
            return tokens[:, start:end, :], padding_mask[:, start:end]

        candidate_tokens, candidate_mask = span_tokens(candidate_span)
        candidate_summary = masked_mean(candidate_tokens, candidate_mask)
        cross_tokens, cross_mask = span_tokens(cross_span)
        global_tokens, global_mask = span_tokens(global_span)
        context_tokens, context_mask = span_tokens(context_span)
        item_tokens, item_mask = span_tokens(item_span)
        sequence_tokens, sequence_mask = span_tokens(sequence_span)
        cross_summary = self.cross_pooler(cross_tokens, cross_mask)
        global_summary = self.global_pooler(global_tokens, global_mask)
        context_summary = self.context_pooler(context_tokens, context_mask)
        item_summary = self.item_pooler(item_tokens, item_mask)
        sequence_summary = self.sequence_pooler(sequence_tokens, sequence_mask)
        if not self.symbiosis_use_item_prior:
            item_summary = torch.zeros_like(item_summary)
        self._record_span_usage("candidate", candidate_tokens, candidate_mask, candidate_summary)
        self._record_span_usage("cross", cross_tokens, cross_mask, cross_summary)
        self._record_span_usage("global", global_tokens, global_mask, global_summary)
        self._record_span_usage("context", context_tokens, context_mask, context_summary)
        self._record_span_usage("item", item_tokens, item_mask, item_summary)
        self._record_span_usage("sequence", sequence_tokens, sequence_mask, sequence_summary)
        self._record_classifier_span_usage()
        return torch.cat(
            [candidate_summary, cross_summary, global_summary, context_summary, item_summary, sequence_summary],
            dim=-1,
        )

    @property
    def uses_internal_compile(self) -> bool:
        return self.symbiosis_compile_fusion_core

    def prepare_for_runtime_compile(self) -> None:
        if not self.symbiosis_compile_fusion_core or self._fusion_core_compiled:
            return
        self._compiled_fusion_core = torch.compile(self._run_fusion_core)
        self._fusion_core_compiled = True

    def forward(self, inputs: ModelInput) -> torch.Tensor:
        return self.classifier(self._embed(inputs))

    def predict(self, inputs: ModelInput) -> tuple[torch.Tensor, torch.Tensor]:
        embeddings = self._embed(inputs)
        return self.classifier(embeddings), embeddings


def dense_or_empty(tokens: torch.Tensor | None, features: torch.Tensor, d_model: int) -> torch.Tensor:
    if tokens is not None:
        return tokens
    return features.new_zeros(features.shape[0], 0, d_model, dtype=torch.float32)


__all__ = ["ModelInput", "PCVRSymbiosis"]
