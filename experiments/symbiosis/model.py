"""Symbiosis v2: unified token-stream PCVR model.

This implementation keeps the experiment package focused on the current TAAC
data shape: strong item-side fields, very long and repetitive sequence domains,
and a latency-constrained inference path.  The ablation surface is intentionally
small and maps to structural choices instead of legacy parallel readers.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from taac2026.api import (
    DenseTokenProjector,
    EmbeddingParameterMixin,
    FeatureEmbeddingBank,
    ModelInput,
    NonSequentialTokenizer,
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


def configure_rms_norm_runtime(*, rms_norm_backend: str, rms_norm_block_rows: int) -> None:
    _configure_rms_norm_runtime(
        backend=rms_norm_backend,
        block_rows=rms_norm_block_rows,
    )


class FieldTokenProjector(nn.Module):
    """Projects each sparse field as a separate token for unified interaction."""

    def __init__(
        self,
        feature_specs: list[tuple[int, int, int]],
        emb_dim: int,
        d_model: int,
        emb_skip_threshold: int,
    ) -> None:
        super().__init__()
        self.bank = FeatureEmbeddingBank(feature_specs, emb_dim, emb_skip_threshold)
        self.feature_count = len(feature_specs)
        self.d_model = d_model
        self.project = nn.Sequential(
            nn.Linear(emb_dim, d_model),
            nn.SiLU(),
            nn.LayerNorm(d_model),
        )

    @property
    def num_tokens(self) -> int:
        return self.feature_count

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        tokens = self.bank(features)
        if tokens.shape[1] == 0:
            return features.new_zeros(features.shape[0], 0, self.d_model, dtype=torch.float32)
        return self.project(tokens)


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


class FeedForward(nn.Module):
    def __init__(self, d_model: int, hidden_mult: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model * hidden_mult),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * hidden_mult, d_model),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.net(tokens)


class UnifiedSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.dropout = dropout
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.query_norm = RMSNorm(d_model)
        self.key_norm = RMSNorm(d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, tokens: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        query, key, value = self.qkv(tokens).chunk(3, dim=-1)
        query = self.query_norm(query)
        key = self.key_norm(key)
        query = query.to(dtype=value.dtype)
        key = key.to(dtype=value.dtype)
        attn_mask = (~padding_mask).view(padding_mask.shape[0], 1, 1, padding_mask.shape[1])
        attended = scaled_dot_product_attention(
            query,
            key,
            value,
            num_heads=self.num_heads,
            attn_mask=attn_mask,
            dropout_p=self.dropout,
            training=self.training,
        )
        return self.out(attended)


class UnifiedInteractionBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, hidden_mult: int, dropout: float) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(d_model)
        self.attention = UnifiedSelfAttention(d_model, num_heads, dropout)
        self.ffn_norm = RMSNorm(d_model)
        self.ffn = FeedForward(d_model, hidden_mult, dropout)

    def forward(self, tokens: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        tokens = tokens + self.attention(self.attn_norm(tokens), padding_mask)
        return tokens + self.ffn(self.ffn_norm(tokens))


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
        logits = logits.masked_fill(block_mask.unsqueeze(-1), torch.finfo(logits.dtype).min)
        weights = torch.softmax(logits.float(), dim=2).to(values.dtype)
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
        scores = scores.masked_fill(block_mask, torch.finfo(scores.dtype).min)
        indices = torch.topk(scores, k=keep_count, dim=1).indices
        gather_index = indices.unsqueeze(-1).expand(-1, -1, blocks.shape[-1])
        return blocks.gather(1, gather_index), block_mask.gather(1, indices)


class TargetAwareLatentPooler(nn.Module):
    """Pools bounded sequence memory into fixed-budget candidate-aware latents."""

    def __init__(self, d_model: int, latent_tokens: int) -> None:
        super().__init__()
        self.latent_tokens = max(1, int(latent_tokens))
        self.score_scale = float(d_model) ** -0.5
        self.latents = nn.Parameter(torch.randn(self.latent_tokens, d_model) * 0.02)
        self.query_projection = nn.Linear(d_model, d_model)
        self.key_projection = nn.Linear(d_model, d_model)
        self.value_projection = nn.Linear(d_model, d_model)
        self.output_norm = RMSNorm(d_model)

    def forward(
        self,
        query: torch.Tensor,
        tokens: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        valid = (~padding_mask).to(tokens.dtype).unsqueeze(-1)
        masked_tokens = tokens * valid
        latent_query = self.latents.unsqueeze(0) + self.query_projection(query).unsqueeze(1)
        keys = self.key_projection(masked_tokens)
        values = self.value_projection(masked_tokens) * valid
        scores = torch.einsum("bld,bnd->bln", latent_query, keys) * self.score_scale
        safe_mask = safe_key_padding_mask(padding_mask).unsqueeze(1)
        scores = scores.masked_fill(safe_mask, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores.float(), dim=-1).to(values.dtype)
        latents = torch.einsum("bln,bnd->bld", weights, values)
        all_padded = padding_mask.all(dim=1, keepdim=True)
        latents = self.output_norm(latents)
        latents = latents * (~all_padded).to(latents.dtype).unsqueeze(-1)
        latent_mask = all_padded.expand(-1, self.latent_tokens)
        return latents, latent_mask


class SequenceMemoryEncoder(nn.Module):
    """Builds bounded sequence memory tokens from long, repetitive domains."""

    def __init__(
        self,
        seq_vocab_sizes: dict[str, list[int]],
        emb_dim: int,
        d_model: int,
        num_time_buckets: int,
        emb_skip_threshold: int,
        *,
        num_index_heads: int,
        recent_tokens: int,
        memory_block_size: int,
        memory_top_k: int,
        use_compressed_memory: bool,
        latent_tokens_per_domain: int,
    ) -> None:
        super().__init__()
        self.seq_domains = sorted(seq_vocab_sizes)
        self.d_model = int(d_model)
        self.recent_tokens = max(0, int(recent_tokens))
        self.memory_block_size = max(1, int(memory_block_size))
        self.memory_top_k = max(0, int(memory_top_k))
        self.use_compressed_memory = bool(use_compressed_memory)
        self.latent_tokens_per_domain = max(0, int(latent_tokens_per_domain))
        self.sequence_tokenizers = nn.ModuleDict(
            {
                domain: SequenceTokenizer(vocab_sizes, emb_dim, d_model, num_time_buckets, emb_skip_threshold)
                for domain, vocab_sizes in seq_vocab_sizes.items()
            }
        )
        self.block_compressors = nn.ModuleDict(
            {domain: LearnedBlockCompressor(d_model, self.memory_block_size) for domain in self.seq_domains}
        )
        self.block_indexers = nn.ModuleDict({domain: SparseBlockIndexer(d_model, num_index_heads) for domain in self.seq_domains})
        self.latent_poolers = nn.ModuleDict(
            {
                domain: TargetAwareLatentPooler(d_model, self.latent_tokens_per_domain)
                for domain in self.seq_domains
            }
            if self.latent_tokens_per_domain > 0
            else {}
        )

    def _recent_window(
        self,
        tokens: torch.Tensor,
        lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        keep_count = min(self.recent_tokens, tokens.shape[1])
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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.block_indexers[domain](query, blocks, block_mask, self.memory_top_k)

    def _add_recent_positions(self, tokens: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        if tokens.shape[1] == 0:
            return tokens
        positions = sinusoidal_positions(tokens.shape[1], self.d_model, tokens.device).unsqueeze(0)
        tokens = tokens + positions
        return tokens * (~padding_mask).to(tokens.dtype).unsqueeze(-1)

    def _latent_memory_tokens(
        self,
        query: torch.Tensor,
        domain: str,
        tokens: torch.Tensor,
        padding_mask: torch.Tensor,
        seq_len: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        source_pieces: list[torch.Tensor] = []
        source_masks: list[torch.Tensor] = []

        recent_tokens, recent_mask = self._recent_window(tokens, seq_len)
        if recent_tokens.shape[1] > 0:
            source_pieces.append(self._add_recent_positions(recent_tokens, recent_mask))
            source_masks.append(recent_mask)

        if self.use_compressed_memory:
            blocks, block_mask = self._compressed_blocks(domain, tokens, padding_mask)
            blocks, block_mask = self._select_topk_blocks(query, domain, blocks, block_mask)
            if blocks.shape[1] > 0:
                source_pieces.append(blocks)
                source_masks.append(block_mask)

        global_token = masked_mean(tokens, padding_mask).unsqueeze(1)
        global_mask = padding_mask.all(dim=1, keepdim=True)
        source_pieces.append(global_token)
        source_masks.append(global_mask)

        source_tokens = torch.cat(source_pieces, dim=1)
        source_mask = torch.cat(source_masks, dim=1)
        return self.latent_poolers[domain](query, source_tokens, source_mask)

    def _compact_raw_sequence(
        self,
        sequence: torch.Tensor,
        time_buckets: torch.Tensor | None,
        lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        batch_size, feature_count, max_len = sequence.shape
        if max_len <= 0:
            empty_sequence = sequence.new_zeros(batch_size, feature_count, 0)
            empty_mask = torch.ones(batch_size, 0, dtype=torch.bool, device=sequence.device)
            return empty_sequence, None, empty_mask

        clamped_lengths = lengths.clamp(min=0, max=max_len).to(torch.long)
        positions: list[torch.Tensor] = []
        masks: list[torch.Tensor] = []

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

        if recent_keep > 0:
            offsets = torch.arange(recent_keep, device=sequence.device)
            start = (clamped_lengths - recent_keep).clamp_min(0)
            recent_positions = (start.unsqueeze(1) + offsets).clamp_max(max_len - 1)
            recent_lengths = clamped_lengths.clamp_max(recent_keep)
            positions.append(recent_positions)
            masks.append(offsets.unsqueeze(0) >= recent_lengths.unsqueeze(1))

        if not positions:
            empty_sequence = sequence.new_zeros(batch_size, feature_count, 0)
            empty_mask = torch.ones(batch_size, 0, dtype=torch.bool, device=sequence.device)
            return empty_sequence, None, empty_mask

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
        return compact_sequence, compact_time_buckets, padding_mask

    def _latent_memory_tokens_from_raw(
        self,
        query: torch.Tensor,
        domain: str,
        sequence: torch.Tensor,
        time_buckets: torch.Tensor | None,
        seq_len: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        compact_sequence, compact_time_buckets, compact_mask = self._compact_raw_sequence(sequence, time_buckets, seq_len)
        if compact_sequence.shape[2] == 0:
            pooler = self.latent_poolers[domain]
            empty_tokens = query.new_zeros(query.shape[0], pooler.latent_tokens, self.d_model)
            empty_mask = torch.ones(query.shape[0], pooler.latent_tokens, dtype=torch.bool, device=query.device)
            return empty_tokens, empty_mask
        tokens = self.sequence_tokenizers[domain](compact_sequence, compact_time_buckets)
        tokens = self._add_recent_positions(tokens, compact_mask)
        return self.latent_poolers[domain](query, tokens, compact_mask)

    def forward(
        self,
        inputs: ModelInput,
        query: torch.Tensor,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[int]]:
        pieces: list[torch.Tensor] = []
        masks: list[torch.Tensor] = []
        domain_indices: list[int] = []
        for domain_index, domain in enumerate(self.seq_domains):
            raw_sequence = inputs.seq_data[domain]
            seq_len = inputs.seq_lens[domain].to(raw_sequence.device).clamp_max(raw_sequence.shape[2])
            if self.latent_tokens_per_domain > 0:
                latent_tokens, latent_mask = self._latent_memory_tokens_from_raw(
                    query,
                    domain,
                    raw_sequence,
                    inputs.seq_time_buckets.get(domain),
                    seq_len,
                )
                pieces.append(latent_tokens)
                masks.append(latent_mask)
                domain_indices.append(domain_index)
                continue

            tokens = self.sequence_tokenizers[domain](raw_sequence, inputs.seq_time_buckets.get(domain))
            padding_mask = make_padding_mask(seq_len, tokens.shape[1])

            recent_tokens, recent_mask = self._recent_window(tokens, seq_len)
            if recent_tokens.shape[1] > 0:
                recent_tokens = self._add_recent_positions(recent_tokens, recent_mask)
                masks.append(recent_mask)
                pieces.append(recent_tokens)
                domain_indices.append(domain_index)

            if self.use_compressed_memory:
                blocks, block_mask = self._compressed_blocks(domain, tokens, padding_mask)
                blocks, block_mask = self._select_topk_blocks(query, domain, blocks, block_mask)
                if blocks.shape[1] > 0:
                    pieces.append(blocks)
                    masks.append(block_mask)
                    domain_indices.append(domain_index)

            global_token = masked_mean(tokens, padding_mask).unsqueeze(1)
            global_mask = padding_mask.all(dim=1, keepdim=True)
            pieces.append(global_token)
            masks.append(global_mask)
            domain_indices.append(domain_index)
        return pieces, masks, domain_indices


class PCVRSymbiosis(EmbeddingParameterMixin, nn.Module):
    """Unified sequence-memory and field-interaction PCVR model."""

    def __init__(
        self,
        user_int_feature_specs: list[tuple[int, int, int]],
        item_int_feature_specs: list[tuple[int, int, int]],
        user_dense_dim: int,
        item_dense_dim: int,
        seq_vocab_sizes: dict[str, list[int]],
        user_ns_groups: list[list[int]],
        item_ns_groups: list[list[int]],
        d_model: int = 64,
        emb_dim: int = 64,
        num_queries: int = 1,
        num_blocks: int = 2,
        num_heads: int = 4,
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
        symbiosis_use_field_tokens: bool = False,
        symbiosis_use_dense_packets: bool = True,
        symbiosis_use_sequence_memory: bool = True,
        symbiosis_use_compressed_memory: bool = True,
        symbiosis_use_candidate_token: bool = True,
        symbiosis_use_item_prior: bool = True,
        symbiosis_use_domain_type: bool = True,
        symbiosis_memory_block_size: int = 32,
        symbiosis_memory_top_k: int = 8,
        symbiosis_recent_tokens: int = 32,
        symbiosis_sequence_latent_tokens: int = 3,
        symbiosis_compile_fusion_core: bool = True,
    ) -> None:
        super().__init__()
        del num_queries, seq_encoder_type, seq_top_k, seq_causal, rank_mixer_mode, use_rope, rope_base, seq_id_threshold
        num_heads = choose_num_heads(d_model, num_heads)
        self.d_model = int(d_model)
        self.action_num = int(action_num)
        self.gradient_checkpointing = bool(gradient_checkpointing)
        self.seq_domains = sorted(seq_vocab_sizes)
        self.symbiosis_use_field_tokens = bool(symbiosis_use_field_tokens)
        self.symbiosis_use_dense_packets = bool(symbiosis_use_dense_packets)
        self.symbiosis_use_sequence_memory = bool(symbiosis_use_sequence_memory)
        self.symbiosis_use_compressed_memory = bool(symbiosis_use_compressed_memory)
        self.symbiosis_use_candidate_token = bool(symbiosis_use_candidate_token)
        self.symbiosis_use_item_prior = bool(symbiosis_use_item_prior)
        self.symbiosis_use_domain_type = bool(symbiosis_use_domain_type)
        self.symbiosis_memory_block_size = max(1, int(symbiosis_memory_block_size))
        self.symbiosis_memory_top_k = max(0, int(symbiosis_memory_top_k))
        self.symbiosis_recent_tokens = max(0, int(symbiosis_recent_tokens))
        self.symbiosis_sequence_latent_tokens = max(0, int(symbiosis_sequence_latent_tokens))
        self.symbiosis_compile_fusion_core = bool(symbiosis_compile_fusion_core)

        if self.symbiosis_use_field_tokens:
            self.user_sparse = FieldTokenProjector(user_int_feature_specs, emb_dim, d_model, emb_skip_threshold)
            self.item_sparse = FieldTokenProjector(item_int_feature_specs, emb_dim, d_model, emb_skip_threshold)
        else:
            force_auto_split = ns_tokenizer_type == "rankmixer"
            self.user_sparse = NonSequentialTokenizer(
                user_int_feature_specs,
                user_ns_groups,
                emb_dim,
                d_model,
                user_ns_tokens,
                emb_skip_threshold,
                force_auto_split=force_auto_split,
            )
            self.item_sparse = NonSequentialTokenizer(
                item_int_feature_specs,
                item_ns_groups,
                emb_dim,
                d_model,
                item_ns_tokens,
                emb_skip_threshold,
                force_auto_split=force_auto_split,
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

        self.num_ns = self.user_sparse.num_tokens + self.item_sparse.num_tokens
        self.num_ns += user_dense_tokens + item_dense_tokens + int(self.symbiosis_use_candidate_token)

        self.sequence_memory = (
            SequenceMemoryEncoder(
                seq_vocab_sizes,
                emb_dim,
                d_model,
                num_time_buckets,
                emb_skip_threshold,
                num_index_heads=num_heads,
                recent_tokens=self.symbiosis_recent_tokens,
                memory_block_size=self.symbiosis_memory_block_size,
                memory_top_k=self.symbiosis_memory_top_k,
                use_compressed_memory=self.symbiosis_use_compressed_memory,
                latent_tokens_per_domain=self.symbiosis_sequence_latent_tokens,
            )
            if self.symbiosis_use_sequence_memory
            else None
        )

        type_count = 5 + len(self.seq_domains)
        self.type_embedding = nn.Embedding(type_count, d_model)
        self.candidate_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.sequence_query_projection = nn.Sequential(RMSNorm(d_model * 2), nn.Linear(d_model * 2, d_model), nn.SiLU())
        self.blocks = nn.ModuleList(
            [UnifiedInteractionBlock(d_model, num_heads, hidden_mult, dropout_rate) for _ in range(max(1, num_blocks))]
        )
        self.final_norm = RMSNorm(d_model)
        self.classifier = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(d_model, action_num),
        )
        self._fusion_core_compiled = False
        self._compiled_fusion_core = None

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

    def _sequence_tokens(
        self,
        inputs: ModelInput,
        query: torch.Tensor,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        if self.sequence_memory is None:
            return [], []
        pieces, masks, domain_indices = self.sequence_memory(inputs, query)
        typed_pieces: list[torch.Tensor] = []
        for tokens, domain_index in zip(pieces, domain_indices, strict=True):
            if tokens.shape[1] == 0 or not self.symbiosis_use_domain_type:
                typed_pieces.append(tokens)
                continue
            type_ids = self._type_ids(tokens.shape[0], tokens.shape[1], 5 + domain_index, tokens.device)
            typed_pieces.append(tokens + self.type_embedding(type_ids))
        return typed_pieces, masks

    def _encode_tokens(self, inputs: ModelInput) -> tuple[torch.Tensor, torch.Tensor, int, int, int]:
        user_sparse, user_sparse_mask, item_sparse, item_sparse_mask = self._sparse_tokens(inputs)
        user_dense, user_dense_mask, item_dense, item_dense_mask = self._dense_tokens(inputs)

        user_context = masked_mean(torch.cat([user_sparse, user_dense], dim=1), torch.cat([user_sparse_mask, user_dense_mask], dim=1))
        item_summary = masked_mean(torch.cat([item_sparse, item_dense], dim=1), torch.cat([item_sparse_mask, item_dense_mask], dim=1))
        sequence_query = self.sequence_query_projection(torch.cat([user_context, item_summary], dim=-1))
        sequence_pieces, sequence_masks = self._sequence_tokens(inputs, sequence_query)

        pieces: list[torch.Tensor] = []
        masks: list[torch.Tensor] = []
        candidate_count = 0
        if self.symbiosis_use_candidate_token:
            candidate = self.candidate_token.expand(inputs.user_int_feats.shape[0], -1, -1)
            if self.symbiosis_use_item_prior:
                candidate = candidate + item_summary.unsqueeze(1)
            candidate, candidate_mask = self._add_type(candidate, 4)
            pieces.append(candidate)
            masks.append(candidate_mask)
            candidate_count = 1

        for tokens, mask in (
            (user_sparse, user_sparse_mask),
            (user_dense, user_dense_mask),
            *zip(sequence_pieces, sequence_masks, strict=True),
            (item_sparse, item_sparse_mask),
            (item_dense, item_dense_mask),
        ):
            if tokens.shape[1] > 0:
                pieces.append(tokens)
                masks.append(mask)

        tokens = torch.cat(pieces, dim=1)
        padding_mask = torch.cat(masks, dim=1)
        context_start = candidate_count
        item_start = tokens.shape[1] - item_sparse.shape[1] - item_dense.shape[1]
        return tokens, padding_mask, candidate_count, context_start, item_start

    def _run_fusion_core(self, tokens: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            tokens = maybe_gradient_checkpoint(
                block,
                tokens,
                padding_mask,
                enabled=self.gradient_checkpointing,
            )
        return self.final_norm(tokens)

    def _embed(self, inputs: ModelInput) -> torch.Tensor:
        tokens, padding_mask, candidate_count, context_start, item_start = self._encode_tokens(inputs)
        fusion_core = self._compiled_fusion_core if self._compiled_fusion_core is not None else self._run_fusion_core
        tokens = fusion_core(tokens, padding_mask)
        if candidate_count:
            candidate_summary = tokens[:, 0, :]
        else:
            candidate_summary = masked_mean(tokens[:, context_start:item_start, :], padding_mask[:, context_start:item_start])
        context_summary = masked_mean(tokens[:, context_start:item_start, :], padding_mask[:, context_start:item_start])
        item_summary = masked_mean(tokens[:, item_start:, :], padding_mask[:, item_start:])
        if not self.symbiosis_use_item_prior:
            item_summary = torch.zeros_like(item_summary)
        return torch.cat([candidate_summary, context_summary, item_summary], dim=-1)

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
