"""Baseline+ PCVR model built from shared TAAC modeling primitives."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from taac2026.api import (
    DenseTokenProjector,
    EmbeddingParameterMixin,
    ModelInput,
    NonSequentialTokenizer,
    RMSNorm,
    SequenceTokenizer,
    choose_num_heads,
    configure_flash_attention_runtime as _configure_flash_attention_runtime,
    configure_rms_norm_runtime as _configure_rms_norm_runtime,
    make_padding_mask,
    masked_last,
    masked_mean,
    maybe_gradient_checkpoint,
    safe_key_padding_mask,
    scaled_dot_product_attention,
    sinusoidal_positions,
)


def configure_flash_attention_runtime(*, flash_attention_backend: str) -> None:
    _configure_flash_attention_runtime(backend=flash_attention_backend)


def configure_rms_norm_runtime(*, rms_norm_backend: str, rms_norm_block_rows: int) -> None:
    _configure_rms_norm_runtime(
        backend=rms_norm_backend,
        block_rows=rms_norm_block_rows,
    )


class FeedForward(nn.Module):
    def __init__(self, d_model: int, hidden_mult: int, dropout: float) -> None:
        super().__init__()
        hidden_dim = max(d_model, d_model * hidden_mult)
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.net(tokens)


class GatedScaledAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float,
        *,
        use_rope: bool = False,
        rope_base: float = 10000.0,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.dropout = dropout
        self.use_rope = use_rope and self.head_dim >= 2
        self.rope_base = rope_base
        self.rotary_dim = self.head_dim - self.head_dim % 2
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.gate = nn.Linear(d_model, d_model)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, 1.0)

    def _apply_rope(self, tokens: torch.Tensor) -> torch.Tensor:
        if not self.use_rope or self.rotary_dim < 2 or tokens.shape[1] == 0:
            return tokens
        batch_size, token_count, _d_model = tokens.shape
        heads = tokens.view(batch_size, token_count, self.num_heads, self.head_dim).transpose(1, 2)
        rotary = heads[..., : self.rotary_dim]
        positions = torch.arange(token_count, dtype=torch.float32, device=tokens.device).unsqueeze(1)
        frequencies = torch.exp(
            torch.arange(0, self.rotary_dim, 2, dtype=torch.float32, device=tokens.device)
            * (-math.log(self.rope_base) / self.rotary_dim)
        )
        angles = positions * frequencies.unsqueeze(0)
        sin = angles.sin().view(1, 1, token_count, -1).to(dtype=tokens.dtype)
        cos = angles.cos().view(1, 1, token_count, -1).to(dtype=tokens.dtype)
        even = rotary[..., 0::2]
        odd = rotary[..., 1::2]
        rotated = torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2)
        heads = torch.cat([rotated, heads[..., self.rotary_dim :]], dim=-1)
        return heads.transpose(1, 2).contiguous().view(batch_size, token_count, self.d_model)

    def _attention_mask(self, key_padding_mask: torch.Tensor | None, query_len: int) -> torch.Tensor | None:
        if key_padding_mask is None:
            return None
        safe_mask = safe_key_padding_mask(key_padding_mask)
        return (~safe_mask).view(safe_mask.shape[0], 1, 1, safe_mask.shape[1]).expand(
            safe_mask.shape[0],
            self.num_heads,
            query_len,
            safe_mask.shape[1],
        )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if query.shape[1] == 0 or key.shape[1] == 0:
            return query.new_zeros(query.shape)
        projected_query = self._apply_rope(self.q_proj(query))
        projected_key = self._apply_rope(self.k_proj(key))
        projected_value = self.v_proj(value)
        output = scaled_dot_product_attention(
            projected_query,
            projected_key,
            projected_value,
            num_heads=self.num_heads,
            attn_mask=self._attention_mask(key_padding_mask, query.shape[1]),
            dropout_p=self.dropout,
            training=self.training,
        )
        if key_padding_mask is not None:
            has_valid_keys = (~key_padding_mask).any(dim=1).to(output.dtype).view(-1, 1, 1)
            output = output * has_valid_keys
        return self.out_proj(output * torch.sigmoid(self.gate(query)))


class SequenceQueryGenerator(nn.Module):
    def __init__(self, d_model: int, num_domains: int, num_queries: int, hidden_mult: int, dropout: float) -> None:
        super().__init__()
        self.num_domains = num_domains
        self.num_queries = max(1, int(num_queries))
        domain_count = max(1, num_domains)
        hidden_dim = max(d_model, d_model * hidden_mult)
        self.query_seed = nn.Parameter(torch.randn(domain_count, self.num_queries, d_model) * 0.02)
        self.domain_embedding = nn.Embedding(domain_count, d_model)
        self.context_projection = nn.Sequential(
            RMSNorm(d_model * 2),
            nn.Linear(d_model * 2, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
        )
        self.query_norm = RMSNorm(d_model)

    def forward(
        self,
        ns_tokens: torch.Tensor,
        sequences: list[torch.Tensor],
        masks: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        if not sequences:
            return []
        batch_size = ns_tokens.shape[0]
        ns_context = masked_mean(ns_tokens)
        queries: list[torch.Tensor] = []
        for domain_index, (tokens, mask) in enumerate(zip(sequences, masks, strict=True)):
            seq_context = masked_mean(tokens, mask)
            context = self.context_projection(torch.cat([ns_context, seq_context], dim=-1))
            domain_ids = torch.full((batch_size,), domain_index, dtype=torch.long, device=tokens.device)
            seed = self.query_seed[domain_index].unsqueeze(0).expand(batch_size, -1, -1)
            query = seed + context.unsqueeze(1) + self.domain_embedding(domain_ids).unsqueeze(1)
            queries.append(self.query_norm(query))
        return queries


class BaselinePlusBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        hidden_mult: int,
        dropout: float,
        *,
        use_rope: bool,
        rope_base: float,
    ) -> None:
        super().__init__()
        self.query_norm = RMSNorm(d_model)
        self.sequence_norm = RMSNorm(d_model)
        self.query_attention = GatedScaledAttention(
            d_model,
            num_heads,
            dropout,
            use_rope=use_rope,
            rope_base=rope_base,
        )
        self.query_ffn_norm = RMSNorm(d_model)
        self.query_ffn = FeedForward(d_model, hidden_mult, dropout)

        self.ns_norm = RMSNorm(d_model)
        self.compact_norm = RMSNorm(d_model)
        self.ns_attention = GatedScaledAttention(d_model, num_heads, dropout)
        self.ns_gate = nn.Linear(d_model * 2, d_model)
        self.ns_ffn_norm = RMSNorm(d_model)
        self.ns_ffn = FeedForward(d_model, hidden_mult, dropout)

        self.sequence_query_norm = RMSNorm(d_model)
        self.sequence_memory_norm = RMSNorm(d_model)
        self.sequence_attention = GatedScaledAttention(d_model, num_heads, dropout)
        self.sequence_gate = nn.Linear(d_model * 2, d_model)
        self.sequence_ffn_norm = RMSNorm(d_model)
        self.sequence_ffn = FeedForward(d_model, hidden_mult, dropout)

    def _update_queries(
        self,
        query_tokens: list[torch.Tensor],
        sequence_tokens: list[torch.Tensor],
        sequence_masks: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        next_queries: list[torch.Tensor] = []
        for query, sequence, mask in zip(query_tokens, sequence_tokens, sequence_masks, strict=True):
            attended = self.query_attention(
                self.query_norm(query),
                self.sequence_norm(sequence),
                self.sequence_norm(sequence),
                key_padding_mask=mask,
            )
            query = query + attended
            next_queries.append(query + self.query_ffn(self.query_ffn_norm(query)))
        return next_queries

    def _update_ns(self, ns_tokens: torch.Tensor, query_tokens: list[torch.Tensor]) -> torch.Tensor:
        if ns_tokens.shape[1] == 0:
            return ns_tokens
        compact_tokens = torch.cat([ns_tokens, *query_tokens], dim=1) if query_tokens else ns_tokens
        compact_tokens = self.compact_norm(compact_tokens)
        attended = self.ns_attention(self.ns_norm(ns_tokens), compact_tokens, compact_tokens)
        gate = torch.sigmoid(self.ns_gate(torch.cat([ns_tokens, attended], dim=-1)))
        ns_tokens = ns_tokens + gate * attended
        return ns_tokens + self.ns_ffn(self.ns_ffn_norm(ns_tokens))

    def _update_sequences(
        self,
        ns_tokens: torch.Tensor,
        query_tokens: list[torch.Tensor],
        sequence_tokens: list[torch.Tensor],
        sequence_masks: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        ns_context = masked_mean(ns_tokens).unsqueeze(1)
        next_sequences: list[torch.Tensor] = []
        for query, sequence, mask in zip(query_tokens, sequence_tokens, sequence_masks, strict=True):
            if sequence.shape[1] == 0:
                next_sequences.append(sequence)
                continue
            memory = self.sequence_memory_norm(torch.cat([ns_context, query], dim=1))
            attended = self.sequence_attention(self.sequence_query_norm(sequence), memory, memory)
            valid_positions = (~mask).to(sequence.dtype).unsqueeze(-1)
            gate = torch.sigmoid(self.sequence_gate(torch.cat([sequence, attended], dim=-1)))
            sequence = sequence + gate * attended * valid_positions
            sequence = sequence + self.sequence_ffn(self.sequence_ffn_norm(sequence)) * valid_positions
            next_sequences.append(sequence)
        return next_sequences

    def forward(
        self,
        query_tokens: list[torch.Tensor],
        ns_tokens: torch.Tensor,
        sequence_tokens: list[torch.Tensor],
        sequence_masks: list[torch.Tensor],
    ) -> tuple[list[torch.Tensor], torch.Tensor, list[torch.Tensor]]:
        query_tokens = self._update_queries(query_tokens, sequence_tokens, sequence_masks)
        ns_tokens = self._update_ns(ns_tokens, query_tokens)
        sequence_tokens = self._update_sequences(ns_tokens, query_tokens, sequence_tokens, sequence_masks)
        return query_tokens, ns_tokens, sequence_tokens


class PCVRBaselinePlus(EmbeddingParameterMixin, nn.Module):
    """Compact HyFormer-style Baseline+ using shared accelerator-aware layers."""

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
    ) -> None:
        super().__init__()
        del seq_encoder_type, seq_causal, rank_mixer_mode, seq_id_threshold
        if ns_tokenizer_type not in {"group", "rankmixer"}:
            raise ValueError(f"unknown ns_tokenizer_type: {ns_tokenizer_type}")
        num_heads = choose_num_heads(d_model, num_heads)
        self.d_model = d_model
        self.action_num = action_num
        self.num_queries = max(1, int(num_queries))
        self.num_time_buckets = int(num_time_buckets)
        self.gradient_checkpointing = bool(gradient_checkpointing)
        self.seq_domains = sorted(seq_vocab_sizes)
        self.num_sequences = len(self.seq_domains)
        self.seq_keep_per_domain = max(1, int(seq_top_k))

        force_auto_split = ns_tokenizer_type == "rankmixer"
        self.user_tokenizer = NonSequentialTokenizer(
            user_int_feature_specs,
            user_ns_groups,
            emb_dim,
            d_model,
            user_ns_tokens,
            emb_skip_threshold,
            force_auto_split=force_auto_split,
        )
        self.item_tokenizer = NonSequentialTokenizer(
            item_int_feature_specs,
            item_ns_groups,
            emb_dim,
            d_model,
            item_ns_tokens,
            emb_skip_threshold,
            force_auto_split=force_auto_split,
        )
        self.user_dense = DenseTokenProjector(user_dense_dim, d_model)
        self.item_dense = DenseTokenProjector(item_dense_dim, d_model)
        self.sequence_tokenizers = nn.ModuleDict(
            {
                domain: SequenceTokenizer(vocab_sizes, emb_dim, d_model, num_time_buckets, emb_skip_threshold)
                for domain, vocab_sizes in seq_vocab_sizes.items()
            }
        )
        self.num_ns = self.user_tokenizer.num_tokens + self.item_tokenizer.num_tokens
        self.num_ns += int(user_dense_dim > 0) + int(item_dense_dim > 0)

        self.query_generator = SequenceQueryGenerator(
            d_model,
            len(self.seq_domains),
            self.num_queries,
            hidden_mult,
            dropout_rate,
        )
        self.blocks = nn.ModuleList(
            [
                BaselinePlusBlock(
                    d_model,
                    num_heads,
                    hidden_mult,
                    dropout_rate,
                    use_rope=use_rope,
                    rope_base=rope_base,
                )
                for _ in range(max(1, num_blocks))
            ]
        )
        hidden_dim = max(d_model, d_model * hidden_mult)
        self.fusion = nn.Sequential(
            RMSNorm(d_model * 4),
            nn.Linear(d_model * 4, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, d_model),
        )
        self.fusion_gate = nn.Sequential(nn.Linear(d_model * 4, d_model), nn.Sigmoid())
        self.out_norm = RMSNorm(d_model)
        self.classifier = nn.Sequential(
            RMSNorm(d_model),
            nn.Linear(d_model, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, action_num),
        )

    def _empty_tokens(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, 0, self.d_model, device=device)

    def _encode_non_sequence(self, inputs: ModelInput) -> torch.Tensor:
        parts: list[torch.Tensor] = []
        user_tokens = self.user_tokenizer(inputs.user_int_feats)
        if user_tokens.shape[1] > 0:
            parts.append(user_tokens)
        user_dense = self.user_dense(inputs.user_dense_feats)
        if user_dense is not None:
            parts.append(user_dense)
        item_tokens = self.item_tokenizer(inputs.item_int_feats)
        if item_tokens.shape[1] > 0:
            parts.append(item_tokens)
        item_dense = self.item_dense(inputs.item_dense_feats)
        if item_dense is not None:
            parts.append(item_dense)
        if parts:
            return torch.cat(parts, dim=1)
        return self._empty_tokens(inputs.user_int_feats.shape[0], inputs.user_int_feats.device)

    def _tail_crop(self, tokens: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        if tokens.shape[1] <= self.seq_keep_per_domain:
            return tokens
        keep_count = self.seq_keep_per_domain
        start = (lengths - keep_count).clamp_min(0)
        offsets = torch.arange(keep_count, device=tokens.device).unsqueeze(0)
        positions = (start.unsqueeze(1) + offsets).clamp_max(tokens.shape[1] - 1)
        return tokens.gather(dim=1, index=positions.unsqueeze(-1).expand(-1, -1, tokens.shape[-1]))

    def _encode_sequences(self, inputs: ModelInput) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        sequences: list[torch.Tensor] = []
        masks: list[torch.Tensor] = []
        lengths: list[torch.Tensor] = []
        for domain in self.seq_domains:
            raw_sequence = inputs.seq_data[domain]
            seq_len = inputs.seq_lens[domain].to(raw_sequence.device).clamp_min(0).clamp_max(raw_sequence.shape[2])
            tokens = self.sequence_tokenizers[domain](raw_sequence, inputs.seq_time_buckets.get(domain))
            tokens = self._tail_crop(tokens, seq_len)
            seq_len = seq_len.clamp_max(tokens.shape[1])
            positions = sinusoidal_positions(tokens.shape[1], self.d_model, tokens.device).to(dtype=tokens.dtype)
            tokens = tokens + positions.unsqueeze(0)
            sequences.append(tokens)
            masks.append(make_padding_mask(seq_len, tokens.shape[1]))
            lengths.append(seq_len)
        return sequences, masks, lengths

    def _average(self, vectors: list[torch.Tensor], fallback: torch.Tensor) -> torch.Tensor:
        if not vectors:
            return fallback
        return torch.stack(vectors, dim=1).mean(dim=1)

    def _embed(self, inputs: ModelInput) -> torch.Tensor:
        ns_tokens = self._encode_non_sequence(inputs)
        sequences, masks, lengths = self._encode_sequences(inputs)
        query_tokens = self.query_generator(ns_tokens, sequences, masks)
        for block in self.blocks:
            query_tokens, ns_tokens, sequences = maybe_gradient_checkpoint(
                block,
                query_tokens,
                ns_tokens,
                sequences,
                masks,
                enabled=self.gradient_checkpointing,
            )

        ns_context = masked_mean(ns_tokens)
        zero_context = torch.zeros_like(ns_context)
        query_context = self._average([masked_mean(tokens) for tokens in query_tokens], zero_context)
        sequence_context = self._average(
            [masked_mean(tokens, mask) for tokens, mask in zip(sequences, masks, strict=True)],
            zero_context,
        )
        recent_context = self._average(
            [
                masked_last(tokens, seq_len) * (seq_len > 0).to(tokens.dtype).unsqueeze(-1)
                for tokens, seq_len in zip(sequences, lengths, strict=True)
            ],
            zero_context,
        )
        joined = torch.cat([query_context, ns_context, sequence_context, recent_context], dim=-1)
        candidate = self.fusion(joined)
        residual = (query_context + ns_context + sequence_context + recent_context) * 0.25
        gate = self.fusion_gate(joined)
        return self.out_norm(gate * candidate + (1.0 - gate) * residual)

    def forward(self, inputs: ModelInput) -> torch.Tensor:
        return self.classifier(self._embed(inputs))

    def predict(self, inputs: ModelInput) -> tuple[torch.Tensor, torch.Tensor]:
        embeddings = self._embed(inputs)
        return self.classifier(embeddings), embeddings


__all__ = [
    "ModelInput",
    "PCVRBaselinePlus",
    "configure_flash_attention_runtime",
    "configure_rms_norm_runtime",
]
