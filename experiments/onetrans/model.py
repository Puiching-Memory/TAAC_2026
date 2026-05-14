"""OneTrans-style unified PCVR model."""

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
    configure_rms_norm_runtime as _configure_rms_norm_runtime,
    maybe_gradient_checkpoint,
    causal_valid_attention_mask,
    choose_num_heads,
    make_padding_mask,
    masked_mean,
    scaled_dot_product_attention,
    sinusoidal_positions,
)


def configure_rms_norm_runtime(*, rms_norm_backend: str, rms_norm_block_rows: int) -> None:
    _configure_rms_norm_runtime(
        backend=rms_norm_backend,
        block_rows=rms_norm_block_rows,
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MixedCausalAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, num_ns_tokens: int, dropout: float) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.dropout = dropout
        self.shared_qkv = nn.Linear(d_model, d_model * 3)
        self.ns_q = nn.ModuleList(nn.Linear(d_model, d_model) for _ in range(num_ns_tokens))
        self.ns_k = nn.ModuleList(nn.Linear(d_model, d_model) for _ in range(num_ns_tokens))
        self.ns_v = nn.ModuleList(nn.Linear(d_model, d_model) for _ in range(num_ns_tokens))
        self.out = nn.Linear(d_model, d_model)

    def forward(self, tokens: torch.Tensor, padding_mask: torch.Tensor, seq_token_count: int) -> torch.Tensor:
        q, k, v = (part.clone() for part in self.shared_qkv(tokens).chunk(3, dim=-1))
        for ns_index, (q_proj, k_proj, v_proj) in enumerate(zip(self.ns_q, self.ns_k, self.ns_v, strict=True)):
            position = seq_token_count + ns_index
            if position >= tokens.shape[1]:
                continue
            token = tokens[:, position, :]
            q[:, position, :] = q_proj(token)
            k[:, position, :] = k_proj(token)
            v[:, position, :] = v_proj(token)
        attn_mask = causal_valid_attention_mask(padding_mask, self.num_heads)
        output = scaled_dot_product_attention(
            q,
            k,
            v,
            num_heads=self.num_heads,
            attn_mask=attn_mask,
            dropout_p=self.dropout,
            training=self.training,
        )
        return self.out(output)


class MixedFeedForward(nn.Module):
    def __init__(self, d_model: int, hidden_mult: int, num_ns_tokens: int, dropout: float) -> None:
        super().__init__()
        self.shared = FeedForward(d_model, hidden_mult, dropout)
        self.ns_specific = nn.ModuleList(FeedForward(d_model, hidden_mult, dropout) for _ in range(num_ns_tokens))

    def forward(self, tokens: torch.Tensor, seq_token_count: int) -> torch.Tensor:
        output = self.shared(tokens)
        for ns_index, ffn in enumerate(self.ns_specific):
            position = seq_token_count + ns_index
            if position >= tokens.shape[1]:
                continue
            output[:, position, :] = ffn(tokens[:, position, :])
        return output


class OneTransBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, hidden_mult: int, num_ns_tokens: int, dropout: float) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(d_model)
        self.attention = MixedCausalAttention(d_model, num_heads, num_ns_tokens, dropout)
        self.ffn_norm = RMSNorm(d_model)
        self.ffn = MixedFeedForward(d_model, hidden_mult, num_ns_tokens, dropout)

    def forward(self, tokens: torch.Tensor, padding_mask: torch.Tensor, seq_token_count: int) -> torch.Tensor:
        tokens = tokens + self.attention(self.attn_norm(tokens), padding_mask, seq_token_count)
        return tokens + self.ffn(self.ffn_norm(tokens), seq_token_count)


class PCVROneTrans(EmbeddingParameterMixin, nn.Module):
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
        del num_queries, seq_encoder_type, seq_causal, rank_mixer_mode, use_rope, rope_base, seq_id_threshold
        num_heads = choose_num_heads(d_model, num_heads)
        self.d_model = d_model
        self.action_num = action_num
        self.gradient_checkpointing = bool(gradient_checkpointing)
        self.seq_domains = sorted(seq_vocab_sizes)
        self.seq_keep_per_domain = max(1, int(seq_top_k))
        self.sequence_stats_dim = 6
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
            compress_high_cardinality=True,
            force_auto_split=force_auto_split,
        )
        self.user_dense = DenseTokenProjector(user_dense_dim, d_model)
        self.item_dense = DenseTokenProjector(item_dense_dim, d_model)
        self.sequence_tokenizers = nn.ModuleDict(
            {
                domain: SequenceTokenizer(
                    vocab_sizes,
                    emb_dim,
                    d_model,
                    num_time_buckets,
                    emb_skip_threshold,
                    compress_high_cardinality=True,
                )
                for domain, vocab_sizes in seq_vocab_sizes.items()
            }
        )
        self.num_ns = self.user_tokenizer.num_tokens + self.item_tokenizer.num_tokens
        self.num_ns += int(user_dense_dim > 0) + int(item_dense_dim > 0)
        self.use_sequence_stats_token = bool(self.seq_domains)
        self.num_ns += int(self.use_sequence_stats_token)
        self.sequence_stats_projector = nn.Sequential(
            nn.Linear(self.sequence_stats_dim, d_model),
            nn.SiLU(),
            nn.LayerNorm(d_model),
        )
        self.separator_tokens = nn.Parameter(torch.randn(max(1, len(self.seq_domains) - 1), d_model) * 0.02)
        self.blocks = nn.ModuleList(
            [OneTransBlock(d_model, num_heads, hidden_mult, self.num_ns, dropout_rate) for _ in range(max(1, num_blocks))]
        )
        self.final_norm = RMSNorm(d_model)
        self.classifier = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(d_model, action_num),
        )

    def _encode_non_sequence(self, inputs: ModelInput) -> torch.Tensor:
        parts = [self.user_tokenizer(inputs.user_int_feats, inputs.user_int_missing_mask)]
        user_dense = self.user_dense(inputs.user_dense_feats, inputs.user_dense_missing_mask)
        if user_dense is not None:
            parts.append(user_dense)
        parts.append(self.item_tokenizer(inputs.item_int_feats, inputs.item_int_missing_mask))
        item_dense = self.item_dense(inputs.item_dense_feats, inputs.item_dense_missing_mask)
        if item_dense is not None:
            parts.append(item_dense)
        stats_token = self._sequence_stats_token(inputs)
        if stats_token is not None:
            parts.append(stats_token)
        return torch.cat(parts, dim=1)

    def _sequence_stats_token(self, inputs: ModelInput) -> torch.Tensor | None:
        if not self.use_sequence_stats_token:
            return None
        batch_size = inputs.user_int_feats.shape[0]
        device = inputs.user_int_feats.device
        dtype = torch.float32
        vectors: list[torch.Tensor] = []
        seq_stats = inputs.seq_stats or {}
        for domain in self.seq_domains:
            stats = seq_stats.get(domain)
            if stats is None:
                continue
            stats = stats.to(dtype=dtype, device=device)
            if stats.shape[-1] < self.sequence_stats_dim:
                pad = stats.new_zeros(stats.shape[0], self.sequence_stats_dim - stats.shape[-1])
                stats = torch.cat([stats, pad], dim=-1)
            vectors.append(self.sequence_stats_projector(stats[:, : self.sequence_stats_dim]))
        if not vectors:
            return inputs.user_int_feats.new_zeros(batch_size, 1, self.d_model, dtype=torch.float32)
        return torch.stack(vectors, dim=1).mean(dim=1, keepdim=True)

    def _tail_crop(self, tokens: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        if tokens.shape[1] <= self.seq_keep_per_domain:
            return tokens
        keep_count = self.seq_keep_per_domain
        start = (lengths - keep_count).clamp_min(0)
        offsets = torch.arange(keep_count, device=tokens.device).unsqueeze(0)
        positions = (start.unsqueeze(1) + offsets).clamp_max(tokens.shape[1] - 1)
        return tokens.gather(dim=1, index=positions.unsqueeze(-1).expand(-1, -1, tokens.shape[-1]))

    def _encode_sequence_stream(self, inputs: ModelInput) -> tuple[torch.Tensor, torch.Tensor]:
        pieces: list[torch.Tensor] = []
        masks: list[torch.Tensor] = []
        sep_index = 0
        for domain_index, domain in enumerate(self.seq_domains):
            raw_sequence = inputs.seq_data[domain]
            seq_len = inputs.seq_lens[domain].to(raw_sequence.device).clamp_min(0).clamp_max(raw_sequence.shape[2])
            tokens = self.sequence_tokenizers[domain](raw_sequence, inputs.seq_time_buckets.get(domain))
            tokens = self._tail_crop(tokens, seq_len)
            seq_len = seq_len.clamp_max(tokens.shape[1])
            tokens = tokens + sinusoidal_positions(tokens.shape[1], self.d_model, tokens.device).unsqueeze(0)
            mask = make_padding_mask(seq_len, tokens.shape[1])
            pieces.append(tokens)
            masks.append(mask)
            if domain_index < len(self.seq_domains) - 1:
                sep = self.separator_tokens[sep_index].view(1, 1, -1).expand(raw_sequence.shape[0], -1, -1)
                pieces.append(sep)
                masks.append(torch.zeros(raw_sequence.shape[0], 1, dtype=torch.bool, device=raw_sequence.device))
                sep_index += 1
        if not pieces:
            batch_size = inputs.user_int_feats.shape[0]
            device = inputs.user_int_feats.device
            return torch.zeros(batch_size, 0, self.d_model, device=device), torch.zeros(batch_size, 0, dtype=torch.bool, device=device)
        return torch.cat(pieces, dim=1), torch.cat(masks, dim=1)

    def _pyramid_keep_count(self, seq_token_count: int, layer_index: int) -> int:
        if seq_token_count <= max(1, self.num_ns):
            return seq_token_count
        remaining_layers = max(1, len(self.blocks) - layer_index)
        target = max(1, self.num_ns)
        decay = (target / seq_token_count) ** (1.0 / remaining_layers)
        return max(target, min(seq_token_count, math.ceil(seq_token_count * decay)))

    def _embed(self, inputs: ModelInput) -> torch.Tensor:
        sequence_tokens, sequence_mask = self._encode_sequence_stream(inputs)
        ns_tokens = self._encode_non_sequence(inputs)
        tokens = torch.cat([sequence_tokens, ns_tokens], dim=1)
        ns_mask = torch.zeros(ns_tokens.shape[0], ns_tokens.shape[1], dtype=torch.bool, device=ns_tokens.device)
        padding_mask = torch.cat([sequence_mask, ns_mask], dim=1)
        seq_token_count = sequence_tokens.shape[1]
        for layer_index, block in enumerate(self.blocks):
            tokens = maybe_gradient_checkpoint(
                block,
                tokens,
                padding_mask,
                seq_token_count,
                enabled=self.gradient_checkpointing,
            )
            keep_count = self._pyramid_keep_count(seq_token_count, layer_index)
            if keep_count < seq_token_count:
                sequence_part = tokens[:, :seq_token_count, :]
                ns_part = tokens[:, seq_token_count:, :]
                sequence_mask = padding_mask[:, :seq_token_count]
                ns_mask = padding_mask[:, seq_token_count:]
                tokens = torch.cat([sequence_part[:, -keep_count:, :], ns_part], dim=1)
                padding_mask = torch.cat([sequence_mask[:, -keep_count:], ns_mask], dim=1)
                seq_token_count = keep_count
        tokens = self.final_norm(tokens)
        seq_summary = masked_mean(tokens[:, :seq_token_count, :], padding_mask[:, :seq_token_count])
        ns_summary = masked_mean(tokens[:, seq_token_count:, :], padding_mask[:, seq_token_count:])
        return torch.cat([seq_summary, ns_summary], dim=-1)

    def forward(self, inputs: ModelInput) -> torch.Tensor:
        return self.classifier(self._embed(inputs))

    def predict(self, inputs: ModelInput) -> tuple[torch.Tensor, torch.Tensor]:
        embeddings = self._embed(inputs)
        return self.classifier(embeddings), embeddings