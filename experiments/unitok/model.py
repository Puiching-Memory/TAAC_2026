"""UniTok: native unified token-stream PCVR model."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from taac2026.api import (
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
    scaled_dot_product_attention,
    sinusoidal_positions,
)


def configure_rms_norm_runtime(*, rms_norm_backend: str, rms_norm_block_rows: int) -> None:
    _configure_rms_norm_runtime(
        backend=rms_norm_backend,
        block_rows=rms_norm_block_rows,
    )


class FieldTokenProjector(nn.Module):
    """Projects per-field sparse embeddings without early field grouping."""

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
        self.project = nn.Sequential(
            nn.Linear(emb_dim, d_model),
            nn.SiLU(),
            nn.LayerNorm(d_model),
        )
        self.d_model = d_model

    @property
    def num_tokens(self) -> int:
        return self.feature_count

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        field_tokens = self.bank(features)
        if field_tokens.shape[1] == 0:
            return features.new_zeros(features.shape[0], 0, self.d_model, dtype=torch.float32)
        return self.project(field_tokens)


class DensePacketTokenizer(nn.Module):
    """Splits dense vectors into packet tokens so the backbone can route them."""

    def __init__(self, input_dim: int, d_model: int, max_packets: int = 2) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.d_model = int(d_model)
        self.num_tokens = 0
        self.chunk_dim = 0
        self._pad_size = 0
        if input_dim <= 0:
            self.projects = nn.ModuleList()
            return
        self.num_tokens = max(1, min(int(max_packets), int(input_dim)))
        self.chunk_dim = math.ceil(int(input_dim) / self.num_tokens)
        self._pad_size = self.chunk_dim * self.num_tokens - int(input_dim)
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
        if self._pad_size > 0:
            pad = features.new_zeros(features.shape[0], self._pad_size)
            features = torch.cat([features, pad], dim=-1)
        chunks = features.view(features.shape[0], self.num_tokens, self.chunk_dim)
        return torch.stack(
            [project(chunks[:, token_index, :]) for token_index, project in enumerate(self.projects)],
            dim=1,
        )


class UnifiedSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.dropout = dropout
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, tokens: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        q, k, v = self.qkv(tokens).chunk(3, dim=-1)
        attn_mask = (~padding_mask).view(padding_mask.shape[0], 1, 1, padding_mask.shape[1])
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


class UniTokBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, hidden_mult: int, dropout: float) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(d_model)
        self.attention = UnifiedSelfAttention(d_model, num_heads, dropout)
        self.ffn_norm = RMSNorm(d_model)
        self.ffn = FeedForward(d_model, hidden_mult, dropout)

    def forward(self, tokens: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        tokens = tokens + self.attention(self.attn_norm(tokens), padding_mask)
        return tokens + self.ffn(self.ffn_norm(tokens))


class PCVRUniTok(EmbeddingParameterMixin, nn.Module):
    """Unified field, dense-packet, sequence-event, and candidate token model."""

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
        del (
            user_ns_groups,
            item_ns_groups,
            num_queries,
            seq_encoder_type,
            seq_causal,
            rank_mixer_mode,
            use_rope,
            rope_base,
            seq_id_threshold,
            ns_tokenizer_type,
            user_ns_tokens,
            item_ns_tokens,
        )
        num_heads = choose_num_heads(d_model, num_heads)
        self.d_model = d_model
        self.action_num = action_num
        self.gradient_checkpointing = bool(gradient_checkpointing)
        self.seq_domains = sorted(seq_vocab_sizes)
        self.seq_keep_per_domain = max(1, int(seq_top_k))

        self.user_fields = FieldTokenProjector(
            user_int_feature_specs,
            emb_dim,
            d_model,
            emb_skip_threshold,
        )
        self.item_fields = FieldTokenProjector(
            item_int_feature_specs,
            emb_dim,
            d_model,
            emb_skip_threshold,
        )
        self.user_dense = DensePacketTokenizer(user_dense_dim, d_model, max_packets=2)
        self.item_dense = DensePacketTokenizer(item_dense_dim, d_model, max_packets=1)
        self.sequence_tokenizers = nn.ModuleDict(
            {
                domain: SequenceTokenizer(vocab_sizes, emb_dim, d_model, num_time_buckets, emb_skip_threshold)
                for domain, vocab_sizes in seq_vocab_sizes.items()
            }
        )

        token_type_count = 4 + len(self.seq_domains)
        self.type_embedding = nn.Embedding(token_type_count, d_model)
        self.candidate_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.blocks = nn.ModuleList(
            [UniTokBlock(d_model, num_heads, hidden_mult, dropout_rate) for _ in range(max(1, num_blocks))]
        )
        self.final_norm = RMSNorm(d_model)
        self.classifier = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(d_model, action_num),
        )
        self.num_ns = self.user_fields.num_tokens + self.item_fields.num_tokens
        self.num_ns += self.user_dense.num_tokens + self.item_dense.num_tokens + 1

    def _type_ids(self, batch_size: int, count: int, type_id: int, device: torch.device) -> torch.Tensor:
        return torch.full((batch_size, count), int(type_id), dtype=torch.long, device=device)

    def _add_type(self, tokens: torch.Tensor, type_id: int) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, token_count, _dim = tokens.shape
        if token_count == 0:
            mask = torch.ones(batch_size, 0, dtype=torch.bool, device=tokens.device)
            return tokens, mask
        type_ids = self._type_ids(batch_size, token_count, type_id, tokens.device)
        mask = torch.zeros(batch_size, token_count, dtype=torch.bool, device=tokens.device)
        return tokens + self.type_embedding(type_ids), mask

    def _encode_dense_packets(self, inputs: ModelInput) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        pieces: list[torch.Tensor] = []
        masks: list[torch.Tensor] = []
        user_dense, user_mask = self._add_type(self.user_dense(inputs.user_dense_feats), 1)
        if user_dense.shape[1] > 0:
            pieces.append(user_dense)
            masks.append(user_mask)
        item_dense, item_mask = self._add_type(self.item_dense(inputs.item_dense_feats), 3)
        if item_dense.shape[1] > 0:
            pieces.append(item_dense)
            masks.append(item_mask)
        return pieces, masks

    def _encode_sequence_events(self, inputs: ModelInput) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        pieces: list[torch.Tensor] = []
        masks: list[torch.Tensor] = []
        for domain_index, domain in enumerate(self.seq_domains):
            raw_sequence = inputs.seq_data[domain]
            seq_len = inputs.seq_lens[domain].to(raw_sequence.device).clamp_max(raw_sequence.shape[2])
            tokens = self.sequence_tokenizers[domain](raw_sequence, inputs.seq_time_buckets.get(domain))
            keep_count = min(tokens.shape[1], self.seq_keep_per_domain)
            if keep_count < tokens.shape[1]:
                start = (seq_len - keep_count).clamp_min(0)
                offsets = torch.arange(keep_count, device=tokens.device).unsqueeze(0)
                gather_positions = start.unsqueeze(1) + offsets
                tokens = tokens.gather(
                    dim=1,
                    index=gather_positions.unsqueeze(-1).expand(-1, -1, tokens.shape[-1]),
                )
            seq_len = seq_len.clamp_max(keep_count)
            positions = sinusoidal_positions(tokens.shape[1], self.d_model, tokens.device).unsqueeze(0)
            type_ids = self._type_ids(
                tokens.shape[0],
                tokens.shape[1],
                4 + domain_index,
                tokens.device,
            )
            tokens = tokens + positions + self.type_embedding(type_ids)
            mask = make_padding_mask(seq_len, tokens.shape[1])
            pieces.append(tokens)
            masks.append(mask)
        return pieces, masks

    def _encode_tokens(self, inputs: ModelInput) -> tuple[torch.Tensor, torch.Tensor, int]:
        user_sparse, user_sparse_mask = self._add_type(self.user_fields(inputs.user_int_feats), 0)
        item_sparse, item_sparse_mask = self._add_type(self.item_fields(inputs.item_int_feats), 2)
        dense_pieces, dense_masks = self._encode_dense_packets(inputs)
        sequence_pieces, sequence_masks = self._encode_sequence_events(inputs)

        candidate = self.candidate_token.expand(inputs.user_int_feats.shape[0], -1, -1)
        candidate = candidate + masked_mean(item_sparse, item_sparse_mask).unsqueeze(1)
        candidate = candidate + self.type_embedding(
            self._type_ids(candidate.shape[0], candidate.shape[1], 2, candidate.device)
        )
        candidate_mask = torch.zeros(candidate.shape[0], 1, dtype=torch.bool, device=candidate.device)

        pieces = [candidate, user_sparse, *dense_pieces, *sequence_pieces, item_sparse]
        masks = [candidate_mask, user_sparse_mask, *dense_masks, *sequence_masks, item_sparse_mask]
        tokens = torch.cat(pieces, dim=1)
        padding_mask = torch.cat(masks, dim=1)
        item_start = tokens.shape[1] - item_sparse.shape[1]
        return tokens, padding_mask, item_start

    def _embed(self, inputs: ModelInput) -> torch.Tensor:
        tokens, padding_mask, item_start = self._encode_tokens(inputs)
        for block in self.blocks:
            tokens = maybe_gradient_checkpoint(
                block,
                tokens,
                padding_mask,
                enabled=self.gradient_checkpointing,
            )
        tokens = self.final_norm(tokens)
        candidate_summary = tokens[:, 0, :]
        context_summary = masked_mean(tokens[:, 1:item_start, :], padding_mask[:, 1:item_start])
        item_summary = masked_mean(tokens[:, item_start:, :], padding_mask[:, item_start:])
        return torch.cat([candidate_summary, context_summary, item_summary], dim=-1)

    def forward(self, inputs: ModelInput) -> torch.Tensor:
        return self.classifier(self._embed(inputs))

    def predict(self, inputs: ModelInput) -> tuple[torch.Tensor, torch.Tensor]:
        embeddings = self._embed(inputs)
        return self.classifier(embeddings), embeddings
