"""UniRec-style unified PCVR model."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from taac2026.api import (
    EmbeddingParameterMixin,
    FeatureEmbeddingBank,
    ModelInput,
    RMSNorm,
    SequenceTokenizer,
    choose_num_heads,
    configure_rms_norm_runtime as _configure_rms_norm_runtime,
    make_padding_mask,
    masked_last,
    masked_mean,
    maybe_gradient_checkpoint,
    safe_key_padding_mask,
    sinusoidal_positions,
)


def configure_rms_norm_runtime(*, rms_norm_backend: str, rms_norm_block_rows: int) -> None:
    _configure_rms_norm_runtime(
        backend=rms_norm_backend,
        block_rows=rms_norm_block_rows,
    )


class FieldTokenProjector(nn.Module):
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
        self.d_model = int(d_model)
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
    def __init__(self, input_dim: int, d_model: int, max_packets: int) -> None:
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
        self.chunk_dim = math.ceil(self.input_dim / self.num_tokens)
        self.pad_size = self.chunk_dim * self.num_tokens - self.input_dim
        self.projects = nn.ModuleList(
            nn.Sequential(
                nn.Linear(self.chunk_dim, d_model),
                nn.SiLU(),
                nn.LayerNorm(d_model),
            )
            for _packet_index in range(self.num_tokens)
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if self.num_tokens <= 0:
            return features.new_zeros(features.shape[0], 0, self.d_model, dtype=torch.float32)
        if self.pad_size > 0:
            features = torch.cat([features, features.new_zeros(features.shape[0], self.pad_size)], dim=-1)
        packets = features.view(features.shape[0], self.num_tokens, self.chunk_dim)
        return torch.stack(
            [project(packets[:, packet_index, :]) for packet_index, project in enumerate(self.projects)],
            dim=1,
        )


class FeatureCrossLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, hidden_mult: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.attention = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * hidden_mult),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * hidden_mult, d_model),
        )

    def forward(self, tokens: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        if tokens.shape[1] == 0:
            return tokens
        attention_input = self.norm(tokens)
        attended, _weights = self.attention(
            attention_input,
            attention_input,
            attention_input,
            key_padding_mask=safe_key_padding_mask(padding_mask),
            need_weights=False,
        )
        tokens = tokens + attended
        return tokens + self.ffn(tokens)


class BranchEncoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, hidden_mult: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.attention = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * hidden_mult),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * hidden_mult, d_model),
        )

    def forward(self, tokens: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        attention_input = self.norm(tokens)
        attended, _weights = self.attention(
            attention_input,
            attention_input,
            attention_input,
            key_padding_mask=safe_key_padding_mask(padding_mask),
            need_weights=False,
        )
        tokens = tokens + attended
        return tokens + self.ffn(tokens)


class SequenceBranchEncoder(nn.Module):
    def __init__(self, d_model: int, num_heads: int, hidden_mult: int, dropout: float, num_layers: int = 1) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            BranchEncoderLayer(d_model, num_heads, hidden_mult, dropout) for _layer_index in range(max(1, num_layers))
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, tokens: torch.Tensor, padding_mask: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        if tokens.shape[1] == 0:
            return tokens.new_zeros(tokens.shape[0], tokens.shape[-1])
        for layer in self.layers:
            tokens = layer(tokens, padding_mask)
        summary = masked_last(self.norm(tokens), lengths)
        return torch.where(lengths.gt(0).unsqueeze(-1), summary, torch.zeros_like(summary))


class MixtureOfTransducers(nn.Module):
    def __init__(self, domains: list[str], d_model: int, num_heads: int, hidden_mult: int, dropout: float) -> None:
        super().__init__()
        self.domains = list(domains)
        self.branches = nn.ModuleDict(
            {
                domain: SequenceBranchEncoder(d_model, num_heads, hidden_mult, dropout)
                for domain in self.domains
            }
        )
        self.gate = nn.Sequential(
            nn.Linear(d_model, max(1, len(self.domains))),
            nn.Softmax(dim=-1),
        )
        self.value_projects = nn.ModuleDict({domain: nn.Linear(d_model, d_model) for domain in self.domains})
        self.output = nn.Sequential(nn.Linear(d_model, d_model), nn.SiLU(), nn.LayerNorm(d_model))
        self.d_model = int(d_model)

    def forward(
        self,
        sequences: dict[str, torch.Tensor],
        masks: dict[str, torch.Tensor],
        lengths: dict[str, torch.Tensor],
        fallback: torch.Tensor,
    ) -> torch.Tensor:
        if not self.domains:
            return fallback.new_zeros(fallback.shape[0], self.d_model)
        branch_outputs = [self.branches[domain](sequences[domain], masks[domain], lengths[domain]) for domain in self.domains]
        gates = self.gate(fallback).unsqueeze(-1)
        values = torch.stack(
            [self.value_projects[domain](branch_outputs[domain_index]) for domain_index, domain in enumerate(self.domains)],
            dim=1,
        )
        return self.output((gates * values).sum(dim=1))


class TargetAwareInterest(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.query = nn.Linear(d_model, d_model, bias=False)
        self.key = nn.Linear(d_model, d_model, bias=False)
        self.value = nn.Linear(d_model, d_model, bias=False)
        self.output = nn.Sequential(nn.Linear(d_model, d_model), nn.SiLU(), nn.LayerNorm(d_model))
        self.scale = float(d_model) ** -0.5

    def forward(self, item_summary: torch.Tensor, sequence_tokens: torch.Tensor, sequence_mask: torch.Tensor) -> torch.Tensor:
        if sequence_tokens.shape[1] == 0:
            return item_summary.new_zeros(item_summary.shape[0], item_summary.shape[-1])
        safe_mask = safe_key_padding_mask(sequence_mask)
        query = self.query(item_summary).unsqueeze(1)
        key = self.key(sequence_tokens)
        value = self.value(sequence_tokens)
        scores = torch.matmul(query, key.transpose(-1, -2)) * self.scale
        scores = scores.masked_fill(safe_mask.unsqueeze(1), torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1)
        output = self.output(torch.matmul(weights, value).squeeze(1))
        has_valid_sequence = (~sequence_mask).any(dim=1).unsqueeze(-1)
        return torch.where(has_valid_sequence, output, torch.zeros_like(output))


class HybridSiLUGatedAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.num_heads = int(num_heads)
        self.head_dim = d_model // self.num_heads
        self.qkuv = nn.Linear(d_model, d_model * 4, bias=False)
        self.output = nn.Linear(d_model, d_model, bias=False)
        self.attention_norm = nn.LayerNorm(self.head_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = float(self.head_dim) ** -0.5

    def forward(self, tokens: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        batch_size, token_count, d_model = tokens.shape
        projected = F.silu(self.qkuv(tokens))
        query, key, value, gate = projected.chunk(4, dim=-1)
        query = query.view(batch_size, token_count, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, token_count, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, token_count, self.num_heads, self.head_dim).transpose(1, 2)
        gate = gate.view(batch_size, token_count, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        mask = attention_mask.to(dtype=scores.dtype)
        scores = F.silu(scores) * mask
        normalizer = mask.sum(dim=-1, keepdim=True).clamp_min(1.0).sqrt()
        attended = torch.matmul(scores / normalizer, value)
        attended = self.attention_norm(attended) * gate
        attended = attended.transpose(1, 2).contiguous().view(batch_size, token_count, d_model)
        return self.dropout(self.output(attended))


class BlockAttentionResidual(nn.Module):
    def __init__(self, d_model: int, num_layers: int, block_size: int = 2) -> None:
        super().__init__()
        self.queries = nn.Parameter(torch.zeros(max(1, num_layers), d_model))
        self.norm = RMSNorm(d_model)
        self.block_size = max(1, int(block_size))
        self.scale = float(d_model) ** -0.5

    def forward(
        self,
        tokens: torch.Tensor,
        layer_index: int,
        block_summaries: list[torch.Tensor],
        partial_summary: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        current_summary = tokens[:, -1, :]
        partial_summary = partial_summary + current_summary
        if block_summaries:
            summaries = self.norm(torch.stack(block_summaries, dim=1))
            query = self.queries[layer_index].to(dtype=summaries.dtype)
            weights = torch.softmax(torch.matmul(summaries, query) * self.scale, dim=-1)
            residual = torch.matmul(weights.unsqueeze(1), summaries).squeeze(1)
            tokens = tokens + residual.unsqueeze(1)
        if (layer_index + 1) % self.block_size == 0:
            block_summaries.append(partial_summary)
            partial_summary = torch.zeros_like(partial_summary)
        return tokens, partial_summary


class UniRecBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, hidden_mult: int, dropout: float) -> None:
        super().__init__()
        self.attention_norm = RMSNorm(d_model)
        self.attention = HybridSiLUGatedAttention(d_model, num_heads, dropout)
        self.ffn_norm = RMSNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * hidden_mult),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * hidden_mult, d_model),
        )

    def forward(self, tokens: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        tokens = tokens + self.attention(self.attention_norm(tokens), attention_mask)
        return tokens + self.ffn(self.ffn_norm(tokens))


class PCVRUniRec(EmbeddingParameterMixin, nn.Module):
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
        del user_ns_groups, item_ns_groups, num_queries, seq_encoder_type, seq_causal, rank_mixer_mode, use_rope, rope_base
        del seq_id_threshold, ns_tokenizer_type, user_ns_tokens, item_ns_tokens
        num_heads = choose_num_heads(d_model, num_heads)
        self.d_model = int(d_model)
        self.action_num = int(action_num)
        self.gradient_checkpointing = bool(gradient_checkpointing)
        self.seq_domains = sorted(seq_vocab_sizes)
        self.seq_keep_per_domain = max(1, int(seq_top_k))

        self.user_fields = FieldTokenProjector(user_int_feature_specs, emb_dim, d_model, emb_skip_threshold)
        self.item_fields = FieldTokenProjector(item_int_feature_specs, emb_dim, d_model, emb_skip_threshold)
        self.user_dense = DensePacketTokenizer(user_dense_dim, d_model, max_packets=2)
        self.item_dense = DensePacketTokenizer(item_dense_dim, d_model, max_packets=1)
        self.sequence_tokenizers = nn.ModuleDict(
            {
                domain: SequenceTokenizer(vocab_sizes, emb_dim, d_model, num_time_buckets, emb_skip_threshold)
                for domain, vocab_sizes in seq_vocab_sizes.items()
            }
        )

        self.feature_token_count = (
            self.user_fields.num_tokens
            + self.user_dense.num_tokens
            + self.item_fields.num_tokens
            + self.item_dense.num_tokens
        )
        self.num_ns = self.feature_token_count + 3
        self.field_embedding = nn.Embedding(max(1, self.feature_token_count), d_model)
        self.source_embedding = nn.Embedding(max(4, 4 + len(self.seq_domains)), d_model)
        self.target_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.mot_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.interest_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        self.feature_cross = FeatureCrossLayer(d_model, num_heads, hidden_mult, dropout_rate)
        self.mot = MixtureOfTransducers(self.seq_domains, d_model, num_heads, hidden_mult, dropout_rate)
        self.interest = TargetAwareInterest(d_model)
        self.target_fusion = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.SiLU(),
            nn.LayerNorm(d_model),
        )
        self.blocks = nn.ModuleList(
            UniRecBlock(d_model, num_heads, hidden_mult, dropout_rate) for _layer_index in range(max(1, num_blocks))
        )
        self.block_attn_res = BlockAttentionResidual(d_model, len(self.blocks), block_size=2)
        self.final_norm = RMSNorm(d_model)
        self.classifier = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(d_model, action_num),
        )

    def _add_source(self, tokens: torch.Tensor, source_id: int) -> torch.Tensor:
        if tokens.shape[1] == 0:
            return tokens
        source_ids = torch.full((tokens.shape[0], tokens.shape[1]), int(source_id), dtype=torch.long, device=tokens.device)
        return tokens + self.source_embedding(source_ids)

    def _feature_tokens(self, inputs: ModelInput) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        user_tokens = self._add_source(self.user_fields(inputs.user_int_feats), 0)
        user_dense = self._add_source(self.user_dense(inputs.user_dense_feats), 1)
        item_tokens = self._add_source(self.item_fields(inputs.item_int_feats), 2)
        item_dense = self._add_source(self.item_dense(inputs.item_dense_feats), 3)
        user_parts = [user_tokens, user_dense]
        item_parts = [item_tokens, item_dense]
        user_feature_tokens = torch.cat(user_parts, dim=1)
        item_feature_tokens = torch.cat(item_parts, dim=1)
        feature_tokens = torch.cat([user_feature_tokens, item_feature_tokens], dim=1)
        feature_mask = torch.zeros(feature_tokens.shape[0], feature_tokens.shape[1], dtype=torch.bool, device=feature_tokens.device)
        if feature_tokens.shape[1] > 0:
            field_ids = torch.arange(feature_tokens.shape[1], device=feature_tokens.device).clamp_max(self.field_embedding.num_embeddings - 1)
            feature_tokens = feature_tokens + self.field_embedding(field_ids).unsqueeze(0)
            feature_tokens = self.feature_cross(feature_tokens, feature_mask)
        return feature_tokens, feature_mask, user_feature_tokens, item_feature_tokens

    def _tail_sequence(self, tokens: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        keep_count = min(tokens.shape[1], self.seq_keep_per_domain)
        if keep_count < tokens.shape[1]:
            start = (lengths - keep_count).clamp_min(0)
            offsets = torch.arange(keep_count, device=tokens.device).unsqueeze(0)
            gather_positions = start.unsqueeze(1) + offsets
            tokens = tokens.gather(dim=1, index=gather_positions.unsqueeze(-1).expand(-1, -1, tokens.shape[-1]))
        lengths = lengths.clamp_max(keep_count)
        return tokens, lengths

    def _sequence_tokens(
        self,
        inputs: ModelInput,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        sequences: dict[str, torch.Tensor] = {}
        masks: dict[str, torch.Tensor] = {}
        lengths: dict[str, torch.Tensor] = {}
        pieces: list[torch.Tensor] = []
        mask_pieces: list[torch.Tensor] = []
        for domain_index, domain in enumerate(self.seq_domains):
            raw_sequence = inputs.seq_data[domain]
            seq_len = inputs.seq_lens[domain].to(raw_sequence.device).clamp_max(raw_sequence.shape[2])
            tokens = self.sequence_tokenizers[domain](raw_sequence, inputs.seq_time_buckets.get(domain))
            tokens, seq_len = self._tail_sequence(tokens, seq_len)
            positions = sinusoidal_positions(tokens.shape[1], self.d_model, tokens.device).unsqueeze(0)
            tokens = tokens + positions
            tokens = self._add_source(tokens, 4 + domain_index)
            mask = make_padding_mask(seq_len, tokens.shape[1])
            tokens = tokens.masked_fill(mask.unsqueeze(-1), 0.0)
            sequences[domain] = tokens
            masks[domain] = mask
            lengths[domain] = seq_len
            pieces.append(tokens)
            mask_pieces.append(mask)
        if pieces:
            return sequences, masks, lengths, torch.cat(pieces, dim=1), torch.cat(mask_pieces, dim=1)
        batch_size = inputs.user_int_feats.shape[0]
        device = inputs.user_int_feats.device
        empty_tokens = torch.zeros(batch_size, 0, self.d_model, dtype=torch.float32, device=device)
        empty_mask = torch.zeros(batch_size, 0, dtype=torch.bool, device=device)
        return sequences, masks, lengths, empty_tokens, empty_mask

    def _build_attention_mask(
        self,
        padding_mask: torch.Tensor,
        feature_token_count: int,
        sequence_token_count: int,
        special_token_count: int,
    ) -> torch.Tensor:
        token_count = padding_mask.shape[1]
        device = padding_mask.device
        row_positions = torch.arange(token_count, device=device).unsqueeze(1)
        col_positions = torch.arange(token_count, device=device).unsqueeze(0)
        base_mask = col_positions <= row_positions
        if feature_token_count > 0:
            base_mask[:feature_token_count, :feature_token_count] = True
        sequence_start = feature_token_count
        sequence_end = feature_token_count + sequence_token_count
        if sequence_token_count > 0:
            query_in_sequence = (row_positions >= sequence_start) & (row_positions < sequence_end)
            key_in_sequence = (col_positions >= sequence_start) & (col_positions < sequence_end)
            local_window = max(1, self.seq_keep_per_domain)
            local_sequence = (col_positions <= row_positions) & ((row_positions - col_positions) < local_window)
            feature_keys = col_positions < feature_token_count
            base_mask = torch.where(query_in_sequence & key_in_sequence, local_sequence, base_mask)
            base_mask = torch.where(query_in_sequence & feature_keys, torch.ones_like(base_mask), base_mask)
        if special_token_count > 0:
            base_mask[-special_token_count:, :] = True
        key_valid = ~padding_mask
        return base_mask.unsqueeze(0).unsqueeze(0) & key_valid.unsqueeze(1).unsqueeze(2)

    def _encode_tokens(self, inputs: ModelInput) -> tuple[torch.Tensor, torch.Tensor, int, int, torch.Tensor, torch.Tensor]:
        feature_tokens, feature_mask, user_tokens, item_tokens = self._feature_tokens(inputs)
        sequences, sequence_masks, sequence_lengths, sequence_tokens, sequence_mask = self._sequence_tokens(inputs)
        user_summary = masked_mean(user_tokens)
        item_summary = masked_mean(item_tokens)
        fallback = (user_summary + item_summary) * 0.5
        mot_summary = self.mot(sequences, sequence_masks, sequence_lengths, fallback)
        interest_summary = self.interest(item_summary, sequence_tokens, sequence_mask)
        target_summary = self.target_fusion(torch.cat([user_summary, item_summary, user_summary * item_summary], dim=-1))

        mot_token = self.mot_token.expand(inputs.user_int_feats.shape[0], -1, -1) + mot_summary.unsqueeze(1)
        interest_token = self.interest_token.expand(inputs.user_int_feats.shape[0], -1, -1) + interest_summary.unsqueeze(1)
        target_token = self.target_token.expand(inputs.user_int_feats.shape[0], -1, -1) + target_summary.unsqueeze(1)
        special_tokens = self._add_source(torch.cat([mot_token, interest_token, target_token], dim=1), 3)
        special_mask = torch.zeros(special_tokens.shape[0], special_tokens.shape[1], dtype=torch.bool, device=special_tokens.device)

        tokens = torch.cat([feature_tokens, sequence_tokens, special_tokens], dim=1)
        padding_mask = torch.cat([feature_mask, sequence_mask, special_mask], dim=1)
        return tokens, padding_mask, feature_tokens.shape[1], sequence_tokens.shape[1], interest_summary, fallback

    def _embed(self, inputs: ModelInput) -> torch.Tensor:
        tokens, padding_mask, feature_token_count, sequence_token_count, interest_summary, fallback = self._encode_tokens(inputs)
        attention_mask = self._build_attention_mask(
            padding_mask,
            feature_token_count=feature_token_count,
            sequence_token_count=sequence_token_count,
            special_token_count=3,
        )
        block_summaries: list[torch.Tensor] = []
        partial_summary = tokens.new_zeros(tokens.shape[0], tokens.shape[-1])
        for layer_index, block in enumerate(self.blocks):
            tokens = maybe_gradient_checkpoint(block, tokens, attention_mask, enabled=self.gradient_checkpointing)
            tokens, partial_summary = self.block_attn_res(tokens, layer_index, block_summaries, partial_summary)
        tokens = self.final_norm(tokens)
        target_summary = tokens[:, -1, :]
        return torch.cat([target_summary, interest_summary, fallback], dim=-1)

    def forward(self, inputs: ModelInput) -> torch.Tensor:
        return self.classifier(self._embed(inputs))

    def predict(self, inputs: ModelInput) -> tuple[torch.Tensor, torch.Tensor]:
        embeddings = self._embed(inputs)
        return self.classifier(embeddings), embeddings