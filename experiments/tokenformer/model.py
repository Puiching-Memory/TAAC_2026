"""TokenFormer: unified field and sequence token interaction model for PCVR."""

from __future__ import annotations

import math
from typing import NamedTuple

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
    masked_mean,
    maybe_gradient_checkpoint,
    safe_key_padding_mask,
    scaled_dot_product_attention,
    sinusoidal_positions,
)


ROLE_USER = 0
ROLE_SEQUENCE = 1
ROLE_ITEM = 2
ROLE_TARGET = 3
ROLE_SEPARATOR = 4
ROLE_DENSE = 5


def configure_rms_norm_runtime(*, rms_norm_backend: str, rms_norm_block_rows: int) -> None:
    _configure_rms_norm_runtime(
        backend=rms_norm_backend,
        block_rows=rms_norm_block_rows,
    )


class TokenFormerBatch(NamedTuple):
    tokens: torch.Tensor
    padding_mask: torch.Tensor
    role_ids: torch.Tensor
    item_start: int
    target_index: int


class FieldTokenProjector(nn.Module):
    """Projects each sparse field as its own token to avoid early field collapse."""

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
    """Splits dense values into packet tokens that can join the same token stream."""

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
        features = torch.nan_to_num(features.float(), nan=0.0, posinf=0.0, neginf=0.0)
        features = torch.sign(features) * torch.log1p(features.abs().clamp_max(1.0e8))
        if self.pad_size > 0:
            features = torch.cat([features, features.new_zeros(features.shape[0], self.pad_size)], dim=-1)
        packets = features.view(features.shape[0], self.num_tokens, self.chunk_dim)
        return torch.stack(
            [project(packets[:, packet_index, :]) for packet_index, project in enumerate(self.projects)],
            dim=1,
        )


class SwiGLUFeedForward(nn.Module):
    def __init__(self, d_model: int, hidden_mult: int, dropout: float) -> None:
        super().__init__()
        hidden_dim = int(d_model) * int(hidden_mult)
        self.up = nn.Linear(d_model, hidden_dim * 2)
        self.dropout = nn.Dropout(dropout)
        self.down = nn.Linear(hidden_dim, d_model)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        gate, value = self.up(tokens).chunk(2, dim=-1)
        return self.down(self.dropout(F.silu(gate) * value))


def _rotate_half_pairs(values: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    even = values[..., 0::2]
    odd = values[..., 1::2]
    rotated_even = even * cos - odd * sin
    rotated_odd = even * sin + odd * cos
    return torch.stack((rotated_even, rotated_odd), dim=-1).flatten(-2)


class BFTSAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float, *, use_rope: bool, rope_base: float) -> None:
        super().__init__()
        self.num_heads = int(num_heads)
        self.dropout = float(dropout)
        self.use_rope = bool(use_rope)
        self.rope_base = float(rope_base)
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.query_norm = RMSNorm(d_model)
        self.key_norm = RMSNorm(d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, tokens: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        query, key, value = self.qkv(tokens).chunk(3, dim=-1)
        query = self.query_norm(query).to(dtype=value.dtype)
        key = self.key_norm(key).to(dtype=value.dtype)
        if self.use_rope:
            query, key = self._apply_rope(query, key)
        attended = scaled_dot_product_attention(
            query,
            key,
            value,
            num_heads=self.num_heads,
            attn_mask=attention_mask,
            dropout_p=self.dropout,
            training=self.training,
        )
        return self.out(attended)

    def _apply_rope(self, query: torch.Tensor, key: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, token_count, d_model = query.shape
        head_dim = d_model // self.num_heads
        rotary_dim = (head_dim // 2) * 2
        if rotary_dim <= 0:
            return query, key
        query_heads = query.reshape(batch_size, token_count, self.num_heads, head_dim)
        key_heads = key.reshape(batch_size, token_count, self.num_heads, head_dim)
        positions = torch.arange(token_count, device=query.device, dtype=torch.float32)
        frequencies = torch.arange(0, rotary_dim, 2, device=query.device, dtype=torch.float32)
        inv_freq = torch.pow(self.rope_base, -frequencies / max(1, rotary_dim))
        angles = positions[:, None] * inv_freq[None, :]
        cos = angles.cos().to(dtype=query.dtype).view(1, token_count, 1, -1)
        sin = angles.sin().to(dtype=query.dtype).view(1, token_count, 1, -1)

        def rotate(values: torch.Tensor) -> torch.Tensor:
            rotating = values[..., :rotary_dim]
            rotated = _rotate_half_pairs(rotating, cos, sin)
            if rotary_dim == head_dim:
                return rotated
            return torch.cat([rotated, values[..., rotary_dim:]], dim=-1)

        query_heads = rotate(query_heads)
        key_heads = rotate(key_heads)
        return query_heads.reshape(batch_size, token_count, d_model), key_heads.reshape(batch_size, token_count, d_model)


class TokenFormerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, hidden_mult: int, dropout: float, *, use_rope: bool, rope_base: float) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(d_model)
        self.attention = BFTSAttention(d_model, num_heads, dropout, use_rope=use_rope, rope_base=rope_base)
        self.nlir_gate = nn.Linear(d_model, d_model)
        self.ffn_norm = RMSNorm(d_model)
        self.ffn = SwiGLUFeedForward(d_model, hidden_mult, dropout)

    def forward(
        self,
        tokens: torch.Tensor,
        padding_mask: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        valid = (~padding_mask).to(dtype=tokens.dtype).unsqueeze(-1)
        attention_input = self.attn_norm(tokens)
        attention_update = self.attention(attention_input, attention_mask)
        nlir_update = torch.sigmoid(self.nlir_gate(attention_input)) * attention_update
        tokens = tokens + nlir_update * valid
        return tokens + self.ffn(self.ffn_norm(tokens)) * valid


class PCVRTokenFormer(EmbeddingParameterMixin, nn.Module):
    """TokenFormer-style PCVR model with BFTS attention and NLIR gating."""

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
        num_blocks: int = 4,
        num_heads: int = 4,
        seq_encoder_type: str = "transformer",
        hidden_mult: int = 4,
        dropout_rate: float = 0.01,
        seq_top_k: int = 64,
        seq_causal: bool = True,
        action_num: int = 1,
        num_time_buckets: int = 65,
        rank_mixer_mode: str = "full",
        use_rope: bool = True,
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
            rank_mixer_mode,
            seq_id_threshold,
            ns_tokenizer_type,
            user_ns_tokens,
            item_ns_tokens,
        )
        num_heads = choose_num_heads(d_model, num_heads)
        self.d_model = int(d_model)
        self.action_num = int(action_num)
        self.gradient_checkpointing = bool(gradient_checkpointing)
        self.seq_causal = bool(seq_causal)
        self.use_rope = bool(use_rope)
        self.seq_domains = sorted(seq_vocab_sizes)
        self.seq_keep_per_domain = max(1, int(seq_top_k))

        self.user_fields = FieldTokenProjector(user_int_feature_specs, emb_dim, d_model, emb_skip_threshold)
        self.item_fields = FieldTokenProjector(item_int_feature_specs, emb_dim, d_model, emb_skip_threshold)
        self.user_dense = DensePacketTokenizer(user_dense_dim, d_model, max_packets=2)
        self.item_dense = DensePacketTokenizer(item_dense_dim, d_model, max_packets=1)
        self.sequence_tokenizers = nn.ModuleDict(
            {
                domain: SequenceTokenizer(
                    vocab_sizes,
                    emb_dim,
                    d_model,
                    num_time_buckets,
                    emb_skip_threshold,
                )
                for domain, vocab_sizes in seq_vocab_sizes.items()
            }
        )

        self.separator_tokens = nn.Parameter(torch.randn(3, d_model) * 0.02)
        self.target_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.target_project = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.SiLU(),
            nn.LayerNorm(d_model),
        )
        self.blocks = nn.ModuleList(
            [
                TokenFormerBlock(
                    d_model,
                    num_heads,
                    hidden_mult,
                    dropout_rate,
                    use_rope=use_rope,
                    rope_base=rope_base,
                )
                for _block_index in range(max(1, int(num_blocks)))
            ]
        )
        self.final_norm = RMSNorm(d_model)
        self.classifier = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(d_model, action_num),
        )
        self.num_ns = self.user_fields.num_tokens + self.item_fields.num_tokens
        self.num_ns += self.user_dense.num_tokens + self.item_dense.num_tokens + 3

    def _zeros_mask(self, tokens: torch.Tensor) -> torch.Tensor:
        return torch.zeros(tokens.shape[0], tokens.shape[1], dtype=torch.bool, device=tokens.device)

    def _role_ids(self, count: int, role_id: int, device: torch.device) -> torch.Tensor:
        return torch.full((count,), int(role_id), dtype=torch.long, device=device)

    def _separator(self, batch_size: int, index: int, reference: torch.Tensor) -> torch.Tensor:
        token = self.separator_tokens[index].view(1, 1, -1)
        return token.expand(batch_size, -1, -1).to(device=reference.device, dtype=reference.dtype)

    def _cat_or_empty(self, pieces: tuple[torch.Tensor, ...], batch_size: int, device: torch.device) -> torch.Tensor:
        non_empty = [piece for piece in pieces if piece.shape[1] > 0]
        if non_empty:
            return torch.cat(non_empty, dim=1)
        return torch.zeros(batch_size, 0, self.d_model, dtype=torch.float32, device=device)

    def _tail_crop(self, tokens: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        if tokens.shape[1] <= self.seq_keep_per_domain:
            return tokens
        keep_count = self.seq_keep_per_domain
        start = (lengths - keep_count).clamp_min(0)
        offsets = torch.arange(keep_count, device=tokens.device).unsqueeze(0)
        positions = (start.unsqueeze(1) + offsets).clamp_max(tokens.shape[1] - 1)
        return tokens.gather(dim=1, index=positions.unsqueeze(-1).expand(-1, -1, tokens.shape[-1]))

    def _encode_sequence_events(self, inputs: ModelInput) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        pieces: list[torch.Tensor] = []
        masks: list[torch.Tensor] = []
        role_pieces: list[torch.Tensor] = []
        for domain_index, domain in enumerate(self.seq_domains):
            raw_sequence = inputs.seq_data[domain]
            seq_len = inputs.seq_lens[domain].to(raw_sequence.device).clamp_min(0).clamp_max(raw_sequence.shape[2])
            tokens = self.sequence_tokenizers[domain](raw_sequence, inputs.seq_time_buckets.get(domain))
            tokens = self._tail_crop(tokens, seq_len)
            seq_len = seq_len.clamp_max(tokens.shape[1])
            if not self.use_rope:
                tokens = tokens + sinusoidal_positions(tokens.shape[1], self.d_model, tokens.device).unsqueeze(0)
            pieces.append(tokens)
            masks.append(make_padding_mask(seq_len, tokens.shape[1]))
            role_pieces.append(self._role_ids(tokens.shape[1], ROLE_SEQUENCE, tokens.device))
            if domain_index < len(self.seq_domains) - 1:
                separator = self._separator(raw_sequence.shape[0], 1, tokens)
                pieces.append(separator)
                masks.append(self._zeros_mask(separator))
                role_pieces.append(self._role_ids(1, ROLE_SEPARATOR, tokens.device))
        return pieces, masks, role_pieces

    def _append_piece(
        self,
        pieces: list[torch.Tensor],
        masks: list[torch.Tensor],
        role_pieces: list[torch.Tensor],
        tokens: torch.Tensor,
        role_id: int,
    ) -> None:
        if tokens.shape[1] == 0:
            return
        pieces.append(tokens)
        masks.append(self._zeros_mask(tokens))
        role_pieces.append(self._role_ids(tokens.shape[1], role_id, tokens.device))

    def _build_target_token(self, user_tokens: torch.Tensor, item_tokens: torch.Tensor) -> torch.Tensor:
        user_summary = masked_mean(user_tokens, self._zeros_mask(user_tokens))
        item_summary = masked_mean(item_tokens, self._zeros_mask(item_tokens))
        interaction = user_summary * item_summary
        target_context = self.target_project(torch.cat([user_summary, item_summary, interaction], dim=-1)).unsqueeze(1)
        return self.target_token.expand(user_tokens.shape[0], -1, -1) + target_context

    def _encode_tokens(self, inputs: ModelInput) -> TokenFormerBatch:
        user_sparse = self.user_fields(inputs.user_int_feats)
        user_dense = self.user_dense(inputs.user_dense_feats)
        item_sparse = self.item_fields(inputs.item_int_feats)
        item_dense = self.item_dense(inputs.item_dense_feats)
        batch_size = inputs.user_int_feats.shape[0]
        device = inputs.user_int_feats.device
        user_tokens = self._cat_or_empty((user_sparse, user_dense), batch_size, device)
        item_tokens = self._cat_or_empty((item_sparse, item_dense), batch_size, device)

        pieces: list[torch.Tensor] = []
        masks: list[torch.Tensor] = []
        role_pieces: list[torch.Tensor] = []

        self._append_piece(pieces, masks, role_pieces, user_sparse, ROLE_USER)
        self._append_piece(pieces, masks, role_pieces, user_dense, ROLE_DENSE)
        first_separator = self._separator(inputs.user_int_feats.shape[0], 0, user_tokens)
        pieces.append(first_separator)
        masks.append(self._zeros_mask(first_separator))
        role_pieces.append(self._role_ids(1, ROLE_SEPARATOR, first_separator.device))

        sequence_pieces, sequence_masks, sequence_roles = self._encode_sequence_events(inputs)
        pieces.extend(sequence_pieces)
        masks.extend(sequence_masks)
        role_pieces.extend(sequence_roles)

        second_separator = self._separator(inputs.user_int_feats.shape[0], 2, user_tokens)
        pieces.append(second_separator)
        masks.append(self._zeros_mask(second_separator))
        role_pieces.append(self._role_ids(1, ROLE_SEPARATOR, second_separator.device))

        item_start = sum(piece.shape[1] for piece in pieces)
        self._append_piece(pieces, masks, role_pieces, item_sparse, ROLE_ITEM)
        self._append_piece(pieces, masks, role_pieces, item_dense, ROLE_DENSE)
        target = self._build_target_token(user_tokens, item_tokens)
        pieces.append(target)
        masks.append(self._zeros_mask(target))
        role_pieces.append(self._role_ids(1, ROLE_TARGET, target.device))

        tokens = torch.cat(pieces, dim=1)
        padding_mask = torch.cat(masks, dim=1)
        role_ids = torch.cat(role_pieces, dim=0)
        return TokenFormerBatch(
            tokens=tokens,
            padding_mask=padding_mask,
            role_ids=role_ids,
            item_start=item_start,
            target_index=tokens.shape[1] - 1,
        )

    def _bfts_attention_mask(self, padding_mask: torch.Tensor, role_ids: torch.Tensor, layer_index: int) -> torch.Tensor:
        batch_size, token_count = padding_mask.shape
        positions = torch.arange(token_count, device=padding_mask.device)
        key_positions = positions.unsqueeze(0)
        query_positions = positions.unsqueeze(1)
        bottom_full_layers = max(1, len(self.blocks) // 2)
        if layer_index < bottom_full_layers:
            if self.seq_causal:
                structural = key_positions <= query_positions
            else:
                structural = torch.ones(token_count, token_count, dtype=torch.bool, device=padding_mask.device)
        else:
            sliding_index = layer_index - bottom_full_layers
            window = max(4, math.ceil(self.seq_keep_per_domain / (2 ** sliding_index)))
            distance = query_positions - key_positions
            if self.seq_causal:
                structural = (distance >= 0) & (distance <= window)
            else:
                structural = distance.abs() <= window
        target_queries = role_ids == ROLE_TARGET
        separator_keys = role_ids == ROLE_SEPARATOR
        structural = structural | torch.eye(token_count, dtype=torch.bool, device=padding_mask.device)
        structural = torch.where(target_queries.unsqueeze(1), torch.ones_like(structural), structural)
        structural = structural | separator_keys.unsqueeze(0)

        key_valid = ~safe_key_padding_mask(padding_mask)
        mask = structural.unsqueeze(0) & key_valid.unsqueeze(1)
        fallback = torch.eye(token_count, dtype=torch.bool, device=padding_mask.device).unsqueeze(0)
        mask = torch.where(padding_mask.unsqueeze(-1), fallback.expand(batch_size, -1, -1), mask)
        return mask.unsqueeze(1)

    def _embed(self, inputs: ModelInput) -> torch.Tensor:
        batch = self._encode_tokens(inputs)
        tokens = batch.tokens
        for layer_index, block in enumerate(self.blocks):
            attention_mask = self._bfts_attention_mask(batch.padding_mask, batch.role_ids, layer_index)
            tokens = maybe_gradient_checkpoint(
                block,
                tokens,
                batch.padding_mask,
                attention_mask,
                enabled=self.gradient_checkpointing,
            )
        tokens = self.final_norm(tokens)
        target_summary = tokens[:, batch.target_index, :]
        context_summary = masked_mean(tokens[:, : batch.item_start, :], batch.padding_mask[:, : batch.item_start])
        item_summary = masked_mean(
            tokens[:, batch.item_start : batch.target_index, :],
            batch.padding_mask[:, batch.item_start : batch.target_index],
        )
        return torch.cat([target_summary, context_summary, item_summary], dim=-1)

    def forward(self, inputs: ModelInput) -> torch.Tensor:
        return self.classifier(self._embed(inputs))

    def predict(self, inputs: ModelInput) -> tuple[torch.Tensor, torch.Tensor]:
        embeddings = self._embed(inputs)
        return self.classifier(embeddings), embeddings


__all__ = ["BFTSAttention", "ModelInput", "PCVRTokenFormer", "TokenFormerBlock", "configure_rms_norm_runtime"]
