"""InterFormer-style PCVR model.

The implementation keeps the paper's three-way split: Interaction Arch for
non-sequential features, Sequence Arch with context-conditioned feed-forward
updates, and Cross Arch summaries for bidirectional exchange.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from taac2026.api import (
    DenseTokenProjector,
    EmbeddingParameterMixin,
    ModelInput,
    NonSequentialTokenizer,
    SequenceTokenizer,
    configure_rms_norm_runtime as _configure_rms_norm_runtime,
    maybe_gradient_checkpoint,
    choose_num_heads,
    make_padding_mask,
    masked_last,
    masked_mean,
    safe_key_padding_mask,
    sinusoidal_positions,
)


def configure_rms_norm_runtime(*, rms_norm_backend: str, rms_norm_block_rows: int) -> None:
    _configure_rms_norm_runtime(
        backend=rms_norm_backend,
        block_rows=rms_norm_block_rows,
    )


class PersonalizedFeedForward(nn.Module):
    def __init__(self, d_model: int, hidden_mult: int, dropout: float) -> None:
        super().__init__()
        hidden_dim = d_model * hidden_mult
        self.gate = nn.Linear(d_model, d_model)
        self.bias = nn.Linear(d_model, d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
        )

    def forward(self, sequence: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.gate(context)).unsqueeze(1)
        bias = self.bias(context).unsqueeze(1)
        return self.ffn(sequence * gate + bias)


class CrossSummary(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.summary_queries = nn.Parameter(torch.randn(2, d_model) * 0.02)
        self.attention = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.gate = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        sequences: list[torch.Tensor],
        masks: list[torch.Tensor],
        lengths: list[torch.Tensor],
    ) -> torch.Tensor:
        if not sequences:
            raise ValueError("InterFormer requires at least one sequence domain")
        all_tokens = torch.cat(sequences, dim=1)
        all_masks = safe_key_padding_mask(torch.cat(masks, dim=1))
        query = self.summary_queries.unsqueeze(0).expand(all_tokens.shape[0], -1, -1)
        attended, _weights = self.attention(query, all_tokens, all_tokens, key_padding_mask=all_masks, need_weights=False)
        cls_summary = attended[:, 0]
        pma_summary = attended[:, 1]
        recent = torch.stack([masked_last(tokens, seq_len) for tokens, seq_len in zip(sequences, lengths, strict=True)], dim=1).mean(dim=1)
        joined = torch.cat([cls_summary, pma_summary, recent], dim=-1)
        gated = torch.sigmoid(self.gate(joined)) * (cls_summary + pma_summary + recent) / 3.0
        return self.norm(gated)


class InterFormerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, hidden_mult: int, dropout: float, add_context_token: bool) -> None:
        super().__init__()
        self.add_context_token = add_context_token
        self.cross_summary = CrossSummary(d_model, num_heads, dropout)
        self.ns_norm = nn.LayerNorm(d_model)
        self.ns_attention = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.ns_ffn = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * hidden_mult),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * hidden_mult, d_model),
        )
        self.ns_summary = nn.Sequential(nn.Linear(d_model, d_model), nn.SiLU(), nn.Linear(d_model, d_model))
        self.pffn = PersonalizedFeedForward(d_model, hidden_mult, dropout)
        self.seq_attention = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.seq_norm = nn.LayerNorm(d_model)
        self.seq_ffn = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * hidden_mult),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * hidden_mult, d_model),
        )

    def forward(
        self,
        ns_tokens: torch.Tensor,
        sequences: list[torch.Tensor],
        masks: list[torch.Tensor],
        lengths: list[torch.Tensor],
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        sequence_summary = self.cross_summary(sequences, masks, lengths).unsqueeze(1)
        interaction_tokens = torch.cat([ns_tokens, sequence_summary], dim=1)
        interaction_base = self.ns_norm(interaction_tokens)
        interacted, _weights = self.ns_attention(interaction_base, interaction_base, interaction_base, need_weights=False)
        interacted = interaction_tokens + interacted
        interacted = interacted + self.ns_ffn(interacted)
        ns_tokens = interacted[:, : ns_tokens.shape[1]]

        context = torch.sigmoid(self.ns_summary(masked_mean(ns_tokens))) * masked_mean(ns_tokens)
        updated_sequences: list[torch.Tensor] = []
        for tokens, mask in zip(sequences, masks, strict=True):
            personalized = tokens + self.pffn(tokens, context)
            if self.add_context_token:
                context_token = context.unsqueeze(1)
                attention_input = torch.cat([context_token, personalized], dim=1)
                context_mask = torch.zeros(mask.shape[0], 1, dtype=torch.bool, device=mask.device)
                attention_mask = torch.cat([context_mask, safe_key_padding_mask(mask)], dim=1)
                attended, _weights = self.seq_attention(
                    attention_input,
                    attention_input,
                    attention_input,
                    key_padding_mask=attention_mask,
                    need_weights=False,
                )
                attended = attended[:, 1:]
            else:
                attended, _weights = self.seq_attention(
                    personalized,
                    personalized,
                    personalized,
                    key_padding_mask=safe_key_padding_mask(mask),
                    need_weights=False,
                )
            updated = self.seq_norm(personalized + attended)
            updated_sequences.append(updated + self.seq_ffn(updated))
        return ns_tokens, updated_sequences


class PCVRInterFormer(EmbeddingParameterMixin, nn.Module):
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
        ns_tokenizer_type: str = "group",
        user_ns_tokens: int = 0,
        item_ns_tokens: int = 0,
    ) -> None:
        super().__init__()
        del num_queries, seq_encoder_type, seq_top_k, seq_causal, rank_mixer_mode, use_rope, rope_base, seq_id_threshold
        num_heads = choose_num_heads(d_model, num_heads)
        self.d_model = d_model
        self.action_num = action_num
        self.gradient_checkpointing = bool(gradient_checkpointing)
        self.seq_domains = sorted(seq_vocab_sizes)
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
        self.blocks = nn.ModuleList(
            [
                InterFormerBlock(d_model, num_heads, hidden_mult, dropout_rate, add_context_token=layer_index == 0)
                for layer_index in range(max(1, num_blocks))
            ]
        )
        self.final_cross = CrossSummary(d_model, num_heads, dropout_rate)
        self.final_gate = nn.Sequential(nn.Linear(d_model * 2, d_model), nn.Sigmoid())
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * hidden_mult),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(d_model * hidden_mult, action_num),
        )

    def _encode_non_sequence(self, inputs: ModelInput) -> torch.Tensor:
        parts = [self.user_tokenizer(inputs.user_int_feats)]
        user_dense = self.user_dense(inputs.user_dense_feats)
        if user_dense is not None:
            parts.append(user_dense)
        parts.append(self.item_tokenizer(inputs.item_int_feats))
        item_dense = self.item_dense(inputs.item_dense_feats)
        if item_dense is not None:
            parts.append(item_dense)
        return torch.cat(parts, dim=1)

    def _encode_sequences(self, inputs: ModelInput) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        sequences: list[torch.Tensor] = []
        masks: list[torch.Tensor] = []
        lengths: list[torch.Tensor] = []
        for domain in self.seq_domains:
            raw_sequence = inputs.seq_data[domain]
            seq_len = inputs.seq_lens[domain].to(raw_sequence.device)
            tokens = self.sequence_tokenizers[domain](raw_sequence, inputs.seq_time_buckets.get(domain))
            tokens = tokens + sinusoidal_positions(tokens.shape[1], self.d_model, tokens.device).unsqueeze(0)
            sequences.append(tokens)
            masks.append(make_padding_mask(seq_len, raw_sequence.shape[2]))
            lengths.append(seq_len)
        return sequences, masks, lengths

    def _embed(self, inputs: ModelInput) -> torch.Tensor:
        ns_tokens = self._encode_non_sequence(inputs)
        sequences, masks, lengths = self._encode_sequences(inputs)
        for block in self.blocks:
            ns_tokens, sequences = maybe_gradient_checkpoint(
                block,
                ns_tokens,
                sequences,
                masks,
                lengths,
                enabled=self.gradient_checkpointing,
            )
        ns_summary = masked_mean(ns_tokens)
        seq_summary = self.final_cross(sequences, masks, lengths)
        gate = self.final_gate(torch.cat([ns_summary, seq_summary], dim=-1))
        return gate * ns_summary + (1.0 - gate) * seq_summary

    def forward(self, inputs: ModelInput) -> torch.Tensor:
        return self.classifier(self._embed(inputs))

    def predict(self, inputs: ModelInput) -> tuple[torch.Tensor, torch.Tensor]:
        embeddings = self._embed(inputs)
        return self.classifier(embeddings), embeddings