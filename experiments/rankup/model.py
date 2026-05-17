"""RankUp-inspired high-rank token representation model for PCVR."""

from __future__ import annotations

import math
from typing import NamedTuple

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


TYPE_GLOBAL = 0
TYPE_CANDIDATE = 1
TYPE_TASK = 2
TYPE_USER = 3
TYPE_ITEM = 4
TYPE_DENSE = 5
TYPE_CROSS = 6
TYPE_SEQUENCE_START = 7


class RankUpTokenBatch(NamedTuple):
    tokens: torch.Tensor
    padding_mask: torch.Tensor
    readout_count: int


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


def _normalized_dense(features: torch.Tensor) -> torch.Tensor:
    normalized = torch.nan_to_num(features.float(), nan=0.0, posinf=0.0, neginf=0.0)
    return torch.sign(normalized) * torch.log1p(normalized.abs().clamp_max(1.0e8))


class RandomPermutationSparseTokenizer(nn.Module):
    """Sparse tokenizer with deterministic feature shuffling and multi-embedding views."""

    def __init__(
        self,
        feature_specs: list[tuple[int, int, int]],
        groups: list[list[int]],
        emb_dim: int,
        d_model: int,
        num_tokens: int,
        emb_skip_threshold: int,
        *,
        compress_high_cardinality: bool,
        num_tables: int,
        seed: int,
        use_missing_embeddings: bool,
    ) -> None:
        super().__init__()
        self.feature_count = len(feature_specs)
        self.d_model = int(d_model)
        self.table_count = max(1, int(num_tables))
        fallback_tokens = len([group for group in groups if group]) or self.feature_count
        self.num_tokens = 0 if self.feature_count <= 0 else max(1, int(num_tokens) if num_tokens > 0 else fallback_tokens)
        self.banks = nn.ModuleList(
            FeatureEmbeddingBank(
                feature_specs,
                emb_dim,
                emb_skip_threshold,
                compress_high_cardinality=compress_high_cardinality,
            )
            for _ in range(self.table_count)
        )
        self.use_missing_embeddings = bool(use_missing_embeddings) and self.feature_count > 0
        if self.use_missing_embeddings:
            self.missing_embeddings = nn.Parameter(torch.empty(self.table_count, self.feature_count, emb_dim))
            nn.init.normal_(self.missing_embeddings, mean=0.0, std=0.02)
        else:
            self.register_parameter("missing_embeddings", None)
        self.token_groups = self._make_token_groups(seed)
        self.project = nn.Sequential(
            nn.Linear(emb_dim * self.table_count, d_model),
            nn.SiLU(),
            nn.LayerNorm(d_model),
        )

    def _make_token_groups(self, seed: int) -> list[list[int]]:
        if self.num_tokens <= 0:
            return []
        generator = torch.Generator()
        generator.manual_seed(int(seed))
        permutation = torch.randperm(self.feature_count, generator=generator).tolist()
        groups: list[list[int]] = []
        for token_index in range(self.num_tokens):
            start = self.feature_count * token_index // self.num_tokens
            end = self.feature_count * (token_index + 1) // self.num_tokens
            groups.append(permutation[start:end])
        return groups

    def forward(self, int_feats: torch.Tensor, missing_mask: torch.Tensor | None = None) -> torch.Tensor:
        batch_size = int_feats.shape[0]
        if self.num_tokens <= 0:
            return int_feats.new_zeros(batch_size, 0, self.d_model, dtype=torch.float32)

        table_features: list[torch.Tensor] = []
        for table_index, bank in enumerate(self.banks):
            features = bank(int_feats)
            if self.missing_embeddings is not None and missing_mask is not None and features.shape[1] > 0:
                mask = missing_mask[:, : features.shape[1]].to(dtype=features.dtype, device=features.device)
                features = features + mask.unsqueeze(-1) * self.missing_embeddings[table_index, : features.shape[1]].unsqueeze(0)
            table_features.append(features)

        token_pieces: list[torch.Tensor] = []
        for group in self.token_groups:
            if group:
                table_views = [features[:, group, :].mean(dim=1) for features in table_features]
            else:
                table_views = [
                    int_feats.new_zeros(batch_size, bank.output_dim, dtype=torch.float32)
                    for bank in self.banks
                ]
            token_pieces.append(self.project(torch.cat(table_views, dim=-1)))
        return torch.stack(token_pieces, dim=1)


class DensePacketTokenizer(nn.Module):
    def __init__(self, input_dim: int, d_model: int, *, max_packets: int, use_missing_indicators: bool) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.d_model = int(d_model)
        self.use_missing_indicators = bool(use_missing_indicators)
        self.num_tokens = 0
        self.chunk_dim = 0
        self.pad_size = 0
        if self.input_dim <= 0:
            self.projects = nn.ModuleList()
            return
        effective_dim = self.input_dim * 2 if self.use_missing_indicators else self.input_dim
        self.num_tokens = max(1, min(int(max_packets), effective_dim))
        self.chunk_dim = (effective_dim + self.num_tokens - 1) // self.num_tokens
        self.pad_size = self.chunk_dim * self.num_tokens - effective_dim
        self.projects = nn.ModuleList(
            nn.Sequential(nn.Linear(self.chunk_dim, d_model), nn.SiLU(), nn.LayerNorm(d_model))
            for _ in range(self.num_tokens)
        )

    def forward(self, features: torch.Tensor, missing_mask: torch.Tensor | None = None) -> torch.Tensor:
        if self.num_tokens <= 0:
            return features.new_zeros(features.shape[0], 0, self.d_model, dtype=torch.float32)
        normalized = _normalized_dense(features)
        if self.use_missing_indicators:
            if missing_mask is None:
                missing = torch.zeros_like(normalized)
            else:
                missing = missing_mask[:, : normalized.shape[1]].to(dtype=normalized.dtype, device=normalized.device)
            normalized = torch.cat([normalized, missing], dim=-1)
        if self.pad_size > 0:
            normalized = torch.cat([normalized, normalized.new_zeros(normalized.shape[0], self.pad_size)], dim=-1)
        chunks = normalized.view(normalized.shape[0], self.num_tokens, self.chunk_dim)
        return torch.stack([project(chunks[:, index, :]) for index, project in enumerate(self.projects)], dim=1)


class CrossDenseToken(nn.Module):
    def __init__(self, user_dense_dim: int, item_dense_dim: int, d_model: int) -> None:
        super().__init__()
        self.cross_dim = max(0, min(int(user_dense_dim), int(item_dense_dim)))
        self.d_model = int(d_model)
        if self.cross_dim > 0:
            self.project = nn.Sequential(nn.Linear(self.cross_dim, d_model), nn.SiLU(), nn.LayerNorm(d_model))
        else:
            self.project = None

    @property
    def num_tokens(self) -> int:
        return int(self.project is not None)

    def forward(self, user_dense: torch.Tensor, item_dense: torch.Tensor) -> torch.Tensor:
        if self.project is None:
            return user_dense.new_zeros(user_dense.shape[0], 0, self.d_model, dtype=torch.float32)
        cross = _normalized_dense(user_dense[:, : self.cross_dim]) * _normalized_dense(item_dense[:, : self.cross_dim])
        return self.project(cross).unsqueeze(1)


class SwiGLUFeedForward(nn.Module):
    def __init__(self, d_model: int, hidden_mult: int, dropout: float) -> None:
        super().__init__()
        hidden_dim = int(d_model) * int(hidden_mult)
        self.up = nn.Linear(d_model, hidden_dim * 2)
        self.dropout = nn.Dropout(dropout)
        self.down = nn.Linear(hidden_dim, d_model)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        gate, value = self.up(tokens).chunk(2, dim=-1)
        return self.down(self.dropout(nn.functional.silu(gate) * value))


class RankUpSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.num_heads = int(num_heads)
        self.dropout = float(dropout)
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.query_norm = RMSNorm(d_model)
        self.key_norm = RMSNorm(d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, tokens: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        query, key, value = self.qkv(tokens).chunk(3, dim=-1)
        query = self.query_norm(query).to(dtype=value.dtype)
        key = self.key_norm(key).to(dtype=value.dtype)
        attention_mask = (~padding_mask).view(padding_mask.shape[0], 1, 1, padding_mask.shape[1])
        output = scaled_dot_product_attention(
            query,
            key,
            value,
            num_heads=self.num_heads,
            attn_mask=attention_mask,
            dropout_p=self.dropout,
            training=self.training,
        )
        return self.out(output)


class RankUpBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, hidden_mult: int, dropout: float) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(d_model)
        self.attention = RankUpSelfAttention(d_model, num_heads, dropout)
        self.ffn_norm = RMSNorm(d_model)
        self.ffn = SwiGLUFeedForward(d_model, hidden_mult, dropout)

    def forward(self, tokens: torch.Tensor, padding_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        attn_update = self.attention(self.attn_norm(tokens), padding_mask)
        attn_tokens = tokens + attn_update * (~padding_mask).to(attn_update.dtype).unsqueeze(-1)
        ffn_update = self.ffn(self.ffn_norm(attn_tokens))
        output = attn_tokens + ffn_update * (~padding_mask).to(ffn_update.dtype).unsqueeze(-1)
        return output, attn_tokens


class PCVRRankUp(EmbeddingParameterMixin, nn.Module):
    """RankUp-style PCVR model focused on high-rank token representations."""

    def __init__(
        self,
        user_int_feature_specs: list[tuple[int, int, int]],
        item_int_feature_specs: list[tuple[int, int, int]],
        user_dense_dim: int,
        item_dense_dim: int,
        seq_vocab_sizes: dict[str, list[int]],
        user_ns_groups: list[list[int]],
        item_ns_groups: list[list[int]],
        d_model: int = 128,
        emb_dim: int = 64,
        num_queries: int = 1,
        num_blocks: int = 2,
        num_heads: int = 4,
        seq_encoder_type: str = "transformer",
        hidden_mult: int = 4,
        dropout_rate: float = 0.02,
        seq_top_k: int = 96,
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
        user_ns_tokens: int = 8,
        item_ns_tokens: int = 4,
    ) -> None:
        super().__init__()
        del num_queries, seq_encoder_type, seq_causal, rank_mixer_mode, use_rope, rope_base, ns_tokenizer_type
        num_heads = choose_num_heads(d_model, num_heads)
        self.d_model = int(d_model)
        self.action_num = int(action_num)
        self.gradient_checkpointing = bool(gradient_checkpointing)
        self.seq_domains = sorted(seq_vocab_sizes)
        self.seq_keep_per_domain = max(1, int(seq_top_k))
        self.task_token_count = max(1, self.action_num)

        self.user_sparse = RandomPermutationSparseTokenizer(
            user_int_feature_specs,
            user_ns_groups,
            emb_dim,
            d_model,
            user_ns_tokens,
            emb_skip_threshold,
            compress_high_cardinality=True,
            num_tables=2,
            seed=20260420,
            use_missing_embeddings=True,
        )
        self.item_sparse = RandomPermutationSparseTokenizer(
            item_int_feature_specs,
            item_ns_groups,
            emb_dim,
            d_model,
            item_ns_tokens,
            emb_skip_threshold,
            compress_high_cardinality=True,
            num_tables=2,
            seed=20260512,
            use_missing_embeddings=True,
        )
        self.user_dense = DensePacketTokenizer(user_dense_dim, d_model, max_packets=3, use_missing_indicators=True)
        self.item_dense = DensePacketTokenizer(item_dense_dim, d_model, max_packets=1, use_missing_indicators=True)
        self.cross_dense = CrossDenseToken(user_dense_dim, item_dense_dim, d_model)
        self.sequence_tokenizers = nn.ModuleDict(
            {
                domain: SequenceTokenizer(
                    vocab_sizes,
                    emb_dim,
                    d_model,
                    num_time_buckets,
                    seq_id_threshold or emb_skip_threshold,
                    compress_high_cardinality=True,
                )
                for domain, vocab_sizes in seq_vocab_sizes.items()
            }
        )

        self.type_embedding = nn.Embedding(TYPE_SEQUENCE_START + max(1, len(self.seq_domains)), d_model)
        self.global_project = nn.Sequential(nn.Linear(d_model, d_model), nn.SiLU(), nn.LayerNorm(d_model))
        self.candidate_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.task_tokens = nn.Parameter(torch.randn(1, self.task_token_count, d_model) * 0.02)
        self.blocks = nn.ModuleList(
            [RankUpBlock(d_model, num_heads, hidden_mult, dropout_rate) for _ in range(max(1, int(num_blocks)))]
        )
        self.final_norm = RMSNorm(d_model)
        self.classifier = nn.Sequential(
            nn.Linear(d_model * 4, d_model),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(d_model, action_num),
        )

        self.num_ns = (
            2
            + self.task_token_count
            + self.user_sparse.num_tokens
            + self.item_sparse.num_tokens
            + self.user_dense.num_tokens
            + self.item_dense.num_tokens
            + self.cross_dense.num_tokens
        )
        self._training_diagnostics_enabled = False
        self._diagnostic_scalars: dict[str, float] = {}

    def set_tensorboard_diagnostics_enabled(self, enabled: bool) -> None:
        self.set_training_diagnostics_enabled(enabled)

    def set_training_diagnostics_enabled(self, enabled: bool) -> None:
        self._training_diagnostics_enabled = bool(enabled)

    def consume_tensorboard_scalars(self, *, phase: str) -> dict[str, float]:
        return self.consume_training_scalars(phase=phase)

    def consume_training_scalars(self, *, phase: str) -> dict[str, float]:
        clean_phase = str(phase).strip().replace("/", "_") or "train"
        scalars = {
            f"RankUp/{metric_name}/{clean_phase}": value
            for metric_name, value in self._diagnostic_scalars.items()
        }
        self._diagnostic_scalars = {}
        return scalars

    def _should_collect_diagnostics(self) -> bool:
        return self._training_diagnostics_enabled and not _torch_is_compiling()

    def _put_scalar(self, metric_name: str, value: float | torch.Tensor) -> None:
        if not self._should_collect_diagnostics():
            return
        if isinstance(value, torch.Tensor):
            if value.numel() == 0:
                return
            value = float(value.detach().float().mean().cpu())
        else:
            value = float(value)
        if math.isfinite(value):
            self._diagnostic_scalars[metric_name] = value

    def _effective_rank(self, tokens: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        if not self._should_collect_diagnostics() or tokens.shape[1] <= 0:
            return tokens.new_tensor(0.0)
        with torch.no_grad():
            hidden = tokens.detach().float()
            valid = (~padding_mask).float().unsqueeze(-1)
            centered = hidden - (hidden * valid).sum(dim=1, keepdim=True) / valid.sum(dim=1, keepdim=True).clamp_min(1.0)
            centered = centered * valid
            singular_values = torch.linalg.svdvals(centered)
            weights = singular_values / singular_values.sum(dim=-1, keepdim=True).clamp_min(1.0e-12)
            entropy = -(weights * torch.log(weights.clamp_min(1.0e-12))).sum(dim=-1)
            return torch.exp(entropy).mean()

    def _decorate(
        self,
        tokens: torch.Tensor,
        type_id: int,
        *,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, token_count, _dim = tokens.shape
        if token_count <= 0:
            empty_mask = torch.ones(batch_size, 0, dtype=torch.bool, device=tokens.device)
            return tokens, empty_mask
        type_ids = torch.full((batch_size, token_count), int(type_id), dtype=torch.long, device=tokens.device)
        if mask is None:
            mask = torch.zeros(batch_size, token_count, dtype=torch.bool, device=tokens.device)
        return tokens + self.type_embedding(type_ids), mask

    def _missing_mask_or_default(self, mask: torch.Tensor | None, features: torch.Tensor, *, dense: bool) -> torch.Tensor:
        if mask is not None:
            return mask.to(features.device).bool()
        if dense:
            return ~torch.isfinite(features.float())
        return features <= 0

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
            positions = sinusoidal_positions(tokens.shape[1], self.d_model, tokens.device).unsqueeze(0).to(tokens.dtype)
            tokens = tokens + positions
            decorated, mask = self._decorate(
                tokens,
                TYPE_SEQUENCE_START + domain_index,
                mask=make_padding_mask(seq_len, tokens.shape[1]),
            )
            pieces.append(decorated)
            masks.append(mask)
        return pieces, masks

    def _encode_tokens(self, inputs: ModelInput) -> RankUpTokenBatch:
        batch_size = inputs.user_int_feats.shape[0]
        user_int_missing = self._missing_mask_or_default(inputs.user_int_missing_mask, inputs.user_int_feats, dense=False)
        item_int_missing = self._missing_mask_or_default(inputs.item_int_missing_mask, inputs.item_int_feats, dense=False)
        user_dense_missing = self._missing_mask_or_default(inputs.user_dense_missing_mask, inputs.user_dense_feats, dense=True)
        item_dense_missing = self._missing_mask_or_default(inputs.item_dense_missing_mask, inputs.item_dense_feats, dense=True)

        user_sparse = self.user_sparse(inputs.user_int_feats, user_int_missing)
        item_sparse = self.item_sparse(inputs.item_int_feats, item_int_missing)
        user_dense = self.user_dense(inputs.user_dense_feats, user_dense_missing)
        item_dense = self.item_dense(inputs.item_dense_feats, item_dense_missing)
        cross_dense = self.cross_dense(inputs.user_dense_feats, inputs.item_dense_feats)

        non_sequence_parts = [user_sparse, item_sparse]
        if user_dense.shape[1] > 0:
            non_sequence_parts.append(user_dense)
        if item_dense.shape[1] > 0:
            non_sequence_parts.append(item_dense)
        if cross_dense.shape[1] > 0:
            non_sequence_parts.append(cross_dense)
        non_sequence_tokens = torch.cat(non_sequence_parts, dim=1)
        empty_body_mask = torch.zeros(batch_size, non_sequence_tokens.shape[1], dtype=torch.bool, device=non_sequence_tokens.device)
        global_token = self.global_project(masked_mean(non_sequence_tokens, empty_body_mask)).unsqueeze(1)

        item_context_parts = [item_sparse]
        if item_dense.shape[1] > 0:
            item_context_parts.append(item_dense)
        item_context = torch.cat(item_context_parts, dim=1)
        candidate = self.candidate_token.expand(batch_size, -1, -1) + item_context.mean(dim=1, keepdim=True)
        task_tokens = self.task_tokens.expand(batch_size, -1, -1)

        pieces: list[torch.Tensor] = []
        masks: list[torch.Tensor] = []
        for tokens, type_id in (
            (global_token, TYPE_GLOBAL),
            (candidate, TYPE_CANDIDATE),
            (task_tokens, TYPE_TASK),
            (user_sparse, TYPE_USER),
            (item_sparse, TYPE_ITEM),
            (user_dense, TYPE_DENSE),
            (item_dense, TYPE_DENSE),
            (cross_dense, TYPE_CROSS),
        ):
            decorated, mask = self._decorate(tokens, type_id)
            if decorated.shape[1] > 0:
                pieces.append(decorated)
                masks.append(mask)

        sequence_pieces, sequence_masks = self._encode_sequence_events(inputs)
        pieces.extend(sequence_pieces)
        masks.extend(sequence_masks)
        return RankUpTokenBatch(
            tokens=torch.cat(pieces, dim=1),
            padding_mask=torch.cat(masks, dim=1),
            readout_count=2 + self.task_token_count,
        )

    def _run_backbone(self, batch: RankUpTokenBatch) -> torch.Tensor:
        tokens = batch.tokens
        self._put_scalar("effective_rank/input", self._effective_rank(tokens, batch.padding_mask))
        for block_index, block in enumerate(self.blocks, start=1):
            tokens, attn_tokens = maybe_gradient_checkpoint(
                block,
                tokens,
                batch.padding_mask,
                enabled=self.gradient_checkpointing,
            )
            self._put_scalar(f"effective_rank/block{block_index}_tm", self._effective_rank(attn_tokens, batch.padding_mask))
            self._put_scalar(f"effective_rank/block{block_index}_ffn", self._effective_rank(tokens, batch.padding_mask))
        return self.final_norm(tokens)

    def _embed(self, inputs: ModelInput) -> torch.Tensor:
        batch = self._encode_tokens(inputs)
        tokens = self._run_backbone(batch)
        global_summary = tokens[:, 0, :]
        candidate_summary = tokens[:, 1, :]
        task_summary = tokens[:, 2 : batch.readout_count, :].mean(dim=1)
        body_summary = masked_mean(tokens[:, batch.readout_count :, :], batch.padding_mask[:, batch.readout_count :])
        embedding = torch.cat([global_summary, candidate_summary, task_summary, body_summary], dim=-1)
        self._put_scalar("tokens/count", float(batch.tokens.shape[1]))
        self._put_scalar("embedding/norm_mean", embedding.detach().float().norm(dim=-1).mean())
        return embedding

    def forward(self, inputs: ModelInput) -> torch.Tensor:
        return self.classifier(self._embed(inputs))

    def predict(self, inputs: ModelInput) -> tuple[torch.Tensor, torch.Tensor]:
        embeddings = self._embed(inputs)
        return self.classifier(embeddings), embeddings


__all__ = ["ModelInput", "PCVRRankUp", "RankUpSelfAttention", "configure_rms_norm_runtime"]
