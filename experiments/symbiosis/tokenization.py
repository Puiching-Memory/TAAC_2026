"""Unified tokenization for Symbiosis V2/V3."""

from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn as nn

from taac2026.api import FeatureEmbeddingBank, ModelInput, RMSNorm, SequenceTokenizer, sinusoidal_positions


SEQUENCE_STATS_DIM = 6

ROLE_CLS = 0
ROLE_CANDIDATE = 1
ROLE_USER = 2
ROLE_ITEM = 3
ROLE_DENSE = 4
ROLE_MISSING = 5
ROLE_SEQUENCE = 6
ROLE_STATS = 7
ROLE_COUNT = 8
V3_MEMORY_SELECTION_MODES = {"uniform", "stratified", "quality_stratified"}


class UnifiedTokenBatch(NamedTuple):
    tokens: torch.Tensor
    padding_mask: torch.Tensor
    role_ids: torch.Tensor
    domain_ids: torch.Tensor
    risk_ids: torch.Tensor
    cls_index: int
    candidate_index: int


def _feature_width(feature_specs: list[tuple[int, int, int]]) -> int:
    return max((offset + length for _vocab_size, offset, length in feature_specs), default=0)


def _parse_domain_budget(value: str, *, default: int) -> dict[str, int]:
    budgets: dict[str, int] = {}
    for chunk in str(value or "").split(","):
        token = chunk.strip()
        if not token:
            continue
        if ":" not in token:
            raise ValueError(f"invalid domain token budget entry: {token!r}")
        domain, raw_count = token.split(":", 1)
        clean_domain = domain.strip()
        if not clean_domain:
            raise ValueError(f"invalid empty domain in token budget entry: {token!r}")
        budgets[clean_domain] = max(0, int(raw_count.strip()))
    budgets.setdefault("*", max(0, int(default)))
    return budgets


class V2GroupedSparseTokenizer(nn.Module):
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
        *,
        use_missing_embeddings: bool,
    ) -> None:
        super().__init__()
        self.mode = str(mode).strip().lower() or "group"
        if self.mode not in {"group", "group_compressed", "random_chunk"}:
            raise ValueError(f"unknown symbiosis_v2_tokenization_mode: {mode}")
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
            self.groups = []
        elif self.mode == "group":
            self.num_tokens = len(self.groups)
        else:
            self.num_tokens = max(1, int(num_tokens))
        self.use_missing_embeddings = bool(use_missing_embeddings) and self.feature_count > 0
        if self.use_missing_embeddings:
            self.missing_embeddings = nn.Parameter(torch.empty(self.feature_count, emb_dim))
            nn.init.normal_(self.missing_embeddings, mean=0.0, std=0.02)
        else:
            self.register_parameter("missing_embeddings", None)
        self.group_project = nn.Sequential(nn.Linear(emb_dim, d_model), nn.SiLU(), nn.LayerNorm(d_model))
        if self.mode == "group_compressed" and self.feature_count > 0:
            self.compress_project = nn.Sequential(
                RMSNorm(d_model * len(self.groups)),
                nn.Linear(d_model * len(self.groups), self.num_tokens * d_model),
                nn.SiLU(),
                nn.LayerNorm(self.num_tokens * d_model),
            )
        elif self.mode == "random_chunk" and self.feature_count > 0:
            self.compress_project = nn.Sequential(
                RMSNorm(d_model * len(self.groups)),
                nn.Linear(d_model * len(self.groups), self.num_tokens * d_model),
                nn.SiLU(),
                nn.LayerNorm(self.num_tokens * d_model),
            )
        else:
            self.compress_project = None

    def forward(self, int_feats: torch.Tensor, missing_mask: torch.Tensor | None) -> torch.Tensor:
        batch_size = int_feats.shape[0]
        if self.num_tokens <= 0:
            return int_feats.new_zeros(batch_size, 0, self.d_model, dtype=torch.float32)
        feature_tokens = self.bank(int_feats)
        if self.missing_embeddings is not None and missing_mask is not None and feature_tokens.shape[1] > 0:
            mask = missing_mask[:, : feature_tokens.shape[1]].to(dtype=feature_tokens.dtype, device=feature_tokens.device)
            feature_tokens = feature_tokens + mask.unsqueeze(-1) * self.missing_embeddings[: feature_tokens.shape[1]].unsqueeze(0)
        group_pieces: list[torch.Tensor] = []
        for group in self.groups:
            valid_indices = [index for index in group if 0 <= index < feature_tokens.shape[1]]
            if valid_indices:
                group_pieces.append(feature_tokens[:, valid_indices, :].mean(dim=1))
            else:
                group_pieces.append(int_feats.new_zeros(batch_size, self.bank.output_dim, dtype=torch.float32))
        group_tokens = self.group_project(torch.stack(group_pieces, dim=1))
        if self.compress_project is None:
            return group_tokens
        return self.compress_project(group_tokens.reshape(batch_size, -1)).view(batch_size, self.num_tokens, self.d_model)


class V2DensePacketTokenizer(nn.Module):
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
        normalized = torch.nan_to_num(features.float(), nan=0.0, posinf=0.0, neginf=0.0)
        normalized = torch.sign(normalized) * torch.log1p(normalized.abs().clamp_max(1.0e8))
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


class V2SequenceStatsTokenizer(nn.Module):
    def __init__(self, domain_count: int, d_model: int) -> None:
        super().__init__()
        self.input_dim = int(domain_count) * SEQUENCE_STATS_DIM
        self.d_model = int(d_model)
        if self.input_dim > 0:
            self.project = nn.Sequential(RMSNorm(self.input_dim), nn.Linear(self.input_dim, d_model), nn.SiLU(), nn.LayerNorm(d_model))
        else:
            self.project = None

    @property
    def num_tokens(self) -> int:
        return int(self.project is not None)

    def forward(self, stats_by_domain: dict[str, torch.Tensor] | None, domains: list[str], reference: torch.Tensor) -> torch.Tensor:
        if self.project is None:
            return reference.new_zeros(reference.shape[0], 0, self.d_model, dtype=torch.float32)
        batch_size = reference.shape[0]
        pieces: list[torch.Tensor] = []
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


class UnifiedSymbiosisTokenizer(nn.Module):
    def __init__(
        self,
        *,
        user_int_feature_specs: list[tuple[int, int, int]],
        item_int_feature_specs: list[tuple[int, int, int]],
        user_dense_dim: int,
        item_dense_dim: int,
        seq_vocab_sizes: dict[str, list[int]],
        user_ns_groups: list[list[int]],
        item_ns_groups: list[list[int]],
        d_model: int,
        emb_dim: int,
        user_ns_tokens: int,
        item_ns_tokens: int,
        emb_skip_threshold: int,
        seq_id_threshold: int,
        num_time_buckets: int,
        tokenization_mode: str,
        recent_event_tokens: int,
        memory_event_tokens: int,
        user_dense_tokens: int,
        item_dense_tokens: int,
        user_missing_tokens: int,
        item_missing_tokens: int,
        compress_large_ids: bool,
        use_dense_tokens: bool,
        use_missing_tokens: bool,
        use_sequence_stats_tokens: bool,
        v3_enabled: bool,
        v3_memory_selection_mode: str,
        v3_recent_event_tokens_by_domain: str,
        v3_memory_event_tokens_by_domain: str,
        v3_memory_density_weight: float,
        v3_memory_time_weight: float,
        v3_memory_recency_weight: float,
        v3_memory_duplicate_penalty: float,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.seq_domains = sorted(seq_vocab_sizes)
        self.recent_event_tokens = max(0, int(recent_event_tokens))
        self.memory_event_tokens = max(0, int(memory_event_tokens))
        self.v3_enabled = bool(v3_enabled)
        self.v3_memory_selection_mode = str(v3_memory_selection_mode).strip().lower() or "quality_stratified"
        if self.v3_memory_selection_mode not in V3_MEMORY_SELECTION_MODES:
            raise ValueError(f"unknown symbiosis_v3_memory_selection_mode: {v3_memory_selection_mode}")
        self.v3_recent_event_tokens_by_domain = _parse_domain_budget(
            v3_recent_event_tokens_by_domain,
            default=self.recent_event_tokens,
        )
        self.v3_memory_event_tokens_by_domain = _parse_domain_budget(
            v3_memory_event_tokens_by_domain,
            default=self.memory_event_tokens,
        )
        self.v3_memory_density_weight = float(v3_memory_density_weight)
        self.v3_memory_time_weight = float(v3_memory_time_weight)
        self.v3_memory_recency_weight = float(v3_memory_recency_weight)
        self.v3_memory_duplicate_penalty = float(v3_memory_duplicate_penalty)
        self.event_tokens_by_domain = {
            domain: self._domain_recent_budget(domain) + self._domain_memory_budget(domain)
            for domain in self.seq_domains
        }
        self.event_tokens_per_domain = self.recent_event_tokens + self.memory_event_tokens
        self.use_dense_tokens = bool(use_dense_tokens)
        self.use_missing_tokens = bool(use_missing_tokens)
        self.use_sequence_stats_tokens = bool(use_sequence_stats_tokens)
        self.user_sparse = V2GroupedSparseTokenizer(
            user_int_feature_specs,
            user_ns_groups,
            emb_dim,
            d_model,
            user_ns_tokens,
            emb_skip_threshold,
            compress_large_ids,
            tokenization_mode,
            use_missing_embeddings=True,
        )
        self.item_sparse = V2GroupedSparseTokenizer(
            item_int_feature_specs,
            item_ns_groups,
            emb_dim,
            d_model,
            item_ns_tokens,
            emb_skip_threshold,
            compress_large_ids,
            tokenization_mode,
            use_missing_embeddings=True,
        )
        self.user_dense = V2DensePacketTokenizer(
            user_dense_dim,
            d_model,
            max_packets=user_dense_tokens,
            use_missing_indicators=True,
        ) if self.use_dense_tokens else None
        self.item_dense = V2DensePacketTokenizer(
            item_dense_dim,
            d_model,
            max_packets=item_dense_tokens,
            use_missing_indicators=True,
        ) if self.use_dense_tokens else None
        user_missing_dim = _feature_width(user_int_feature_specs) + int(user_dense_dim)
        item_missing_dim = _feature_width(item_int_feature_specs) + int(item_dense_dim)
        self.user_missing = V2DensePacketTokenizer(
            user_missing_dim,
            d_model,
            max_packets=user_missing_tokens,
            use_missing_indicators=False,
        ) if self.use_missing_tokens else None
        self.item_missing = V2DensePacketTokenizer(
            item_missing_dim,
            d_model,
            max_packets=item_missing_tokens,
            use_missing_indicators=False,
        ) if self.use_missing_tokens else None
        self.sequence_tokenizers = nn.ModuleDict(
            {
                domain: SequenceTokenizer(
                    vocab_sizes,
                    emb_dim,
                    d_model,
                    num_time_buckets,
                    seq_id_threshold or emb_skip_threshold,
                    compress_high_cardinality=compress_large_ids,
                )
                for domain, vocab_sizes in seq_vocab_sizes.items()
            }
        )
        self.sequence_stats = V2SequenceStatsTokenizer(len(self.seq_domains), d_model) if self.use_sequence_stats_tokens else None
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.candidate_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.role_embedding = nn.Embedding(ROLE_COUNT, d_model)
        self.domain_embedding = nn.Embedding(max(1, len(self.seq_domains) + 1), d_model)
        self.risk_embedding = nn.Embedding(2, d_model)

        self.num_sequence_tokens = sum(self.event_tokens_by_domain.values())
        if self.sequence_stats is not None:
            self.num_sequence_tokens += self.sequence_stats.num_tokens
        self.num_non_sequence_tokens = 2 + self.user_sparse.num_tokens + self.item_sparse.num_tokens
        if self.user_dense is not None:
            self.num_non_sequence_tokens += self.user_dense.num_tokens + self.item_dense.num_tokens
        if self.user_missing is not None:
            self.num_non_sequence_tokens += self.user_missing.num_tokens + self.item_missing.num_tokens

    def _domain_recent_budget(self, domain: str) -> int:
        if not self.v3_enabled:
            return self.recent_event_tokens
        return int(self.v3_recent_event_tokens_by_domain.get(domain, self.v3_recent_event_tokens_by_domain["*"]))

    def _domain_memory_budget(self, domain: str) -> int:
        if not self.v3_enabled:
            return self.memory_event_tokens
        return int(self.v3_memory_event_tokens_by_domain.get(domain, self.v3_memory_event_tokens_by_domain["*"]))

    def forward(self, inputs: ModelInput) -> UnifiedTokenBatch:
        batch_size = inputs.user_int_feats.shape[0]
        user_int_missing = self._missing_mask_or_default(inputs.user_int_missing_mask, inputs.user_int_feats, dense=False)
        item_int_missing = self._missing_mask_or_default(inputs.item_int_missing_mask, inputs.item_int_feats, dense=False)
        user_dense_missing = self._missing_mask_or_default(inputs.user_dense_missing_mask, inputs.user_dense_feats, dense=True)
        item_dense_missing = self._missing_mask_or_default(inputs.item_dense_missing_mask, inputs.item_dense_feats, dense=True)

        user_sparse = self.user_sparse(inputs.user_int_feats, user_int_missing)
        item_sparse = self.item_sparse(inputs.item_int_feats, item_int_missing)
        user_dense = self.user_dense(inputs.user_dense_feats, user_dense_missing) if self.user_dense is not None else None
        item_dense = self.item_dense(inputs.item_dense_feats, item_dense_missing) if self.item_dense is not None else None
        user_missing = self._missing_tokens(user_int_missing, user_dense_missing, inputs.user_int_feats, self.user_missing)
        item_missing = self._missing_tokens(item_int_missing, item_dense_missing, inputs.item_int_feats, self.item_missing)

        item_context_parts = [item_sparse]
        if item_dense is not None:
            item_context_parts.append(item_dense)
        if item_missing is not None:
            item_context_parts.append(item_missing)
        item_context = torch.cat(item_context_parts, dim=1)
        item_summary = item_context.mean(dim=1) if item_context.shape[1] > 0 else inputs.user_int_feats.new_zeros(batch_size, self.d_model, dtype=torch.float32)

        pieces: list[torch.Tensor] = []
        masks: list[torch.Tensor] = []
        role_ids: list[int] = []
        domain_ids: list[int] = []
        risk_ids: list[int] = []

        def append(tokens: torch.Tensor, role_id: int, *, domain_id: int = 0, risk_id: int = 0, mask: torch.Tensor | None = None) -> None:
            if tokens.shape[1] <= 0:
                return
            token_count = tokens.shape[1]
            role_tensor = torch.full((token_count,), role_id, dtype=torch.long, device=tokens.device)
            domain_tensor = torch.full((token_count,), domain_id, dtype=torch.long, device=tokens.device)
            risk_tensor = torch.full((token_count,), risk_id, dtype=torch.long, device=tokens.device)
            decorated = tokens + self.role_embedding(role_tensor).unsqueeze(0)
            decorated = decorated + self.domain_embedding(domain_tensor).unsqueeze(0)
            decorated = decorated + self.risk_embedding(risk_tensor).unsqueeze(0)
            pieces.append(decorated)
            if mask is None:
                masks.append(torch.zeros(tokens.shape[0], token_count, dtype=torch.bool, device=tokens.device))
            else:
                masks.append(mask)
            role_ids.extend([role_id] * token_count)
            domain_ids.extend([domain_id] * token_count)
            risk_ids.extend([risk_id] * token_count)

        cls_index = 0
        candidate_index = 1
        append(self.cls_token.expand(batch_size, -1, -1), ROLE_CLS)
        append(self.candidate_token.expand(batch_size, -1, -1) + item_summary.unsqueeze(1), ROLE_CANDIDATE)
        append(user_sparse, ROLE_USER)
        append(item_sparse, ROLE_ITEM, risk_id=1)
        if user_dense is not None:
            append(user_dense, ROLE_DENSE, risk_id=1)
        if item_dense is not None:
            append(item_dense, ROLE_DENSE, risk_id=1)
        if user_missing is not None:
            append(user_missing, ROLE_MISSING)
        if item_missing is not None:
            append(item_missing, ROLE_MISSING, risk_id=1)

        for domain_index, domain in enumerate(self.seq_domains, start=1):
            sequence_tokens, sequence_mask = self._sequence_event_tokens(domain, inputs)
            append(sequence_tokens, ROLE_SEQUENCE, domain_id=domain_index, mask=sequence_mask)
        if self.sequence_stats is not None:
            append(self.sequence_stats(inputs.seq_stats, self.seq_domains, inputs.user_int_feats), ROLE_STATS)

        tokens = torch.cat(pieces, dim=1)
        padding_mask = torch.cat(masks, dim=1)
        return UnifiedTokenBatch(
            tokens=tokens,
            padding_mask=padding_mask,
            role_ids=torch.tensor(role_ids, dtype=torch.long, device=tokens.device),
            domain_ids=torch.tensor(domain_ids, dtype=torch.long, device=tokens.device),
            risk_ids=torch.tensor(risk_ids, dtype=torch.long, device=tokens.device),
            cls_index=cls_index,
            candidate_index=candidate_index,
        )

    def _missing_mask_or_default(self, mask: torch.Tensor | None, features: torch.Tensor, *, dense: bool) -> torch.Tensor:
        if mask is not None:
            return mask.to(features.device).bool()
        if dense:
            return ~torch.isfinite(features.float())
        return features <= 0

    def _missing_tokens(
        self,
        int_missing: torch.Tensor,
        dense_missing: torch.Tensor,
        reference: torch.Tensor,
        tokenizer: V2DensePacketTokenizer | None,
    ) -> torch.Tensor | None:
        if tokenizer is None:
            return None
        features = torch.cat([int_missing.float(), dense_missing.float()], dim=-1)
        if features.shape[1] <= 0:
            return reference.new_zeros(reference.shape[0], 0, self.d_model, dtype=torch.float32)
        return tokenizer(features)

    def _sequence_event_tokens(self, domain: str, inputs: ModelInput) -> tuple[torch.Tensor, torch.Tensor]:
        sequence = inputs.seq_data[domain]
        time_buckets = inputs.seq_time_buckets.get(domain)
        lengths = inputs.seq_lens[domain].to(sequence.device)
        compact_sequence, compact_time_buckets, compact_mask = self._compact_sequence(domain, sequence, time_buckets, lengths)
        if compact_sequence.shape[2] <= 0:
            tokens = sequence.new_zeros(sequence.shape[0], 0, self.d_model, dtype=torch.float32)
            return tokens, compact_mask
        with torch.autocast(device_type=sequence.device.type, enabled=False):
            tokens = self.sequence_tokenizers[domain](compact_sequence, compact_time_buckets)
        if tokens.shape[1] > 0:
            positions = sinusoidal_positions(tokens.shape[1], self.d_model, tokens.device).unsqueeze(0).to(tokens.dtype)
            tokens = tokens + positions
            tokens = tokens * (~compact_mask).to(tokens.dtype).unsqueeze(-1)
        return tokens, compact_mask

    def _compact_sequence(
        self,
        domain: str,
        sequence: torch.Tensor,
        time_buckets: torch.Tensor | None,
        lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        batch_size, feature_count, max_len = sequence.shape
        recent_count = self._domain_recent_budget(domain)
        memory_count = self._domain_memory_budget(domain)
        token_budget = recent_count + memory_count
        if token_budget <= 0:
            empty = sequence.new_zeros(batch_size, feature_count, 0)
            empty_mask = torch.ones(batch_size, 0, dtype=torch.bool, device=sequence.device)
            return empty, None, empty_mask
        if max_len <= 0:
            empty = sequence.new_zeros(batch_size, feature_count, token_budget)
            empty_mask = torch.ones(batch_size, token_budget, dtype=torch.bool, device=sequence.device)
            return empty, None, empty_mask
        clamped_lengths = lengths.clamp(min=0, max=max_len).to(torch.long)
        positions: list[torch.Tensor] = []
        masks: list[torch.Tensor] = []
        if memory_count > 0:
            prefix_lengths = (clamped_lengths - recent_count).clamp_min(0)
            memory_positions, memory_mask = self._memory_positions(
                sequence,
                time_buckets,
                prefix_lengths,
                memory_count,
            )
            positions.append(memory_positions)
            masks.append(memory_mask)
        if recent_count > 0:
            offsets = torch.arange(recent_count, device=sequence.device)
            start = (clamped_lengths - recent_count).clamp_min(0)
            recent_positions = (start.unsqueeze(1) + offsets).clamp_max(max_len - 1)
            positions.append(recent_positions)
            masks.append(offsets.unsqueeze(0) >= clamped_lengths.clamp_max(recent_count).unsqueeze(1))
        gather_positions = torch.cat(positions, dim=1)
        padding_mask = torch.cat(masks, dim=1)
        gather_index = gather_positions.unsqueeze(1).expand(-1, feature_count, -1)
        compact_sequence = sequence.gather(2, gather_index)
        compact_time_buckets = None
        if time_buckets is not None:
            if time_buckets.shape[1] <= 0:
                compact_time_buckets = torch.zeros_like(gather_positions)
            else:
                compact_time_buckets = time_buckets.gather(1, gather_positions.clamp_max(time_buckets.shape[1] - 1))
        return compact_sequence, compact_time_buckets, padding_mask

    def _memory_positions(
        self,
        sequence: torch.Tensor,
        time_buckets: torch.Tensor | None,
        prefix_lengths: torch.Tensor,
        memory_count: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.v3_enabled or self.v3_memory_selection_mode == "uniform":
            return self._uniform_memory_positions(prefix_lengths, memory_count, sequence.shape[2])
        if self.v3_memory_selection_mode == "stratified":
            return self._stratified_memory_positions(prefix_lengths, memory_count, sequence.shape[2])
        return self._quality_stratified_memory_positions(sequence, time_buckets, prefix_lengths, memory_count)

    def _uniform_memory_positions(
        self,
        prefix_lengths: torch.Tensor,
        memory_count: int,
        max_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        offsets = torch.arange(memory_count, device=prefix_lengths.device)
        positions = (prefix_lengths.unsqueeze(1) * (offsets + 1) // (memory_count + 1)).clamp_max(max_len - 1)
        mask = offsets.unsqueeze(0) >= prefix_lengths.clamp_max(memory_count).unsqueeze(1)
        return positions, mask

    def _stratified_memory_positions(
        self,
        prefix_lengths: torch.Tensor,
        memory_count: int,
        max_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        offsets = torch.arange(memory_count, device=prefix_lengths.device)
        starts = prefix_lengths.unsqueeze(1) * offsets // memory_count
        ends = prefix_lengths.unsqueeze(1) * (offsets + 1) // memory_count
        positions = (ends - 1).clamp_min(0).clamp_max(max_len - 1)
        mask = (ends <= starts) | (starts >= prefix_lengths.unsqueeze(1))
        return positions, mask

    def _quality_stratified_memory_positions(
        self,
        sequence: torch.Tensor,
        time_buckets: torch.Tensor | None,
        prefix_lengths: torch.Tensor,
        memory_count: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        max_len = int(sequence.shape[2])
        scores = self._memory_event_scores(sequence, time_buckets, prefix_lengths)
        positions = torch.arange(max_len, device=sequence.device).unsqueeze(0)
        selected_positions: list[torch.Tensor] = []
        selected_masks: list[torch.Tensor] = []
        for bucket_index in range(memory_count):
            starts = prefix_lengths * bucket_index // memory_count
            ends = prefix_lengths * (bucket_index + 1) // memory_count
            bucket_mask = (positions >= starts.unsqueeze(1)) & (positions < ends.unsqueeze(1))
            bucket_scores = scores.masked_fill(~bucket_mask, torch.finfo(scores.dtype).min)
            best_scores, best_positions = bucket_scores.max(dim=1)
            valid = torch.isfinite(best_scores) & (best_scores > torch.finfo(scores.dtype).min / 2)
            selected_positions.append(torch.where(valid, best_positions, torch.zeros_like(best_positions)))
            selected_masks.append(~valid)
        return torch.stack(selected_positions, dim=1), torch.stack(selected_masks, dim=1)

    def _memory_event_scores(
        self,
        sequence: torch.Tensor,
        time_buckets: torch.Tensor | None,
        prefix_lengths: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, _feature_count, max_len = sequence.shape
        active_features = sequence > 0
        active_density = active_features.float().mean(dim=1)
        active_events = active_features.any(dim=1)

        positions = torch.arange(max_len, device=sequence.device)
        prefix_mask = positions.unsqueeze(0) < prefix_lengths.unsqueeze(1)
        recency = positions.float().unsqueeze(0) / prefix_lengths.clamp_min(1).float().unsqueeze(1)

        time_valid = sequence.new_zeros((batch_size, max_len), dtype=torch.float32)
        if time_buckets is not None and time_buckets.shape[1] > 0:
            used_width = min(max_len, int(time_buckets.shape[1]))
            time_valid[:, :used_width] = time_buckets[:, :used_width].to(sequence.device).gt(0).float()

        duplicate = sequence.new_zeros((batch_size, max_len), dtype=torch.float32)
        if max_len > 1:
            previous_equal = (sequence[:, :, 1:] == sequence[:, :, :-1]).all(dim=1)
            duplicate[:, 1:] = (previous_equal & active_events[:, 1:]).float()

        scores = (
            self.v3_memory_density_weight * active_density
            + self.v3_memory_time_weight * time_valid
            + self.v3_memory_recency_weight * recency
            - self.v3_memory_duplicate_penalty * duplicate
        )
        valid = prefix_mask & active_events
        return scores.masked_fill(~valid, torch.finfo(scores.dtype).min)


__all__ = [
    "ROLE_CANDIDATE",
    "ROLE_CLS",
    "ROLE_COUNT",
    "ROLE_DENSE",
    "ROLE_ITEM",
    "ROLE_MISSING",
    "ROLE_SEQUENCE",
    "ROLE_STATS",
    "ROLE_USER",
    "UnifiedSymbiosisTokenizer",
    "UnifiedTokenBatch",
]
