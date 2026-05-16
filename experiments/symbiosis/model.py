"""Symbiosis V2/V3: unified token-stream PCVR model."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
import torch.nn as nn

from taac2026.api import (
    EmbeddingParameterMixin,
    ModelInput,
    RMSNorm,
    choose_num_heads,
    configure_rms_norm_runtime as _configure_rms_norm_runtime,
    maybe_gradient_checkpoint,
)

try:
    from .attention import MetadataAttentionMask
    from .backbone import UnifiedInteractionBlock, UnifiedSelfAttention
    from .pooling import CandidateClsPooler
    from .tokenization import UnifiedSymbiosisTokenizer, UnifiedTokenBatch
except ImportError:  # pragma: no cover - direct file loading in contract tests
    _PACKAGE_DIR = Path(__file__).resolve().parent
    if str(_PACKAGE_DIR) not in sys.path:
        sys.path.insert(0, str(_PACKAGE_DIR))
    from attention import MetadataAttentionMask
    from backbone import UnifiedInteractionBlock, UnifiedSelfAttention
    from pooling import CandidateClsPooler
    from tokenization import UnifiedSymbiosisTokenizer, UnifiedTokenBatch


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


class PCVRSymbiosis(EmbeddingParameterMixin, nn.Module):
    """Distribution-aware unified token-stream model for PCVR."""

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
        symbiosis_v2_use_dense_tokens: bool = True,
        symbiosis_v2_use_missing_tokens: bool = True,
        symbiosis_v2_use_sequence_stats_tokens: bool = True,
        symbiosis_v2_use_metadata_attention_bias: bool = True,
        symbiosis_v2_use_candidate_readout: bool = True,
        symbiosis_v2_tokenization_mode: str = "group",
        symbiosis_v2_sparse_seed: int = 20260512,
        symbiosis_v2_recent_event_tokens: int = 16,
        symbiosis_v2_memory_event_tokens: int = 8,
        symbiosis_v2_user_dense_tokens: int = 3,
        symbiosis_v2_item_dense_tokens: int = 1,
        symbiosis_v2_user_missing_tokens: int = 2,
        symbiosis_v2_item_missing_tokens: int = 1,
        symbiosis_v2_high_risk_token_dropout_rate: float = 0.08,
        symbiosis_v2_compress_large_ids: bool = True,
        symbiosis_v2_compile_backbone: bool = True,
        symbiosis_v3_enabled: bool = False,
        symbiosis_v3_memory_selection_mode: str = "quality_stratified",
        symbiosis_v3_recent_event_tokens_by_domain: str = "seq_a:8,seq_b:8,seq_c:20,seq_d:24",
        symbiosis_v3_memory_event_tokens_by_domain: str = "seq_a:4,seq_b:4,seq_c:10,seq_d:12",
        symbiosis_v3_memory_density_weight: float = 1.0,
        symbiosis_v3_memory_time_weight: float = 0.30,
        symbiosis_v3_memory_recency_weight: float = 0.20,
        symbiosis_v3_memory_duplicate_penalty: float = 0.50,
    ) -> None:
        super().__init__()
        del num_queries, seq_encoder_type, seq_top_k, seq_causal, rank_mixer_mode, use_rope, rope_base, ns_tokenizer_type, symbiosis_v2_sparse_seed
        num_heads = choose_num_heads(d_model, num_heads)
        self.d_model = int(d_model)
        self.action_num = int(action_num)
        self.gradient_checkpointing = bool(gradient_checkpointing)
        self.symbiosis_v2_compile_backbone = bool(symbiosis_v2_compile_backbone)
        self.symbiosis_v2_high_risk_token_dropout_rate = min(1.0, max(0.0, float(symbiosis_v2_high_risk_token_dropout_rate)))
        self.tokenizer = UnifiedSymbiosisTokenizer(
            user_int_feature_specs=user_int_feature_specs,
            item_int_feature_specs=item_int_feature_specs,
            user_dense_dim=user_dense_dim,
            item_dense_dim=item_dense_dim,
            seq_vocab_sizes=seq_vocab_sizes,
            user_ns_groups=user_ns_groups,
            item_ns_groups=item_ns_groups,
            d_model=d_model,
            emb_dim=emb_dim,
            user_ns_tokens=user_ns_tokens,
            item_ns_tokens=item_ns_tokens,
            emb_skip_threshold=emb_skip_threshold,
            seq_id_threshold=seq_id_threshold,
            num_time_buckets=num_time_buckets,
            tokenization_mode=symbiosis_v2_tokenization_mode,
            recent_event_tokens=symbiosis_v2_recent_event_tokens,
            memory_event_tokens=symbiosis_v2_memory_event_tokens,
            user_dense_tokens=symbiosis_v2_user_dense_tokens,
            item_dense_tokens=symbiosis_v2_item_dense_tokens,
            user_missing_tokens=symbiosis_v2_user_missing_tokens,
            item_missing_tokens=symbiosis_v2_item_missing_tokens,
            compress_large_ids=symbiosis_v2_compress_large_ids,
            use_dense_tokens=symbiosis_v2_use_dense_tokens,
            use_missing_tokens=symbiosis_v2_use_missing_tokens,
            use_sequence_stats_tokens=symbiosis_v2_use_sequence_stats_tokens,
            v3_enabled=symbiosis_v3_enabled,
            v3_memory_selection_mode=symbiosis_v3_memory_selection_mode,
            v3_recent_event_tokens_by_domain=symbiosis_v3_recent_event_tokens_by_domain,
            v3_memory_event_tokens_by_domain=symbiosis_v3_memory_event_tokens_by_domain,
            v3_memory_density_weight=symbiosis_v3_memory_density_weight,
            v3_memory_time_weight=symbiosis_v3_memory_time_weight,
            v3_memory_recency_weight=symbiosis_v3_memory_recency_weight,
            v3_memory_duplicate_penalty=symbiosis_v3_memory_duplicate_penalty,
        )
        self.num_ns = self.tokenizer.num_non_sequence_tokens
        self.num_sequence_tokens = self.tokenizer.num_sequence_tokens
        self.attention_mask = MetadataAttentionMask(enabled=symbiosis_v2_use_metadata_attention_bias)
        self.blocks = nn.ModuleList(
            [UnifiedInteractionBlock(d_model, num_heads, hidden_mult, dropout_rate) for _ in range(max(1, int(num_blocks)))]
        )
        self.final_norm = RMSNorm(d_model)
        self.pooler = CandidateClsPooler(d_model, use_candidate_readout=symbiosis_v2_use_candidate_readout)
        self.classifier = nn.Sequential(
            nn.Linear(self.pooler.output_dim, d_model),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(d_model, action_num),
        )
        self._compiled_backbone = None
        self._backbone_compiled = False
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
            f"SymbiosisV2/{metric_name}/{clean_phase}": value
            for metric_name, value in self._diagnostic_scalars.items()
        }
        self._diagnostic_scalars = {}
        return scalars

    @property
    def uses_internal_compile(self) -> bool:
        return self.symbiosis_v2_compile_backbone

    def prepare_for_runtime_compile(self) -> None:
        if not self.symbiosis_v2_compile_backbone or self._backbone_compiled:
            return
        self._compiled_backbone = torch.compile(self._run_backbone)
        self._backbone_compiled = True

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

    def _apply_high_risk_dropout(self, batch: UnifiedTokenBatch) -> UnifiedTokenBatch:
        if not self.training or not torch.is_grad_enabled() or self.symbiosis_v2_high_risk_token_dropout_rate <= 0.0:
            return batch
        risk_positions = (batch.risk_ids > 0).unsqueeze(0).expand(batch.tokens.shape[0], -1)
        if not bool(risk_positions.any()):
            return batch
        keep = torch.rand(batch.tokens.shape[:2], device=batch.tokens.device) >= self.symbiosis_v2_high_risk_token_dropout_rate
        drop_mask = risk_positions & ~keep
        tokens = batch.tokens * (~drop_mask).to(batch.tokens.dtype).unsqueeze(-1)
        padding_mask = batch.padding_mask | drop_mask
        return batch._replace(tokens=tokens, padding_mask=padding_mask)

    def _run_backbone(self, tokens: torch.Tensor, padding_mask: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            tokens, _attn_tokens = maybe_gradient_checkpoint(
                block,
                tokens,
                padding_mask,
                attention_mask,
                enabled=self.gradient_checkpointing,
            )
        return self.final_norm(tokens)

    def _embed(self, inputs: ModelInput) -> torch.Tensor:
        batch = self._apply_high_risk_dropout(self.tokenizer(inputs))
        attention_mask = self.attention_mask(batch)
        runner = self._compiled_backbone if self._compiled_backbone is not None else self._run_backbone
        tokens = runner(batch.tokens, batch.padding_mask, attention_mask)
        embedding = self.pooler(tokens, batch)
        self._put_scalar("tokens/active_ratio", (~batch.padding_mask).float().mean())
        self._put_scalar("tokens/count", float(batch.tokens.shape[1]))
        self._put_scalar("tokens/high_risk_ratio", (batch.risk_ids > 0).float().mean())
        self._put_scalar("embedding/norm_mean", embedding.detach().float().norm(dim=-1).mean())
        return embedding

    def forward(self, inputs: ModelInput) -> torch.Tensor:
        return self.classifier(self._embed(inputs))

    def predict(self, inputs: ModelInput) -> tuple[torch.Tensor, torch.Tensor]:
        embeddings = self._embed(inputs)
        return self.classifier(embeddings), embeddings


__all__ = ["ModelInput", "PCVRSymbiosis", "UnifiedSelfAttention", "configure_rms_norm_runtime"]
