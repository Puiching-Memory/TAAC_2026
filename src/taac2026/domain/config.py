"""Typed configuration objects for PCVR experiment packages."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from taac2026.infrastructure.runtime.execution import (
    BinaryClassificationLossConfig,
    DENSE_OPTIMIZER_TYPE_CHOICES,
    RuntimeExecutionConfig,
)


SeqEncoderType = Literal["swiglu", "transformer", "longer"]
RankMixerMode = Literal["full", "ffn_only", "none"]
NSTokenizerType = Literal["group", "rankmixer"]
NSGroupingStrategy = Literal["explicit", "singleton"]
PCVRSeqWindowMode = Literal["tail", "random_tail", "rolling"]
PCVRDataCacheMode = Literal["none", "memory", "opt"]
DenseOptimizerType = Literal["adamw", "fused_adamw", "orthogonal_adamw", "muon"]
DenseLRSchedulerType = Literal["none", "linear", "cosine"]
RMSNormBackend = Literal["torch", "tilelang"]
FlashAttentionBackend = Literal["torch", "tilelang"]


DENSE_LR_SCHEDULER_TYPE_CHOICES = ("none", "linear", "cosine")


def _normalize_ns_group_map(groups: Mapping[str, Sequence[int]]) -> dict[str, list[int]]:
    return {
        str(group_name): [int(feature_id) for feature_id in feature_ids]
        for group_name, feature_ids in groups.items()
    }


@dataclass(frozen=True, slots=True)
class PCVRDataConfig:
    batch_size: int = 256
    num_workers: int = 16
    buffer_batches: int = 20
    train_ratio: float = 1.0
    valid_ratio: float = 0.1
    eval_every_n_steps: int = 0
    seq_max_lens: str = "seq_a:256,seq_b:256,seq_c:512,seq_d:512"


@dataclass(frozen=True, slots=True)
class PCVRSequenceCropConfig:
    name: Literal["sequence_crop"] = "sequence_crop"
    enabled: bool = True
    views_per_row: int = 1
    seq_window_mode: PCVRSeqWindowMode = "tail"
    seq_window_min_len: int = 1


@dataclass(frozen=True, slots=True)
class PCVRFeatureMaskConfig:
    name: Literal["feature_mask"] = "feature_mask"
    enabled: bool = True
    probability: float = 0.0


@dataclass(frozen=True, slots=True)
class PCVRDomainDropoutConfig:
    name: Literal["domain_dropout"] = "domain_dropout"
    enabled: bool = True
    probability: float = 0.0


@dataclass(frozen=True, slots=True)
class PCVRDataCacheConfig:
    mode: PCVRDataCacheMode = "none"
    max_batches: int = 0

    @property
    def enabled(self) -> bool:
        return self.mode in {"memory", "opt"}


PCVRDataTransformConfig = (
    PCVRSequenceCropConfig | PCVRFeatureMaskConfig | PCVRDomainDropoutConfig
)


@dataclass(frozen=True, slots=True)
class PCVRDataPipelineConfig:
    cache: PCVRDataCacheConfig = field(default_factory=PCVRDataCacheConfig)
    transforms: tuple[PCVRDataTransformConfig, ...] = ()
    seed: int | None = None
    strict_time_filter: bool = True

    @property
    def enabled(self) -> bool:
        return any(transform.enabled for transform in self.transforms)

    @property
    def transform_names(self) -> tuple[str, ...]:
        return tuple(
            transform.name for transform in self.transforms if transform.enabled
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "cache": {
                "mode": self.cache.mode,
                "max_batches": self.cache.max_batches,
            },
            "seed": self.seed,
            "strict_time_filter": self.strict_time_filter,
            "transforms": [
                _data_transform_config_to_dict(transform)
                for transform in self.transforms
            ],
        }

    def to_flat_dict(self) -> dict[str, Any]:
        return {
            "data_pipeline": self.to_dict(),
        }


def _data_transform_config_to_dict(
    transform: PCVRDataTransformConfig,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "name": transform.name,
        "enabled": transform.enabled,
    }
    if isinstance(transform, PCVRSequenceCropConfig):
        payload.update(
            {
                "views_per_row": transform.views_per_row,
                "seq_window_mode": transform.seq_window_mode,
                "seq_window_min_len": transform.seq_window_min_len,
            }
        )
    elif isinstance(transform, (PCVRFeatureMaskConfig, PCVRDomainDropoutConfig)):
        payload["probability"] = transform.probability
    return payload


@dataclass(frozen=True, slots=True)
class PCVROptimizerConfig:
    lr: float = 1e-4
    max_steps: int = 0
    patience: int = 5
    seed: int = 42
    device: str | None = None
    dense_optimizer_type: DenseOptimizerType = "adamw"
    scheduler_type: DenseLRSchedulerType = "none"
    warmup_steps: int = 0
    min_lr_ratio: float = 0.0

    def __post_init__(self) -> None:
        if self.dense_optimizer_type not in DENSE_OPTIMIZER_TYPE_CHOICES:
            raise ValueError(f"unsupported dense optimizer type: {self.dense_optimizer_type}")
        if self.scheduler_type not in DENSE_LR_SCHEDULER_TYPE_CHOICES:
            raise ValueError(f"unsupported scheduler type: {self.scheduler_type}")
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative")
        if not 0.0 <= self.min_lr_ratio <= 1.0:
            raise ValueError("min_lr_ratio must be between 0.0 and 1.0")


@dataclass(frozen=True, slots=True)
class PCVRSparseOptimizerConfig:
    sparse_lr: float = 0.05
    sparse_weight_decay: float = 0.0
    reinit_sparse_every_n_steps: int = 0
    reinit_cardinality_threshold: int = 0


@dataclass(frozen=True, slots=True)
class PCVRModelConfig:
    d_model: int = 64
    emb_dim: int = 64
    num_queries: int = 1
    num_blocks: int = 2
    num_heads: int = 4
    seq_encoder_type: SeqEncoderType = "transformer"
    hidden_mult: int = 4
    dropout_rate: float = 0.01
    seq_top_k: int = 50
    seq_causal: bool = False
    action_num: int = 1
    use_time_buckets: bool = True
    rank_mixer_mode: RankMixerMode = "full"
    use_rope: bool = False
    rope_base: float = 10000.0
    emb_skip_threshold: int = 0
    seq_id_threshold: int = 10000
    gradient_checkpointing: bool = False
    flash_attention_backend: FlashAttentionBackend = "torch"
    rms_norm_backend: RMSNormBackend = "torch"
    rms_norm_block_rows: int = 1

    def __post_init__(self) -> None:
        if self.flash_attention_backend not in {"torch", "tilelang"}:
            raise ValueError(f"unsupported flash attention backend: {self.flash_attention_backend}")
        if self.rms_norm_backend not in {"torch", "tilelang"}:
            raise ValueError(f"unsupported rms_norm backend: {self.rms_norm_backend}")
        if self.rms_norm_block_rows < 1:
            raise ValueError("rms_norm_block_rows must be positive")


@dataclass(frozen=True, slots=True)
class PCVRNSConfig:
    grouping_strategy: NSGroupingStrategy = "explicit"
    metadata: dict[str, str] = field(default_factory=dict)
    user_groups: dict[str, list[int]] = field(default_factory=dict)
    item_groups: dict[str, list[int]] = field(default_factory=dict)
    tokenizer_type: NSTokenizerType = "rankmixer"
    user_tokens: int = 0
    item_tokens: int = 0

    def to_flat_dict(self) -> dict[str, Any]:
        return {
            "ns_grouping_strategy": self.grouping_strategy,
            "user_ns_groups": _normalize_ns_group_map(self.user_groups),
            "item_ns_groups": _normalize_ns_group_map(self.item_groups),
            "ns_tokenizer_type": self.tokenizer_type,
            "user_ns_tokens": self.user_tokens,
            "item_ns_tokens": self.item_tokens,
        }


@dataclass(frozen=True, slots=True)
class PCVRTrainConfig:
    data: PCVRDataConfig = field(default_factory=PCVRDataConfig)
    data_pipeline: PCVRDataPipelineConfig = field(
        default_factory=PCVRDataPipelineConfig
    )
    optimizer: PCVROptimizerConfig = field(default_factory=PCVROptimizerConfig)
    runtime: RuntimeExecutionConfig = field(default_factory=RuntimeExecutionConfig)
    loss: BinaryClassificationLossConfig = field(
        default_factory=BinaryClassificationLossConfig
    )
    sparse_optimizer: PCVRSparseOptimizerConfig = field(
        default_factory=PCVRSparseOptimizerConfig
    )
    model: PCVRModelConfig = field(default_factory=PCVRModelConfig)
    ns: PCVRNSConfig = field(default_factory=PCVRNSConfig)

    def to_flat_dict(self) -> dict[str, Any]:
        return {
            "batch_size": self.data.batch_size,
            "num_workers": self.data.num_workers,
            "buffer_batches": self.data.buffer_batches,
            "train_ratio": self.data.train_ratio,
            "valid_ratio": self.data.valid_ratio,
            "eval_every_n_steps": self.data.eval_every_n_steps,
            "seq_max_lens": self.data.seq_max_lens,
            **self.data_pipeline.to_flat_dict(),
            "lr": self.optimizer.lr,
            "max_steps": self.optimizer.max_steps,
            "patience": self.optimizer.patience,
            "seed": self.optimizer.seed,
            "device": self.optimizer.device,
            "dense_optimizer_type": self.optimizer.dense_optimizer_type,
            "scheduler_type": self.optimizer.scheduler_type,
            "warmup_steps": self.optimizer.warmup_steps,
            "min_lr_ratio": self.optimizer.min_lr_ratio,
            "amp": self.runtime.amp,
            "amp_dtype": self.runtime.amp_dtype,
            "compile": self.runtime.compile,
            "loss_type": self.loss.loss_type,
            "focal_alpha": self.loss.focal_alpha,
            "focal_gamma": self.loss.focal_gamma,
            "pairwise_auc_weight": self.loss.pairwise_auc_weight,
            "pairwise_auc_temperature": self.loss.pairwise_auc_temperature,
            "sparse_lr": self.sparse_optimizer.sparse_lr,
            "sparse_weight_decay": self.sparse_optimizer.sparse_weight_decay,
            "reinit_sparse_every_n_steps": self.sparse_optimizer.reinit_sparse_every_n_steps,
            "reinit_cardinality_threshold": self.sparse_optimizer.reinit_cardinality_threshold,
            "d_model": self.model.d_model,
            "emb_dim": self.model.emb_dim,
            "num_queries": self.model.num_queries,
            "num_blocks": self.model.num_blocks,
            "num_heads": self.model.num_heads,
            "seq_encoder_type": self.model.seq_encoder_type,
            "hidden_mult": self.model.hidden_mult,
            "dropout_rate": self.model.dropout_rate,
            "seq_top_k": self.model.seq_top_k,
            "seq_causal": self.model.seq_causal,
            "action_num": self.model.action_num,
            "use_time_buckets": self.model.use_time_buckets,
            "rank_mixer_mode": self.model.rank_mixer_mode,
            "use_rope": self.model.use_rope,
            "rope_base": self.model.rope_base,
            "emb_skip_threshold": self.model.emb_skip_threshold,
            "seq_id_threshold": self.model.seq_id_threshold,
            "gradient_checkpointing": self.model.gradient_checkpointing,
            "flash_attention_backend": self.model.flash_attention_backend,
            "rms_norm_backend": self.model.rms_norm_backend,
            "rms_norm_block_rows": self.model.rms_norm_block_rows,
            **self.ns.to_flat_dict(),
        }
REQUIRED_PCVR_TRAIN_CONFIG_KEYS = frozenset(PCVRTrainConfig().to_flat_dict())
