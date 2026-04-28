"""Typed configuration objects for PCVR experiment packages."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from taac2026.infrastructure.training.runtime import (
    BinaryClassificationLossConfig,
    DENSE_OPTIMIZER_TYPE_CHOICES,
    RuntimeExecutionConfig,
)


SeqEncoderType = Literal["swiglu", "transformer", "longer"]
RankMixerMode = Literal["full", "ffn_only", "none"]
NSTokenizerType = Literal["group", "rankmixer"]
PCVRSeqWindowMode = Literal["tail", "random_tail", "rolling"]
PCVRDataCacheMode = Literal["none", "memory"]
DenseOptimizerType = Literal["adamw", "orthogonal_adamw"]


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
        return self.mode == "memory"


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
    num_epochs: int = 999
    patience: int = 5
    seed: int = 42
    device: str | None = None
    dense_optimizer_type: DenseOptimizerType = "adamw"

    def __post_init__(self) -> None:
        if self.dense_optimizer_type not in DENSE_OPTIMIZER_TYPE_CHOICES:
            raise ValueError(f"unsupported dense optimizer type: {self.dense_optimizer_type}")


@dataclass(frozen=True, slots=True)
class PCVRSparseOptimizerConfig:
    sparse_lr: float = 0.05
    sparse_weight_decay: float = 0.0
    reinit_sparse_after_epoch: int = 1
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


@dataclass(frozen=True, slots=True)
class PCVRNSConfig:
    groups_json: str = "ns_groups.json"
    tokenizer_type: NSTokenizerType = "rankmixer"
    user_tokens: int = 0
    item_tokens: int = 0


@dataclass(frozen=True, slots=True)
class PCVRSymbiosisConfig:
    use_user_item_graph: bool = True
    use_fourier_time: bool = True
    use_context_exchange: bool = True
    use_multi_scale: bool = True
    use_domain_gate: bool = True
    use_candidate_decoder: bool = True
    use_action_conditioning: bool = True
    use_compressed_memory: bool = True
    use_attention_sink: bool = True
    use_lane_mixing: bool = True
    use_semantic_id: bool = True
    memory_block_size: int = 16
    memory_top_k: int = 8
    recent_tokens: int = 64


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
    symbiosis: PCVRSymbiosisConfig = field(default_factory=PCVRSymbiosisConfig)

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
            "num_epochs": self.optimizer.num_epochs,
            "patience": self.optimizer.patience,
            "seed": self.optimizer.seed,
            "device": self.optimizer.device,
            "dense_optimizer_type": self.optimizer.dense_optimizer_type,
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
            "reinit_sparse_after_epoch": self.sparse_optimizer.reinit_sparse_after_epoch,
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
            "ns_groups_json": self.ns.groups_json,
            "ns_tokenizer_type": self.ns.tokenizer_type,
            "user_ns_tokens": self.ns.user_tokens,
            "item_ns_tokens": self.ns.item_tokens,
            "symbiosis_use_user_item_graph": self.symbiosis.use_user_item_graph,
            "symbiosis_use_fourier_time": self.symbiosis.use_fourier_time,
            "symbiosis_use_context_exchange": self.symbiosis.use_context_exchange,
            "symbiosis_use_multi_scale": self.symbiosis.use_multi_scale,
            "symbiosis_use_domain_gate": self.symbiosis.use_domain_gate,
            "symbiosis_use_candidate_decoder": self.symbiosis.use_candidate_decoder,
            "symbiosis_use_action_conditioning": self.symbiosis.use_action_conditioning,
            "symbiosis_use_compressed_memory": self.symbiosis.use_compressed_memory,
            "symbiosis_use_attention_sink": self.symbiosis.use_attention_sink,
            "symbiosis_use_lane_mixing": self.symbiosis.use_lane_mixing,
            "symbiosis_use_semantic_id": self.symbiosis.use_semantic_id,
            "symbiosis_memory_block_size": self.symbiosis.memory_block_size,
            "symbiosis_memory_top_k": self.symbiosis.memory_top_k,
            "symbiosis_recent_tokens": self.symbiosis.recent_tokens,
        }


DEFAULT_PCVR_TRAIN_CONFIG = PCVRTrainConfig()
REQUIRED_PCVR_TRAIN_CONFIG_KEYS = frozenset(DEFAULT_PCVR_TRAIN_CONFIG.to_flat_dict())
