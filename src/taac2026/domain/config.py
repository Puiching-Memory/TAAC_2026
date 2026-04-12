from __future__ import annotations

from dataclasses import dataclass, field


DEFAULT_SEQUENCE_NAMES = ("action_seq", "content_seq", "item_seq")
DEFAULT_MAX_PARAMETER_BYTES = 3 * 1024 * 1024 * 1024
DEFAULT_MAX_END_TO_END_INFERENCE_SECONDS = 180.0


@dataclass(slots=True)
class DataConfig:
    dataset_path: str
    max_seq_len: int = 64
    max_feature_tokens: int = 24
    max_event_features: int = 4
    stream_batch_rows: int = 1024
    val_ratio: float = 0.2
    label_action_type: int = 2
    sequence_names: tuple[str, ...] = DEFAULT_SEQUENCE_NAMES
    dense_feature_dim: int = 16


@dataclass(slots=True)
class ModelConfig:
    name: str
    vocab_size: int
    embedding_dim: int
    hidden_dim: int
    dropout: float = 0.1
    num_layers: int = 2
    num_heads: int = 4
    recent_seq_len: int = 16
    memory_slots: int = 0
    ffn_multiplier: float = 4.0
    feature_cross_layers: int = 0
    sequence_layers: int = 0
    static_layers: int = 0
    query_decoder_layers: int = 0
    fusion_layers: int = 0
    num_queries: int = 1
    head_hidden_dim: int | None = None
    segment_count: int = 4
    attention_dropout: float = 0.0


@dataclass(slots=True)
class TrainConfig:
    seed: int = 7
    epochs: int = 1
    batch_size: int = 32
    eval_batch_size: int | None = None
    num_workers: int = 0
    output_dir: str = "outputs/default"
    learning_rate: float = 1.0e-3
    weight_decay: float = 1.0e-4
    grad_clip_norm: float = 1.0
    pairwise_weight: float = 0.0
    latency_warmup_steps: int = 2
    latency_measure_steps: int = 8
    device: str | None = None
    enable_torch_compile: bool = False
    torch_compile_backend: str | None = None
    torch_compile_mode: str | None = None
    enable_amp: bool = False
    amp_dtype: str = "float16"
    switches: dict[str, bool] = field(default_factory=dict)

    @property
    def resolved_eval_batch_size(self) -> int:
        return self.eval_batch_size or self.batch_size


@dataclass(slots=True)
class SearchConfig:
    n_trials: int = 20
    timeout_seconds: int | None = None
    metric_name: str = "best_val_auc"
    direction: str = "maximize"
    sampler_seed: int | None = None
    max_parameter_bytes: int = DEFAULT_MAX_PARAMETER_BYTES
    max_end_to_end_inference_seconds: float = DEFAULT_MAX_END_TO_END_INFERENCE_SECONDS


__all__ = [
    "DEFAULT_MAX_END_TO_END_INFERENCE_SECONDS",
    "DEFAULT_MAX_PARAMETER_BYTES",
    "DEFAULT_SEQUENCE_NAMES",
    "DataConfig",
    "ModelConfig",
    "SearchConfig",
    "TrainConfig",
]
