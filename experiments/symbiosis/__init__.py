"""Symbiosis PCVR experiment package."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from taac2026.api import (
    PCVRDataCacheConfig,
    PCVRDataConfig,
    PCVRDataPipelineConfig,
    PCVRDomainDropoutConfig,
    PCVRFeatureMaskConfig,
    PCVRLossConfig,
    PCVRLossTermConfig,
    PCVRModelConfig,
    PCVRNSConfig,
    PCVRNonSequentialSparseDropoutConfig,
    PCVROptimizerConfig,
    PCVRSparseOptimizerConfig,
    PCVRTrainConfig,
    PCVRValidationConfig,
)
from taac2026.api import create_pcvr_experiment
from taac2026.application.evaluation.workflow import (
    PCVRPredictionContext,
    PCVRPredictionDataBundle,
)
from taac2026.infrastructure.modeling.model_contract import build_pcvr_model, load_ns_groups
from taac2026.infrastructure.logging import logger
from taac2026.application.evaluation.runtime import (
    default_load_train_config,
)
from taac2026.application.training.workflow import (
    PCVRTrainContext,
    PCVRTrainDataBundle,
)
from taac2026.application.training.args import (
    add_flat_config_arguments,
    apply_pcvr_train_arg_env_overrides,
    apply_pcvr_train_non_cli_defaults,
    build_pcvr_train_arg_parser,
    resolve_flat_config_values,
)
from taac2026.api import RuntimeExecutionConfig


@dataclass(frozen=True, slots=True)
class SymbiosisModelDefaults:
    use_dense_packets: bool = True
    use_sequence_memory: bool = True
    use_compressed_memory: bool = True
    use_candidate_token: bool = True
    use_item_prior: bool = True
    use_domain_type: bool = True
    use_cross_token: bool = True
    use_global_token: bool = True
    sparse_seed: int = 20260511
    memory_block_size: int = 32
    memory_top_k: int = 8
    recent_tokens: int = 32
    sequence_recent_output_tokens: int = 12
    sequence_memory_output_tokens: int = 8
    use_sequence_temporal_encoder: bool = True
    ns_tokenizer_mode: str = "group"
    use_structured_fusion: bool = True
    cross_token_count: int = 6
    compile_fusion_core: bool = True
    shortcut_dropout_rate: float = 0.08
    compress_large_ids: bool = True
    use_missing_signals: bool = True
    use_sequence_stats: bool = True

    def to_flat_dict(self) -> dict[str, Any]:
        return {
            "symbiosis_use_dense_packets": self.use_dense_packets,
            "symbiosis_use_sequence_memory": self.use_sequence_memory,
            "symbiosis_use_compressed_memory": self.use_compressed_memory,
            "symbiosis_use_candidate_token": self.use_candidate_token,
            "symbiosis_use_item_prior": self.use_item_prior,
            "symbiosis_use_domain_type": self.use_domain_type,
            "symbiosis_use_cross_token": self.use_cross_token,
            "symbiosis_use_global_token": self.use_global_token,
            "symbiosis_sparse_seed": self.sparse_seed,
            "symbiosis_memory_block_size": self.memory_block_size,
            "symbiosis_memory_top_k": self.memory_top_k,
            "symbiosis_recent_tokens": self.recent_tokens,
            "symbiosis_sequence_recent_output_tokens": self.sequence_recent_output_tokens,
            "symbiosis_sequence_memory_output_tokens": self.sequence_memory_output_tokens,
            "symbiosis_use_sequence_temporal_encoder": self.use_sequence_temporal_encoder,
            "symbiosis_ns_tokenizer_mode": self.ns_tokenizer_mode,
            "symbiosis_use_structured_fusion": self.use_structured_fusion,
            "symbiosis_cross_token_count": self.cross_token_count,
            "symbiosis_compile_fusion_core": self.compile_fusion_core,
            "symbiosis_shortcut_dropout_rate": self.shortcut_dropout_rate,
            "symbiosis_compress_large_ids": self.compress_large_ids,
            "symbiosis_use_missing_signals": self.use_missing_signals,
            "symbiosis_use_sequence_stats": self.use_sequence_stats,
        }


SYMBIOSIS_MODEL_DEFAULTS = SymbiosisModelDefaults()
SYMBIOSIS_MODEL_CONFIG_KEYS = tuple(SYMBIOSIS_MODEL_DEFAULTS.to_flat_dict())
SYMBIOSIS_OPTIONAL_MODEL_CONFIG_DEFAULTS = {
    "symbiosis_shortcut_dropout_rate": SYMBIOSIS_MODEL_DEFAULTS.shortcut_dropout_rate,
    "symbiosis_compress_large_ids": False,
    "symbiosis_use_missing_signals": False,
    "symbiosis_use_sequence_stats": False,
    "symbiosis_sequence_recent_output_tokens": 1,
    "symbiosis_sequence_memory_output_tokens": 1,
    "symbiosis_use_sequence_temporal_encoder": False,
    "symbiosis_ns_tokenizer_mode": "random_chunk",
    "symbiosis_use_structured_fusion": False,
    "symbiosis_cross_token_count": 1,
}
SYMBIOSIS_OPTIONAL_MODEL_CONFIG_KEYS = tuple(SYMBIOSIS_OPTIONAL_MODEL_CONFIG_DEFAULTS)


def _add_symbiosis_train_args(parser: Any) -> None:
    add_flat_config_arguments(parser, SYMBIOSIS_MODEL_DEFAULTS.to_flat_dict())


def parse_symbiosis_train_args(
    argv: Sequence[str] | None = None,
    *,
    package_dir: Path,
    defaults: PCVRTrainConfig,
) -> Any:
    parser = build_pcvr_train_arg_parser(package_dir=package_dir, defaults=defaults)
    _add_symbiosis_train_args(parser)
    args = apply_pcvr_train_arg_env_overrides(parser.parse_args(argv))
    return apply_pcvr_train_non_cli_defaults(args, defaults=defaults)


def _resolve_symbiosis_model_kwargs(config: Mapping[str, Any]) -> dict[str, Any]:
    defaults = SYMBIOSIS_MODEL_DEFAULTS.to_flat_dict()
    resolved_config = dict(config)
    for key, default_value in SYMBIOSIS_OPTIONAL_MODEL_CONFIG_DEFAULTS.items():
        resolved_config.setdefault(key, default_value)
    return resolve_flat_config_values(
        resolved_config,
        defaults,
        config_name="Symbiosis train_config",
    )


def _build_symbiosis_model(
    *,
    model_module: Any,
    model_class_name: str,
    data_module: Any,
    dataset: Any,
    config: Mapping[str, Any],
    package_dir: Path,
    checkpoint_dir: Path,
) -> torch.nn.Module:
    return build_pcvr_model(
        model_module=model_module,
        model_class_name=model_class_name,
        data_module=data_module,
        dataset=dataset,
        config=dict(config),
        package_dir=package_dir,
        checkpoint_dir=checkpoint_dir,
        extra_model_kwargs=_resolve_symbiosis_model_kwargs(config),
    )


def build_symbiosis_train_model(
    context: PCVRTrainContext,
    data_bundle: PCVRTrainDataBundle,
) -> torch.nn.Module:
    user_ns_groups, item_ns_groups = load_ns_groups(
        data_bundle.dataset,
        context.config,
        context.package_dir,
        context.ckpt_dir,
    )
    logger.info("User NS groups: {}", user_ns_groups)
    logger.info("Item NS groups: {}", item_ns_groups)

    model = _build_symbiosis_model(
        model_module=context.model_module,
        model_class_name=context.model_class_name,
        data_module=data_bundle.data_module,
        dataset=data_bundle.dataset,
        config=context.config,
        package_dir=context.package_dir,
        checkpoint_dir=context.ckpt_dir,
    ).to(context.args.device)

    num_ns = model.num_ns
    sequence_tokens = int(getattr(model, "num_sequence_tokens", 0))
    token_count = num_ns + sequence_tokens
    logger.info(
        "PCVR model created: class={}, num_ns={}, sequence_tokens={}, T={}, d_model={}, rank_mixer_mode={}",
        context.model_class_name,
        num_ns,
        sequence_tokens,
        token_count,
        context.args.d_model,
        context.args.rank_mixer_mode,
    )
    total_params = sum(parameter.numel() for parameter in model.parameters())
    logger.info("Total parameters: {}", f"{total_params:,}")
    return model


def build_symbiosis_prediction_model(
    context: PCVRPredictionContext,
    data_bundle: PCVRPredictionDataBundle,
) -> torch.nn.Module:
    return _build_symbiosis_model(
        model_module=context.model_module,
        model_class_name=context.model_class_name,
        data_module=data_bundle.data_module,
        dataset=data_bundle.dataset,
        config=context.config,
        package_dir=context.package_dir,
        checkpoint_dir=context.checkpoint_path.parent,
    )


def load_symbiosis_train_config(experiment: Any, checkpoint_dir: Path) -> dict[str, Any]:
    config = default_load_train_config(experiment, checkpoint_dir)
    _resolve_symbiosis_model_kwargs(config)
    return config


TRAIN_DEFAULTS = PCVRTrainConfig(
    data=PCVRDataConfig(
        batch_size=128,
        num_workers=8,
        buffer_batches=1,
        train_ratio=1.0,
        valid_ratio=0.1,
        split_strategy="row_group_tail",
        sampling_strategy="step_random",
        train_timestamp_start=0,
        train_timestamp_end=0,
        valid_timestamp_start=0,
        valid_timestamp_end=0,
        eval_every_n_steps=500,
        seq_max_lens="seq_a:256,seq_b:256,seq_c:512,seq_d:512",
    ),
    data_pipeline=PCVRDataPipelineConfig(
        cache=PCVRDataCacheConfig(mode="none", max_batches=0),
        transforms=(
            PCVRFeatureMaskConfig(probability=0.03),
            PCVRNonSequentialSparseDropoutConfig(probability=0.10),
            PCVRDomainDropoutConfig(probability=0.02),
        ),
        seed=42,
        strict_time_filter=True,
    ),
    optimizer=PCVROptimizerConfig(
        lr=1e-4,
        max_steps=100_000,
        patience_steps=2500,
        seed=42,
        device=None,
        dense_optimizer_type="muon",
        scheduler_type="none",
        warmup_steps=100,
        min_lr_ratio=0.1,
    ),
    runtime=RuntimeExecutionConfig(
        amp=True,
        amp_dtype="bfloat16",
        compile=True,
        progress_log_interval_steps=100,
    ),
    loss=PCVRLossConfig(terms=(PCVRLossTermConfig(name="bce", kind="bce", weight=1.0),)),
    sparse_optimizer=PCVRSparseOptimizerConfig(
        sparse_lr=0.01,
        sparse_weight_decay=0.0,
        reinit_sparse_every_n_steps=0,
        reinit_cardinality_threshold=0,
    ),
    model=PCVRModelConfig(
        d_model=256,
        emb_dim=256,
        num_queries=1,
        num_blocks=4,
        num_heads=8,
        seq_encoder_type="transformer",
        hidden_mult=4,
        dropout_rate=0.04,
        seq_top_k=64,
        seq_causal=False,
        action_num=1,
        use_time_buckets=True,
        rank_mixer_mode="full",
        use_rope=False,
        rope_base=10000.0,
        emb_skip_threshold=1_000_000,
        seq_id_threshold=10000,
        gradient_checkpointing=False,
    ),
    validation=PCVRValidationConfig(
        probe_mode="none",
        early_stopping_metric="auc",
    ),
    ns=PCVRNSConfig(
        # NS token groups for parquet data. Values are fids, using the numeric suffix
        # from user_int_feats_{fid} or item_int_feats_{fid}; runtime converts them to
        # schema entry indices at load time. For the official schema, num_ns is
        # 7 user-int groups + 1 user-dense token + 4 item-int groups = 12 before
        # model-specific tokenizer changes. User dense features are projected to one
        # dense NS token and are not configured in this mapping. Fids
        # 62/63/64/65/66/89/90/91 also appear in user_dense_feats; the int parts are
        # grouped here, while the float parts stay in the single user_dense NS token.
        grouping_strategy="explicit",
        user_groups={
            "U1": [1, 15],
            "U2": [48, 49, 89, 90, 91],
            "U3": [80],
            "U4": [51, 52, 53, 54, 86],
            "U5": [82, 92, 93],
            "U6": [50, 60, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
            "U7": [3, 4, 55, 56, 57, 58, 59, 62, 63, 64, 65, 66],
        },
        item_groups={
            "I1": [11, 13],
            "I2": [5, 6, 7, 8, 12],
            "I3": [16, 81, 83, 84, 85],
            "I4": [9, 10],
        },
        tokenizer_type="rankmixer",
        user_tokens=7,
        item_tokens=4,
    ),
)

EXPERIMENT = create_pcvr_experiment(
    name="pcvr_symbiosis",
    package_dir=Path(__file__).resolve().parent,
    model_class_name="PCVRSymbiosis",
    train_defaults=TRAIN_DEFAULTS,
    train_arg_parser=parse_symbiosis_train_args,
    train_hook_overrides={"build_model": build_symbiosis_train_model},
    prediction_hook_overrides={"build_model": build_symbiosis_prediction_model},
    runtime_hook_overrides={"load_train_config": load_symbiosis_train_config},
)
TRAIN_HOOKS = EXPERIMENT.train_hooks
PREDICTION_HOOKS = EXPERIMENT.prediction_hooks
RUNTIME_HOOKS = EXPERIMENT.runtime_hooks

__all__ = [
    "EXPERIMENT",
    "PREDICTION_HOOKS",
    "RUNTIME_HOOKS",
    "SYMBIOSIS_MODEL_DEFAULTS",
    "TRAIN_DEFAULTS",
    "TRAIN_HOOKS",
    "build_symbiosis_prediction_model",
    "build_symbiosis_train_model",
    "load_symbiosis_train_config",
    "parse_symbiosis_train_args",
]
