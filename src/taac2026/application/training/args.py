"""Shared PCVR model training entrypoint."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, fields, make_dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Annotated, Any, Literal

import torch
import tyro

from taac2026.domain.config import (
    DenseLRSchedulerType,
    DenseOptimizerType,
    FlashAttentionBackend,
    NSTokenizerType,
    PCVREarlyStoppingMetric,
    PCVRDataSamplingStrategy,
    PCVRDataSplitStrategy,
    PCVRTrainConfig,
    PCVRValidationProbeMode,
    RMSNormBackend,
    RankMixerMode,
    SeqEncoderType,
)
from taac2026.infrastructure.modeling.model_contract import resolve_schema_path
from taac2026.domain.runtime_config import PCVRLossConfig, RuntimeExecutionConfig
from taac2026.infrastructure.io.files import write_json
from taac2026.infrastructure.logging import logger
from taac2026.application.training.workflow import (
    PCVRTrainContext,
    PCVRTrainHooks,
)
from taac2026.infrastructure.runtime.telemetry import RuntimeTelemetry
from taac2026.infrastructure.runtime.execution import (
    create_logger,
    runtime_execution_summary,
    set_seed,
)


AMPDType = Literal["bfloat16", "float16"]


def _arg(*aliases: str) -> object:
    return tyro.conf.arg(aliases=tuple(aliases) or None)


def _hyphen_alias(name: str) -> str:
    return f"--{name.replace('_', '-')}"


@dataclass(frozen=True, slots=True)
class PCVRTrainCLIArgs:
    data_dir: Annotated[str | None, _arg("--data-dir")] = None
    schema_path: Annotated[str | None, _arg("--schema-path")] = None
    ckpt_dir: Annotated[str | None, _arg("--ckpt-dir")] = None
    log_dir: Annotated[str | None, _arg("--log-dir")] = None
    tf_events_dir: Annotated[str | None, _arg("--tf-events-dir")] = None

    batch_size: Annotated[int, _arg("--batch-size")] = 256
    lr: float = 1e-4
    max_steps: Annotated[int, _arg("--max-steps")] = 0
    patience_steps: Annotated[int, _arg("--patience-steps")] = 25_000
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dense_optimizer_type: Annotated[DenseOptimizerType, _arg("--dense-optimizer-type")] = "adamw"
    scheduler_type: Annotated[DenseLRSchedulerType, _arg("--scheduler-type")] = "none"
    warmup_steps: Annotated[int, _arg("--warmup-steps")] = 0
    min_lr_ratio: Annotated[float, _arg("--min-lr-ratio")] = 0.0
    ema_enabled: Annotated[bool, _arg("--ema-enabled")] = False
    ema_decay: Annotated[float, _arg("--ema-decay")] = 0.999
    ema_start_step: Annotated[int, _arg("--ema-start-step")] = 0
    ema_update_every_n_steps: Annotated[int, _arg("--ema-update-every-n-steps")] = 1
    amp: bool = False
    amp_dtype: Annotated[AMPDType, _arg("--amp-dtype")] = "bfloat16"
    compile: bool = False
    progress_log_interval_steps: Annotated[int, _arg("--progress-log-interval-steps")] = 100
    deterministic: bool = True

    num_workers: Annotated[int, _arg("--num-workers")] = 16
    buffer_batches: Annotated[int, _arg("--buffer-batches")] = 1
    train_steps_per_sweep: Annotated[int, _arg("--train-steps-per-sweep")] = 0
    train_ratio: Annotated[float, _arg("--train-ratio")] = 1.0
    valid_ratio: Annotated[float, _arg("--valid-ratio")] = 0.1
    split_strategy: Annotated[PCVRDataSplitStrategy, _arg("--split-strategy")] = "row_group_tail"
    sampling_strategy: Annotated[PCVRDataSamplingStrategy, _arg("--sampling-strategy")] = "step_random"
    eval_every_n_steps: Annotated[int, _arg("--eval-every-n-steps")] = 5_000
    seq_max_lens: Annotated[str, _arg("--seq-max-lens")] = "seq_a:256,seq_b:256,seq_c:512,seq_d:512"
    validation_probe_mode: Annotated[PCVRValidationProbeMode, _arg("--validation-probe-mode")] = "none"
    early_stopping_metric: Annotated[PCVREarlyStoppingMetric, _arg("--early-stopping-metric")] = "auc"

    d_model: Annotated[int, _arg("--d-model")] = 64
    emb_dim: Annotated[int, _arg("--emb-dim")] = 64
    num_queries: Annotated[int, _arg("--num-queries")] = 1
    num_blocks: Annotated[int, _arg("--num-blocks")] = 2
    num_heads: Annotated[int, _arg("--num-heads")] = 4
    seq_encoder_type: Annotated[SeqEncoderType, _arg("--seq-encoder-type")] = "transformer"
    hidden_mult: Annotated[int, _arg("--hidden-mult")] = 4
    dropout_rate: Annotated[float, _arg("--dropout-rate")] = 0.01
    seq_top_k: Annotated[int, _arg("--seq-top-k")] = 50
    seq_causal: Annotated[bool, _arg("--seq-causal")] = False
    action_num: Annotated[int, _arg("--action-num")] = 1
    use_time_buckets: Annotated[bool, _arg("--use-time-buckets")] = True
    rank_mixer_mode: Annotated[RankMixerMode, _arg("--rank-mixer-mode")] = "full"
    use_rope: Annotated[bool, _arg("--use-rope")] = False
    rope_base: Annotated[float, _arg("--rope-base")] = 10000.0
    emb_skip_threshold: Annotated[int, _arg("--emb-skip-threshold")] = 0
    seq_id_threshold: Annotated[int, _arg("--seq-id-threshold")] = 10000
    gradient_checkpointing: Annotated[bool, _arg("--gradient-checkpointing")] = False
    flash_attention_backend: Annotated[FlashAttentionBackend, _arg("--flash-attention-backend")] = "torch"
    rms_norm_backend: Annotated[RMSNormBackend, _arg("--rms-norm-backend")] = "torch"
    rms_norm_block_rows: Annotated[int, _arg("--rms-norm-block-rows")] = 1

    loss_terms: Annotated[str | None, _arg("--loss-terms")] = None
    loss_weight_overrides: Annotated[str, _arg("--loss-weight-overrides")] = ""

    sparse_lr: Annotated[float, _arg("--sparse-lr")] = 0.05
    sparse_weight_decay: Annotated[float, _arg("--sparse-weight-decay")] = 0.0
    reinit_sparse_every_n_steps: Annotated[int, _arg("--reinit-sparse-every-n-steps")] = 0
    reinit_cardinality_threshold: Annotated[int, _arg("--reinit-cardinality-threshold")] = 0

    ns_tokenizer_type: Annotated[NSTokenizerType, _arg("--ns-tokenizer-type")] = "rankmixer"
    user_ns_tokens: Annotated[int, _arg("--user-ns-tokens")] = 0
    item_ns_tokens: Annotated[int, _arg("--item-ns-tokens")] = 0


_PATH_CLI_FIELD_NAMES = frozenset({"data_dir", "schema_path", "ckpt_dir", "log_dir", "tf_events_dir"})
_SPECIAL_CLI_FIELD_NAMES = _PATH_CLI_FIELD_NAMES | frozenset({"loss_terms", "loss_weight_overrides"})
_BASE_CLI_FIELD_NAMES = frozenset(item.name for item in fields(PCVRTrainCLIArgs))


def _flat_config_value_type(default: Any) -> type[Any] | object:
    if isinstance(default, bool):
        return bool
    if isinstance(default, int):
        return int
    if isinstance(default, float):
        return float
    if isinstance(default, str):
        return str
    if default is None:
        return str | None
    raise TypeError(f"unsupported flat config default: {type(default).__name__}")


def _flat_config_field(name: str, default: Any) -> tuple[str, object, Any]:
    aliases = (_hyphen_alias(name),) if "_" in name else ()
    return (
        name,
        Annotated[_flat_config_value_type(default), _arg(*aliases)],
        field(default=default),
    )


def _pcvr_train_cli_type(extra_config_defaults: Mapping[str, Any] | None) -> type[PCVRTrainCLIArgs]:
    if not extra_config_defaults:
        return PCVRTrainCLIArgs
    duplicates = sorted(set(extra_config_defaults).intersection(_BASE_CLI_FIELD_NAMES))
    if duplicates:
        joined = ", ".join(duplicates)
        raise ValueError(f"extra PCVR train CLI config duplicates built-in field(s): {joined}")
    return make_dataclass(
        "PCVRTrainCLIArgsWithExtraConfig",
        [_flat_config_field(name, default) for name, default in extra_config_defaults.items()],
        bases=(PCVRTrainCLIArgs,),
        frozen=True,
        slots=True,
    )


def _pcvr_train_cli_default(
    parser_type: type[PCVRTrainCLIArgs],
    *,
    defaults: PCVRTrainConfig,
    extra_config_defaults: Mapping[str, Any] | None,
) -> PCVRTrainCLIArgs:
    flat_defaults = defaults.to_flat_dict()
    kwargs: dict[str, Any] = {}
    for item in fields(PCVRTrainCLIArgs):
        if item.name in _SPECIAL_CLI_FIELD_NAMES:
            continue
        if item.name == "device":
            kwargs[item.name] = flat_defaults[item.name] or ("cuda" if torch.cuda.is_available() else "cpu")
            continue
        kwargs[item.name] = flat_defaults[item.name]
    kwargs.update(extra_config_defaults or {})
    return parser_type(**kwargs)


def _parse_pcvr_train_cli_args(
    argv: Sequence[str] | None,
    *,
    defaults: PCVRTrainConfig,
    extra_config_defaults: Mapping[str, Any] | None = None,
) -> SimpleNamespace:
    parser_type = _pcvr_train_cli_type(extra_config_defaults)
    parsed = tyro.cli(
        parser_type,
        description="Train a PCVR experiment",
        args=argv,
        default=_pcvr_train_cli_default(
            parser_type,
            defaults=defaults,
            extra_config_defaults=extra_config_defaults,
        ),
        use_underscores=True,
    )
    return SimpleNamespace(**asdict(parsed))


@dataclass(slots=True)
class TyroFlatArgParser:
    defaults: PCVRTrainConfig
    extra_config_defaults: dict[str, Any] = field(default_factory=dict)

    def add_flat_config_arguments(self, defaults: Mapping[str, Any]) -> None:
        requested = set(defaults)
        duplicates = sorted(
            requested.intersection(_BASE_CLI_FIELD_NAMES)
            | requested.intersection(self.extra_config_defaults)
        )
        if duplicates:
            joined = ", ".join(duplicates)
            raise ValueError(f"duplicate flat CLI config field(s): {joined}")
        for name, default in defaults.items():
            _flat_config_value_type(default)
            self.extra_config_defaults[name] = default

    def parse_args(self, argv: Sequence[str] | None = None) -> SimpleNamespace:
        return _parse_pcvr_train_cli_args(
            argv,
            defaults=self.defaults,
            extra_config_defaults=self.extra_config_defaults,
        )


def add_flat_config_arguments(parser: Any, defaults: Mapping[str, Any]) -> None:
    add_flat_defaults = getattr(parser, "add_flat_config_arguments", None)
    if not callable(add_flat_defaults):
        raise TypeError("parser does not support typed flat config arguments")
    add_flat_defaults(defaults)


def _coerce_flat_config_value(value: Any, default: Any) -> Any:
    if isinstance(default, bool):
        return bool(value)
    if isinstance(default, int):
        return int(value)
    if isinstance(default, float):
        return float(value)
    if isinstance(default, str):
        return str(value)
    return value


def resolve_flat_config_values(
    config: Mapping[str, Any],
    defaults: Mapping[str, Any],
    *,
    config_name: str = "config",
) -> dict[str, Any]:
    missing_keys = sorted(key for key in defaults if key not in config)
    if missing_keys:
        joined = ", ".join(missing_keys)
        raise KeyError(f"{config_name} is missing required key(s): {joined}")

    return {
        key: _coerce_flat_config_value(config[key], default)
        for key, default in defaults.items()
    }


def build_pcvr_train_arg_parser(
    *,
    package_dir: Path,
    defaults: PCVRTrainConfig,
) -> TyroFlatArgParser:
    del package_dir
    return TyroFlatArgParser(defaults=defaults)


def apply_pcvr_train_arg_env_overrides(args: SimpleNamespace) -> SimpleNamespace:
    args.data_dir = os.environ.get("TRAIN_DATA_PATH", args.data_dir)
    args.schema_path = os.environ.get("TAAC_SCHEMA_PATH", args.schema_path)
    args.ckpt_dir = os.environ.get("TRAIN_CKPT_PATH", args.ckpt_dir)
    args.log_dir = os.environ.get("TRAIN_LOG_PATH", args.log_dir)
    args.tf_events_dir = os.environ.get("TRAIN_TF_EVENTS_PATH", args.tf_events_dir)
    return args


def apply_pcvr_train_non_cli_defaults(
    args: SimpleNamespace,
    *,
    defaults: PCVRTrainConfig,
) -> SimpleNamespace:
    ns_defaults = defaults.ns.to_flat_dict()
    args.ns_grouping_strategy = ns_defaults["ns_grouping_strategy"]
    args.user_ns_groups = ns_defaults["user_ns_groups"]
    args.item_ns_groups = ns_defaults["item_ns_groups"]
    if hasattr(args, "loss_terms"):
        loss_terms = defaults.loss if args.loss_terms is None else args.loss_terms
        loss_config = PCVRLossConfig.from_value(loss_terms).with_weight_overrides(args.loss_weight_overrides)
        args.loss_terms = loss_config.to_list()
    return args


def parse_pcvr_train_args(
    argv: Sequence[str] | None = None,
    *,
    package_dir: Path,
    defaults: PCVRTrainConfig,
) -> SimpleNamespace:
    parser = build_pcvr_train_arg_parser(package_dir=package_dir, defaults=defaults)
    args = parser.parse_args(argv)
    args = apply_pcvr_train_arg_env_overrides(args)
    return apply_pcvr_train_non_cli_defaults(args, defaults=defaults)


def _required_path(value: str | None, name: str) -> Path:
    if not value:
        raise ValueError(f"{name} is required")
    return Path(value).expanduser().resolve()


def train_pcvr_model(
    *,
    model_module: Any,
    model_class_name: str,
    package_dir: Path,
    defaults: PCVRTrainConfig,
    arg_parser: Any,
    train_hooks: PCVRTrainHooks,
    argv: Sequence[str] | None = None,
) -> dict[str, Any]:
    args = arg_parser(argv, package_dir=package_dir, defaults=defaults)
    data_dir = _required_path(args.data_dir, "data_dir")
    ckpt_dir = _required_path(args.ckpt_dir, "ckpt_dir")
    log_dir = _required_path(args.log_dir, "log_dir")
    tf_events_dir = _required_path(args.tf_events_dir, "tf_events_dir")

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    tf_events_dir.mkdir(parents=True, exist_ok=True)

    deterministic = bool(getattr(args, "deterministic", defaults.runtime.deterministic))
    set_seed(args.seed, deterministic=deterministic)
    create_logger(log_dir / "train.log")
    config = vars(args).copy()
    data_pipeline_config = defaults.data_pipeline
    config.update(data_pipeline_config.to_flat_dict())
    logger.info("Args: {}", config)
    runtime_execution = RuntimeExecutionConfig(
        amp=bool(args.amp),
        amp_dtype=str(args.amp_dtype),
        compile=bool(args.compile),
        progress_log_interval_steps=int(
            getattr(args, "progress_log_interval_steps", defaults.runtime.progress_log_interval_steps)
        ),
        deterministic=deterministic,
    )
    logger.info(
        "Resolved PCVR training runtime: {}", runtime_execution_summary(runtime_execution, args.device)
    )

    reporter = None
    telemetry = RuntimeTelemetry(
        label="training",
        device=args.device,
        metadata={
            "model_class": model_class_name,
            "package_dir": str(package_dir),
            "run_dir": str(ckpt_dir),
        },
    ).start()
    try:
        schema_path = resolve_schema_path(
            data_dir,
            Path(args.schema_path) if args.schema_path else None,
            ckpt_dir,
        )
        context = PCVRTrainContext(
            model_module=model_module,
            model_class_name=model_class_name,
            package_dir=package_dir,
            defaults=defaults,
            args=args,
            config=config,
            data_dir=data_dir,
            ckpt_dir=ckpt_dir,
            log_dir=log_dir,
            tf_events_dir=tf_events_dir,
            schema_path=schema_path,
            runtime_execution=runtime_execution,
            reporter=None,
        )
        reporter = train_hooks.build_reporter(context)
        context = PCVRTrainContext(
            model_module=context.model_module,
            model_class_name=context.model_class_name,
            package_dir=context.package_dir,
            defaults=context.defaults,
            args=context.args,
            config=context.config,
            data_dir=context.data_dir,
            ckpt_dir=context.ckpt_dir,
            log_dir=context.log_dir,
            tf_events_dir=context.tf_events_dir,
            schema_path=context.schema_path,
            runtime_execution=context.runtime_execution,
            reporter=reporter,
        )
        data_bundle = train_hooks.build_data(context)
        model = train_hooks.build_model(context, data_bundle)
        trainer = train_hooks.build_trainer(context, data_bundle, model)
        train_hooks.run_training(context, trainer)
        summary = dict(train_hooks.build_summary(context, trainer) or {})
        train_rows = int(getattr(data_bundle.dataset, "num_rows", 0) or 0)
        valid_dataset = getattr(data_bundle.valid_loader, "dataset", None)
        valid_rows = int(getattr(valid_dataset, "num_rows", 0) or 0)
        summary["telemetry"] = telemetry.finish(
            steps=int(getattr(trainer, "optim_step", 0) or 0),
            train_rows=train_rows,
            valid_rows=valid_rows,
            rows=train_rows,
            model_parameters=int(sum(parameter.numel() for parameter in model.parameters()))
            if hasattr(model, "parameters")
            else 0,
        )
        write_json(ckpt_dir / "training_telemetry.json", summary["telemetry"])
        write_json(ckpt_dir / "training_summary.json", summary)
    finally:
        if reporter is not None:
            reporter.close()

    logger.info("Training complete!")
    return summary
