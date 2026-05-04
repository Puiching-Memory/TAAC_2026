"""Shared PCVR model training entrypoint."""

from __future__ import annotations

import argparse
import logging
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

from taac2026.infrastructure.pcvr.config import DENSE_LR_SCHEDULER_TYPE_CHOICES, PCVRTrainConfig
from taac2026.infrastructure.pcvr.protocol import resolve_schema_path
from taac2026.infrastructure.pcvr.train_stack import (
    PCVRTrainContext,
    PCVRTrainHooks,
)
from taac2026.infrastructure.training.runtime import (
    AMP_DTYPE_CHOICES,
    BINARY_LOSS_TYPE_CHOICES,
    DENSE_OPTIMIZER_TYPE_CHOICES,
    RuntimeExecutionConfig,
    create_logger,
    set_seed,
)


def build_pcvr_train_arg_parser(
    *,
    package_dir: Path,
    defaults: PCVRTrainConfig,
) -> argparse.ArgumentParser:
    del package_dir
    default_values = defaults.to_flat_dict()
    parser = argparse.ArgumentParser(description="Train a PCVR experiment")

    parser.add_argument("--data_dir", "--data-dir", dest="data_dir", default=None)
    parser.add_argument(
        "--schema_path", "--schema-path", dest="schema_path", default=None
    )
    parser.add_argument("--ckpt_dir", "--ckpt-dir", dest="ckpt_dir", default=None)
    parser.add_argument("--log_dir", "--log-dir", dest="log_dir", default=None)
    parser.add_argument(
        "--tf_events_dir", "--tf-events-dir", dest="tf_events_dir", default=None
    )

    parser.add_argument("--batch_size", type=int, default=default_values["batch_size"])
    parser.add_argument("--lr", type=float, default=default_values["lr"])
    parser.add_argument("--max_steps", type=int, default=default_values["max_steps"])
    parser.add_argument("--patience", type=int, default=default_values["patience"])
    parser.add_argument("--seed", type=int, default=default_values["seed"])
    parser.add_argument(
        "--device",
        default=default_values["device"]
        or ("cuda" if torch.cuda.is_available() else "cpu"),
    )
    parser.add_argument(
        "--dense_optimizer_type",
        "--dense-optimizer-type",
        dest="dense_optimizer_type",
        default=default_values["dense_optimizer_type"],
        choices=DENSE_OPTIMIZER_TYPE_CHOICES,
    )
    parser.add_argument(
        "--scheduler_type",
        "--scheduler-type",
        dest="scheduler_type",
        default=default_values["scheduler_type"],
        choices=DENSE_LR_SCHEDULER_TYPE_CHOICES,
    )
    parser.add_argument(
        "--warmup_steps",
        "--warmup-steps",
        dest="warmup_steps",
        type=int,
        default=default_values["warmup_steps"],
    )
    parser.add_argument(
        "--min_lr_ratio",
        "--min-lr-ratio",
        dest="min_lr_ratio",
        type=float,
        default=default_values["min_lr_ratio"],
    )
    parser.add_argument(
        "--amp", action=argparse.BooleanOptionalAction, default=default_values["amp"]
    )
    parser.add_argument(
        "--amp_dtype",
        "--amp-dtype",
        dest="amp_dtype",
        default=default_values["amp_dtype"],
        choices=AMP_DTYPE_CHOICES,
    )
    parser.add_argument(
        "--compile",
        action=argparse.BooleanOptionalAction,
        default=default_values["compile"],
    )

    parser.add_argument(
        "--num_workers", type=int, default=default_values["num_workers"]
    )
    parser.add_argument(
        "--buffer_batches", type=int, default=default_values["buffer_batches"]
    )
    parser.add_argument(
        "--train_ratio", type=float, default=default_values["train_ratio"]
    )
    parser.add_argument(
        "--valid_ratio", type=float, default=default_values["valid_ratio"]
    )
    parser.add_argument(
        "--eval_every_n_steps", type=int, default=default_values["eval_every_n_steps"]
    )
    parser.add_argument("--seq_max_lens", default=default_values["seq_max_lens"])

    parser.add_argument("--d_model", type=int, default=default_values["d_model"])
    parser.add_argument("--emb_dim", type=int, default=default_values["emb_dim"])
    parser.add_argument(
        "--num_queries", type=int, default=default_values["num_queries"]
    )
    parser.add_argument("--num_blocks", type=int, default=default_values["num_blocks"])
    parser.add_argument("--num_heads", type=int, default=default_values["num_heads"])
    parser.add_argument(
        "--seq_encoder_type",
        default=default_values["seq_encoder_type"],
        choices=["swiglu", "transformer", "longer"],
    )
    parser.add_argument(
        "--hidden_mult", type=int, default=default_values["hidden_mult"]
    )
    parser.add_argument(
        "--dropout_rate", type=float, default=default_values["dropout_rate"]
    )
    parser.add_argument("--seq_top_k", type=int, default=default_values["seq_top_k"])
    parser.add_argument(
        "--seq_causal", action="store_true", default=default_values["seq_causal"]
    )
    parser.add_argument("--action_num", type=int, default=default_values["action_num"])
    parser.add_argument(
        "--use_time_buckets",
        action="store_true",
        default=default_values["use_time_buckets"],
    )
    parser.add_argument(
        "--no_time_buckets", dest="use_time_buckets", action="store_false"
    )
    parser.add_argument(
        "--rank_mixer_mode",
        default=default_values["rank_mixer_mode"],
        choices=["full", "ffn_only", "none"],
    )
    parser.add_argument(
        "--use_rope", action="store_true", default=default_values["use_rope"]
    )
    parser.add_argument("--rope_base", type=float, default=default_values["rope_base"])
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=default_values["gradient_checkpointing"],
    )
    parser.add_argument(
        "--rms_norm_backend",
        "--rms-norm-backend",
        dest="rms_norm_backend",
        default=default_values["rms_norm_backend"],
        choices=["torch", "tilelang"],
    )
    parser.add_argument(
        "--rms_norm_block_rows",
        "--rms-norm-block-rows",
        dest="rms_norm_block_rows",
        type=int,
        default=default_values["rms_norm_block_rows"],
    )

    parser.add_argument(
        "--loss_type",
        default=default_values["loss_type"],
        choices=BINARY_LOSS_TYPE_CHOICES,
    )
    parser.add_argument(
        "--focal_alpha", type=float, default=default_values["focal_alpha"]
    )
    parser.add_argument(
        "--focal_gamma", type=float, default=default_values["focal_gamma"]
    )
    parser.add_argument(
        "--pairwise_auc_weight",
        "--pairwise-auc-weight",
        dest="pairwise_auc_weight",
        type=float,
        default=default_values["pairwise_auc_weight"],
    )
    parser.add_argument(
        "--pairwise_auc_temperature",
        "--pairwise-auc-temperature",
        dest="pairwise_auc_temperature",
        type=float,
        default=default_values["pairwise_auc_temperature"],
    )

    parser.add_argument("--sparse_lr", type=float, default=default_values["sparse_lr"])
    parser.add_argument(
        "--sparse_weight_decay",
        type=float,
        default=default_values["sparse_weight_decay"],
    )
    parser.add_argument(
        "--reinit_sparse_every_n_steps",
        type=int,
        default=default_values["reinit_sparse_every_n_steps"],
    )
    parser.add_argument(
        "--reinit_cardinality_threshold",
        type=int,
        default=default_values["reinit_cardinality_threshold"],
    )

    parser.add_argument(
        "--emb_skip_threshold", type=int, default=default_values["emb_skip_threshold"]
    )
    parser.add_argument(
        "--seq_id_threshold", type=int, default=default_values["seq_id_threshold"]
    )

    parser.add_argument(
        "--ns_tokenizer_type",
        default=default_values["ns_tokenizer_type"],
        choices=["group", "rankmixer"],
    )
    parser.add_argument(
        "--user_ns_tokens", type=int, default=default_values["user_ns_tokens"]
    )
    parser.add_argument(
        "--item_ns_tokens", type=int, default=default_values["item_ns_tokens"]
    )

    return parser


def apply_pcvr_train_arg_env_overrides(args: argparse.Namespace) -> argparse.Namespace:
    args.data_dir = os.environ.get("TRAIN_DATA_PATH", args.data_dir)
    args.schema_path = os.environ.get("TAAC_SCHEMA_PATH", args.schema_path)
    args.ckpt_dir = os.environ.get("TRAIN_CKPT_PATH", args.ckpt_dir)
    args.log_dir = os.environ.get("TRAIN_LOG_PATH", args.log_dir)
    args.tf_events_dir = os.environ.get("TRAIN_TF_EVENTS_PATH", args.tf_events_dir)
    return args


def apply_pcvr_train_non_cli_defaults(
    args: argparse.Namespace,
    *,
    defaults: PCVRTrainConfig,
) -> argparse.Namespace:
    ns_defaults = defaults.ns.to_flat_dict()
    args.ns_grouping_strategy = ns_defaults["ns_grouping_strategy"]
    args.user_ns_groups = ns_defaults["user_ns_groups"]
    args.item_ns_groups = ns_defaults["item_ns_groups"]
    return args


def parse_pcvr_train_args(
    argv: Sequence[str] | None = None,
    *,
    package_dir: Path,
    defaults: PCVRTrainConfig,
) -> argparse.Namespace:
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

    set_seed(args.seed)
    create_logger(log_dir / "train.log")
    config = vars(args).copy()
    data_pipeline_config = defaults.data_pipeline
    config.update(data_pipeline_config.to_flat_dict())
    logging.info("Args: %s", config)
    runtime_execution = RuntimeExecutionConfig(
        amp=bool(args.amp),
        amp_dtype=str(args.amp_dtype),
        compile=bool(args.compile),
    )
    logging.info(
        "Resolved PCVR training runtime: %s", runtime_execution.summary(args.device)
    )

    from torch.utils.tensorboard import SummaryWriter

    writer = SummaryWriter(tf_events_dir)
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
            writer=writer,
        )
        data_bundle = train_hooks.build_data(context)
        model = train_hooks.build_model(context, data_bundle)
        trainer = train_hooks.build_trainer(context, data_bundle, model)
        train_hooks.run_training(context, trainer)
        summary = dict(train_hooks.build_summary(context, trainer) or {})
    finally:
        writer.close()

    logging.info("Training complete!")
    return summary
