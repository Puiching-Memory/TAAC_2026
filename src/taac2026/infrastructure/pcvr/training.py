"""Shared PCVR model training entrypoint."""

from __future__ import annotations

import argparse
import logging
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

import taac2026.infrastructure.pcvr.data as pcvr_data
from taac2026.infrastructure.pcvr.config import (
    DEFAULT_PCVR_TRAIN_CONFIG,
    PCVRTrainConfig,
)
from taac2026.infrastructure.pcvr.protocol import (
    build_pcvr_model,
    load_ns_groups,
    parse_seq_max_lens,
    resolve_ns_groups_path,
    resolve_schema_path,
)
from taac2026.infrastructure.pcvr.trainer import PCVRPointwiseTrainer
from taac2026.infrastructure.training.runtime import (
    AMP_DTYPE_CHOICES,
    BINARY_LOSS_TYPE_CHOICES,
    DENSE_OPTIMIZER_TYPE_CHOICES,
    EarlyStopping,
    RuntimeExecutionConfig,
    create_logger,
    set_seed,
)


def parse_pcvr_train_args(
    argv: Sequence[str] | None = None,
    *,
    package_dir: Path,
    defaults: PCVRTrainConfig = DEFAULT_PCVR_TRAIN_CONFIG,
) -> argparse.Namespace:
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
    parser.add_argument("--num_epochs", type=int, default=default_values["num_epochs"])
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
        "--reinit_sparse_after_epoch",
        type=int,
        default=default_values["reinit_sparse_after_epoch"],
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

    parser.add_argument("--ns_groups_json", default=default_values["ns_groups_json"])
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

    parser.add_argument(
        "--symbiosis_use_user_item_graph",
        "--symbiosis-use-user-item-graph",
        action=argparse.BooleanOptionalAction,
        default=default_values["symbiosis_use_user_item_graph"],
    )
    parser.add_argument(
        "--symbiosis_use_fourier_time",
        "--symbiosis-use-fourier-time",
        action=argparse.BooleanOptionalAction,
        default=default_values["symbiosis_use_fourier_time"],
    )
    parser.add_argument(
        "--symbiosis_use_context_exchange",
        "--symbiosis-use-context-exchange",
        action=argparse.BooleanOptionalAction,
        default=default_values["symbiosis_use_context_exchange"],
    )
    parser.add_argument(
        "--symbiosis_use_multi_scale",
        "--symbiosis-use-multi-scale",
        action=argparse.BooleanOptionalAction,
        default=default_values["symbiosis_use_multi_scale"],
    )
    parser.add_argument(
        "--symbiosis_use_domain_gate",
        "--symbiosis-use-domain-gate",
        action=argparse.BooleanOptionalAction,
        default=default_values["symbiosis_use_domain_gate"],
    )
    parser.add_argument(
        "--symbiosis_use_candidate_decoder",
        "--symbiosis-use-candidate-decoder",
        action=argparse.BooleanOptionalAction,
        default=default_values["symbiosis_use_candidate_decoder"],
    )
    parser.add_argument(
        "--symbiosis_use_action_conditioning",
        "--symbiosis-use-action-conditioning",
        action=argparse.BooleanOptionalAction,
        default=default_values["symbiosis_use_action_conditioning"],
    )
    parser.add_argument(
        "--symbiosis_use_compressed_memory",
        "--symbiosis-use-compressed-memory",
        action=argparse.BooleanOptionalAction,
        default=default_values["symbiosis_use_compressed_memory"],
    )
    parser.add_argument(
        "--symbiosis_use_attention_sink",
        "--symbiosis-use-attention-sink",
        action=argparse.BooleanOptionalAction,
        default=default_values["symbiosis_use_attention_sink"],
    )
    parser.add_argument(
        "--symbiosis_use_lane_mixing",
        "--symbiosis-use-lane-mixing",
        action=argparse.BooleanOptionalAction,
        default=default_values["symbiosis_use_lane_mixing"],
    )
    parser.add_argument(
        "--symbiosis_use_semantic_id",
        "--symbiosis-use-semantic-id",
        action=argparse.BooleanOptionalAction,
        default=default_values["symbiosis_use_semantic_id"],
    )
    parser.add_argument(
        "--symbiosis_memory_block_size",
        "--symbiosis-memory-block-size",
        dest="symbiosis_memory_block_size",
        type=int,
        default=default_values["symbiosis_memory_block_size"],
    )
    parser.add_argument(
        "--symbiosis_memory_top_k",
        "--symbiosis-memory-top-k",
        dest="symbiosis_memory_top_k",
        type=int,
        default=default_values["symbiosis_memory_top_k"],
    )
    parser.add_argument(
        "--symbiosis_recent_tokens",
        "--symbiosis-recent-tokens",
        dest="symbiosis_recent_tokens",
        type=int,
        default=default_values["symbiosis_recent_tokens"],
    )

    args = parser.parse_args(argv)
    args.data_dir = os.environ.get("TRAIN_DATA_PATH", args.data_dir)
    args.schema_path = os.environ.get("TRAIN_SCHEMA_PATH", args.schema_path)
    args.ckpt_dir = os.environ.get("TRAIN_CKPT_PATH", args.ckpt_dir)
    args.log_dir = os.environ.get("TRAIN_LOG_PATH", args.log_dir)
    args.tf_events_dir = os.environ.get("TRAIN_TF_EVENTS_PATH", args.tf_events_dir)
    return args


def _required_path(value: str | None, name: str) -> Path:
    if not value:
        raise ValueError(f"{name} is required")
    return Path(value).expanduser().resolve()


def train_pcvr_model(
    *,
    model_module: Any,
    model_class_name: str,
    package_dir: Path,
    defaults: PCVRTrainConfig = DEFAULT_PCVR_TRAIN_CONFIG,
    argv: Sequence[str] | None = None,
) -> dict[str, Any]:
    args = parse_pcvr_train_args(argv, package_dir=package_dir, defaults=defaults)
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

        seq_max_lens = parse_seq_max_lens(str(args.seq_max_lens))
        if seq_max_lens:
            logging.info("Seq max_lens override: %s", seq_max_lens)

        logging.info("Using PCVR Parquet data pipeline")
        train_loader, valid_loader, pcvr_dataset = pcvr_data.get_pcvr_data(
            data_dir=str(data_dir),
            schema_path=str(schema_path),
            batch_size=args.batch_size,
            valid_ratio=args.valid_ratio,
            train_ratio=args.train_ratio,
            num_workers=args.num_workers,
            buffer_batches=args.buffer_batches,
            seed=args.seed,
            seq_max_lens=seq_max_lens,
            data_pipeline_config=data_pipeline_config,
        )

        user_ns_groups, item_ns_groups = load_ns_groups(
            pcvr_dataset, config, package_dir, ckpt_dir
        )
        logging.info("User NS groups: %s", user_ns_groups)
        logging.info("Item NS groups: %s", item_ns_groups)

        model = build_pcvr_model(
            model_module=model_module,
            model_class_name=model_class_name,
            data_module=pcvr_data,
            dataset=pcvr_dataset,
            config=config,
            package_dir=package_dir,
            checkpoint_dir=ckpt_dir,
        ).to(args.device)

        num_sequences = len(pcvr_dataset.seq_domains)
        num_ns = model.num_ns
        token_count = args.num_queries * num_sequences + num_ns
        logging.info(
            "PCVR model created: class=%s, num_ns=%s, T=%s, d_model=%s, rank_mixer_mode=%s",
            model_class_name,
            num_ns,
            token_count,
            args.d_model,
            args.rank_mixer_mode,
        )
        total_params = sum(parameter.numel() for parameter in model.parameters())
        logging.info("Total parameters: %s", f"{total_params:,}")

        early_stopping = EarlyStopping(
            checkpoint_path=ckpt_dir / "placeholder" / "model.pt",
            patience=args.patience,
            label="model",
        )
        checkpoint_params = {
            "blocks": args.num_blocks,
            "head": args.num_heads,
            "hidden": args.d_model,
        }
        resolved_ns_groups_path = resolve_ns_groups_path(
            str(args.ns_groups_json), package_dir, ckpt_dir
        )
        trainer = PCVRPointwiseTrainer(
            model=model,
            model_input_type=model_module.ModelInput,
            train_loader=train_loader,
            valid_loader=valid_loader,
            lr=args.lr,
            num_epochs=args.num_epochs,
            device=args.device,
            save_dir=ckpt_dir,
            early_stopping=early_stopping,
            dense_optimizer_type=args.dense_optimizer_type,
            loss_type=args.loss_type,
            focal_alpha=args.focal_alpha,
            focal_gamma=args.focal_gamma,
            pairwise_auc_weight=args.pairwise_auc_weight,
            pairwise_auc_temperature=args.pairwise_auc_temperature,
            sparse_lr=args.sparse_lr,
            sparse_weight_decay=args.sparse_weight_decay,
            reinit_sparse_after_epoch=args.reinit_sparse_after_epoch,
            reinit_cardinality_threshold=args.reinit_cardinality_threshold,
            ckpt_params=checkpoint_params,
            writer=writer,
            schema_path=schema_path,
            ns_groups_path=resolved_ns_groups_path,
            eval_every_n_steps=args.eval_every_n_steps,
            train_config=config,
            runtime_execution=runtime_execution,
        )
        trainer.train()
    finally:
        writer.close()

    logging.info("Training complete!")
    return {
        "run_dir": str(ckpt_dir),
        "checkpoint_root": str(ckpt_dir),
    }
