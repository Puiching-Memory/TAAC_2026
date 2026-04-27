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
    EarlyStopping,
    RuntimeExecutionConfig,
    create_logger,
    set_seed,
)


def parse_pcvr_train_args(
    argv: Sequence[str] | None = None,
    *,
    package_dir: Path,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a PCVR experiment")

    parser.add_argument("--data_dir", "--data-dir", dest="data_dir", default=None)
    parser.add_argument("--schema_path", "--schema-path", dest="schema_path", default=None)
    parser.add_argument("--ckpt_dir", "--ckpt-dir", dest="ckpt_dir", default=None)
    parser.add_argument("--log_dir", "--log-dir", dest="log_dir", default=None)
    parser.add_argument("--tf_events_dir", "--tf-events-dir", dest="tf_events_dir", default=None)

    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_epochs", type=int, default=999)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--amp_dtype",
        "--amp-dtype",
        dest="amp_dtype",
        default="bfloat16",
        choices=AMP_DTYPE_CHOICES,
    )
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=False)

    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--buffer_batches", type=int, default=20)
    parser.add_argument("--train_ratio", type=float, default=1.0)
    parser.add_argument("--valid_ratio", type=float, default=0.1)
    parser.add_argument("--eval_every_n_steps", type=int, default=0)
    parser.add_argument("--seq_max_lens", default="seq_a:256,seq_b:256,seq_c:512,seq_d:512")

    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--emb_dim", type=int, default=64)
    parser.add_argument("--num_queries", type=int, default=1)
    parser.add_argument("--num_blocks", type=int, default=2)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument(
        "--seq_encoder_type",
        default="transformer",
        choices=["swiglu", "transformer", "longer"],
    )
    parser.add_argument("--hidden_mult", type=int, default=4)
    parser.add_argument("--dropout_rate", type=float, default=0.01)
    parser.add_argument("--seq_top_k", type=int, default=50)
    parser.add_argument("--seq_causal", action="store_true", default=False)
    parser.add_argument("--action_num", type=int, default=1)
    parser.add_argument("--use_time_buckets", action="store_true", default=True)
    parser.add_argument("--no_time_buckets", dest="use_time_buckets", action="store_false")
    parser.add_argument("--rank_mixer_mode", default="full", choices=["full", "ffn_only", "none"])
    parser.add_argument("--use_rope", action="store_true", default=False)
    parser.add_argument("--rope_base", type=float, default=10000.0)

    parser.add_argument("--loss_type", default="bce", choices=["bce", "focal"])
    parser.add_argument("--focal_alpha", type=float, default=0.1)
    parser.add_argument("--focal_gamma", type=float, default=2.0)

    parser.add_argument("--sparse_lr", type=float, default=0.05)
    parser.add_argument("--sparse_weight_decay", type=float, default=0.0)
    parser.add_argument("--reinit_sparse_after_epoch", type=int, default=1)
    parser.add_argument("--reinit_cardinality_threshold", type=int, default=0)

    parser.add_argument("--emb_skip_threshold", type=int, default=0)
    parser.add_argument("--seq_id_threshold", type=int, default=10000)

    default_ns_groups = package_dir / "ns_groups.json"
    parser.add_argument("--ns_groups_json", default=str(default_ns_groups))
    parser.add_argument("--ns_tokenizer_type", default="rankmixer", choices=["group", "rankmixer"])
    parser.add_argument("--user_ns_tokens", type=int, default=0)
    parser.add_argument("--item_ns_tokens", type=int, default=0)

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
    argv: Sequence[str] | None = None,
) -> dict[str, Any]:
    args = parse_pcvr_train_args(argv, package_dir=package_dir)
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
    logging.info("Args: %s", config)
    runtime_execution = RuntimeExecutionConfig(
        amp=bool(args.amp),
        amp_dtype=str(args.amp_dtype),
        compile=bool(args.compile),
    )
    logging.info("Resolved PCVR training runtime: %s", runtime_execution.summary(args.device))

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
        )

        user_ns_groups, item_ns_groups = load_ns_groups(pcvr_dataset, config, package_dir, ckpt_dir)
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
        resolved_ns_groups_path = resolve_ns_groups_path(str(args.ns_groups_json), package_dir, ckpt_dir)
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
            loss_type=args.loss_type,
            focal_alpha=args.focal_alpha,
            focal_gamma=args.focal_gamma,
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