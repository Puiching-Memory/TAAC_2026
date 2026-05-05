"""Composable PCVR training stack hooks."""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import torch

import taac2026.infrastructure.data.dataset as pcvr_data
from taac2026.domain.config import PCVRTrainConfig
from taac2026.domain.model_contract import (
    build_pcvr_model,
    load_ns_groups,
    parse_seq_max_lens,
)
from taac2026.infrastructure.runtime.trainer import PCVRPointwiseTrainer
from taac2026.infrastructure.runtime.execution import EarlyStopping, RuntimeExecutionConfig
from taac2026.infrastructure.checkpoints import preferred_checkpoint_path


@dataclass(slots=True)
class PCVRTrainContext:
    model_module: Any
    model_class_name: str
    package_dir: Path
    defaults: PCVRTrainConfig
    args: Any
    config: dict[str, Any]
    data_dir: Path
    ckpt_dir: Path
    log_dir: Path
    tf_events_dir: Path
    schema_path: Path
    runtime_execution: RuntimeExecutionConfig
    writer: Any

    @property
    def data_pipeline_config(self):
        return self.defaults.data_pipeline


@dataclass(frozen=True, slots=True)
class PCVRTrainDataBundle:
    train_loader: Any
    valid_loader: Any
    dataset: Any
    data_module: Any = pcvr_data


def default_build_train_data(context: PCVRTrainContext) -> PCVRTrainDataBundle:
    seq_max_lens = parse_seq_max_lens(str(context.args.seq_max_lens))
    if seq_max_lens:
        logging.info("Seq max_lens override: %s", seq_max_lens)

    logging.info("Using PCVR train data pipeline: %s.get_pcvr_data", pcvr_data.__name__)
    train_loader, valid_loader, dataset = pcvr_data.get_pcvr_data(
        data_dir=str(context.data_dir),
        schema_path=str(context.schema_path),
        batch_size=context.args.batch_size,
        valid_ratio=context.args.valid_ratio,
        train_ratio=context.args.train_ratio,
        num_workers=context.args.num_workers,
        buffer_batches=context.args.buffer_batches,
        seed=context.args.seed,
        seq_max_lens=seq_max_lens,
        data_pipeline_config=context.data_pipeline_config,
    )
    return PCVRTrainDataBundle(
        train_loader=train_loader,
        valid_loader=valid_loader,
        dataset=dataset,
    )


def default_build_train_model(
    context: PCVRTrainContext,
    data_bundle: PCVRTrainDataBundle,
) -> torch.nn.Module:
    user_ns_groups, item_ns_groups = load_ns_groups(
        data_bundle.dataset,
        context.config,
        context.package_dir,
        context.ckpt_dir,
    )
    logging.info("User NS groups: %s", user_ns_groups)
    logging.info("Item NS groups: %s", item_ns_groups)

    model = build_pcvr_model(
        model_module=context.model_module,
        model_class_name=context.model_class_name,
        data_module=data_bundle.data_module,
        dataset=data_bundle.dataset,
        config=context.config,
        package_dir=context.package_dir,
        checkpoint_dir=context.ckpt_dir,
    ).to(context.args.device)

    num_sequences = len(data_bundle.dataset.seq_domains)
    num_ns = model.num_ns
    token_count = context.args.num_queries * num_sequences + num_ns
    logging.info(
        "PCVR model created: class=%s, num_ns=%s, T=%s, d_model=%s, rank_mixer_mode=%s",
        context.model_class_name,
        num_ns,
        token_count,
        context.args.d_model,
        context.args.rank_mixer_mode,
    )
    total_params = sum(parameter.numel() for parameter in model.parameters())
    logging.info("Total parameters: %s", f"{total_params:,}")
    return model


def default_build_train_trainer(
    context: PCVRTrainContext,
    data_bundle: PCVRTrainDataBundle,
    model: torch.nn.Module,
) -> Any:
    early_stopping = EarlyStopping(
        checkpoint_path=preferred_checkpoint_path(context.ckpt_dir),
        patience=context.args.patience,
        label="model",
        patience_unit="steps",
    )
    checkpoint_params = {
        "blocks": context.args.num_blocks,
        "head": context.args.num_heads,
        "hidden": context.args.d_model,
    }
    return PCVRPointwiseTrainer(
        model=model,
        model_input_type=context.model_module.ModelInput,
        train_loader=data_bundle.train_loader,
        valid_loader=data_bundle.valid_loader,
        lr=context.args.lr,
        max_steps=context.args.max_steps,
        device=context.args.device,
        save_dir=context.ckpt_dir,
        early_stopping=early_stopping,
        dense_optimizer_type=context.args.dense_optimizer_type,
        scheduler_type=context.args.scheduler_type,
        warmup_steps=context.args.warmup_steps,
        min_lr_ratio=context.args.min_lr_ratio,
        loss_type=context.args.loss_type,
        focal_alpha=context.args.focal_alpha,
        focal_gamma=context.args.focal_gamma,
        pairwise_auc_weight=context.args.pairwise_auc_weight,
        pairwise_auc_temperature=context.args.pairwise_auc_temperature,
        sparse_lr=context.args.sparse_lr,
        sparse_weight_decay=context.args.sparse_weight_decay,
        reinit_sparse_every_n_steps=context.args.reinit_sparse_every_n_steps,
        reinit_cardinality_threshold=context.args.reinit_cardinality_threshold,
        ckpt_params=checkpoint_params,
        writer=context.writer,
        schema_path=context.schema_path,
        eval_every_n_steps=context.args.eval_every_n_steps,
        train_config=context.config,
        runtime_execution=context.runtime_execution,
    )


def default_run_training(context: PCVRTrainContext, trainer: Any) -> None:
    del context
    trainer.train()


def default_build_train_summary(
    context: PCVRTrainContext,
    trainer: Any,
) -> dict[str, Any]:
    del trainer
    return {
        "run_dir": str(context.ckpt_dir),
        "checkpoint_root": str(context.ckpt_dir),
        "schema_path": str(context.schema_path),
        "train_ratio": float(context.args.train_ratio),
        "valid_ratio": float(context.args.valid_ratio),
    }


@dataclass(frozen=True, slots=True)
class PCVRTrainHooks:
    build_data: Any = default_build_train_data
    build_model: Any = default_build_train_model
    build_trainer: Any = default_build_train_trainer
    run_training: Any = default_run_training
    build_summary: Any = default_build_train_summary


DEFAULT_PCVR_TRAIN_HOOKS = PCVRTrainHooks()


def build_pcvr_train_hooks(**overrides: Any) -> PCVRTrainHooks:
    return replace(DEFAULT_PCVR_TRAIN_HOOKS, **overrides)


__all__ = [
    "DEFAULT_PCVR_TRAIN_HOOKS",
    "PCVRTrainContext",
    "PCVRTrainDataBundle",
    "PCVRTrainHooks",
    "build_pcvr_train_hooks",
    "default_build_train_data",
    "default_build_train_model",
    "default_build_train_summary",
    "default_build_train_trainer",
    "default_run_training",
]