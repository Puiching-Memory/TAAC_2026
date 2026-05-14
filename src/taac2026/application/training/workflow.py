"""Composable PCVR training stack hooks."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Protocol

import torch

import taac2026.infrastructure.data.dataset as pcvr_data
from taac2026.domain.config import PCVRTrainConfig
from taac2026.domain.runtime_config import RuntimeExecutionConfig
from taac2026.infrastructure.modeling.model_contract import (
    build_pcvr_model,
    load_ns_groups,
    parse_seq_max_lens,
)
from taac2026.infrastructure.logging import logger
from taac2026.infrastructure.modeling.sequence import configure_flash_attention_runtime as configure_shared_flash_attention_runtime
from taac2026.infrastructure.runtime.trainer import PCVRPointwiseTrainer
from taac2026.infrastructure.runtime.execution import EarlyStopping
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
    reporter: Any

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
        logger.info("Seq max_lens override: {}", seq_max_lens)

    logger.info("Using PCVR train data pipeline: {}.get_pcvr_data", pcvr_data.__name__)
    train_loader, valid_loader, dataset = pcvr_data.get_pcvr_data(
        data_dir=str(context.data_dir),
        schema_path=str(context.schema_path),
        batch_size=context.args.batch_size,
        valid_ratio=context.args.valid_ratio,
        train_ratio=context.args.train_ratio,
        split_strategy=getattr(context.args, "split_strategy", context.defaults.data.split_strategy),
        sampling_strategy=getattr(context.args, "sampling_strategy", context.defaults.data.sampling_strategy),
        train_steps_per_sweep=getattr(
            context.args,
            "train_steps_per_sweep",
            context.defaults.data.train_steps_per_sweep,
        ),
        train_timestamp_start=getattr(context.args, "train_timestamp_start", context.defaults.data.train_timestamp_start),
        train_timestamp_end=getattr(context.args, "train_timestamp_end", context.defaults.data.train_timestamp_end),
        valid_timestamp_start=getattr(context.args, "valid_timestamp_start", context.defaults.data.valid_timestamp_start),
        valid_timestamp_end=getattr(context.args, "valid_timestamp_end", context.defaults.data.valid_timestamp_end),
        num_workers=context.args.num_workers,
        buffer_batches=context.args.buffer_batches,
        seed=context.args.seed,
        seq_max_lens=seq_max_lens,
        data_pipeline_config=context.data_pipeline_config,
        max_steps=context.args.max_steps,
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
    logger.info("User NS groups: {}", user_ns_groups)
    logger.info("Item NS groups: {}", item_ns_groups)
    configure_shared_flash_attention_runtime(
        backend=str(context.config.get("flash_attention_backend", "torch")),
    )

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
    logger.info(
        "PCVR model created: class={}, num_ns={}, T={}, d_model={}, rank_mixer_mode={}",
        context.model_class_name,
        num_ns,
        token_count,
        context.args.d_model,
        context.args.rank_mixer_mode,
    )
    total_params = sum(parameter.numel() for parameter in model.parameters())
    logger.info("Total parameters: {}", f"{total_params:,}")
    return model


def default_build_train_trainer(
    context: PCVRTrainContext,
    data_bundle: PCVRTrainDataBundle,
    model: torch.nn.Module,
) -> Any:
    early_stopping = EarlyStopping(
        checkpoint_path=preferred_checkpoint_path(context.ckpt_dir),
        patience_steps=context.args.patience_steps,
        label="model",
    )
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
        ema_enabled=getattr(context.args, "ema_enabled", context.defaults.ema.enabled),
        ema_decay=getattr(context.args, "ema_decay", context.defaults.ema.decay),
        ema_start_step=getattr(context.args, "ema_start_step", context.defaults.ema.start_step),
        ema_update_every_n_steps=getattr(
            context.args,
            "ema_update_every_n_steps",
            context.defaults.ema.update_every_n_steps,
        ),
        loss_terms=context.args.loss_terms,
        sparse_lr=context.args.sparse_lr,
        sparse_weight_decay=context.args.sparse_weight_decay,
        reinit_sparse_every_n_steps=context.args.reinit_sparse_every_n_steps,
        reinit_cardinality_threshold=context.args.reinit_cardinality_threshold,
        reporter=context.reporter,
        schema_path=context.schema_path,
        eval_every_n_steps=context.args.eval_every_n_steps,
        validation_probe_mode=context.args.validation_probe_mode,
        early_stopping_metric=context.args.early_stopping_metric,
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
    summary = {
        "run_dir": str(context.ckpt_dir),
        "checkpoint_root": str(context.ckpt_dir),
        "schema_path": str(context.schema_path),
        "train_ratio": float(context.args.train_ratio),
        "valid_ratio": float(context.args.valid_ratio),
        "split_strategy": str(getattr(context.args, "split_strategy", context.defaults.data.split_strategy)),
        "sampling_strategy": str(getattr(context.args, "sampling_strategy", context.defaults.data.sampling_strategy)),
        "train_steps_per_sweep": int(
            getattr(
                context.args,
                "train_steps_per_sweep",
                context.defaults.data.train_steps_per_sweep,
            )
        ),
        "train_timestamp_start": int(getattr(context.args, "train_timestamp_start", context.defaults.data.train_timestamp_start)),
        "train_timestamp_end": int(getattr(context.args, "train_timestamp_end", context.defaults.data.train_timestamp_end)),
        "valid_timestamp_start": int(getattr(context.args, "valid_timestamp_start", context.defaults.data.valid_timestamp_start)),
        "valid_timestamp_end": int(getattr(context.args, "valid_timestamp_end", context.defaults.data.valid_timestamp_end)),
    }
    train_loader = getattr(trainer, "train_loader", None)
    train_dataset = getattr(train_loader, "dataset", None)
    pipeline = getattr(train_dataset, "pipeline", None)
    cache = getattr(pipeline, "cache", None)
    stats_fn = getattr(cache, "stats", None)
    if callable(stats_fn):
        summary["data_cache_stats"] = stats_fn()
    return summary


class BuildTrainDataHook(Protocol):
    def __call__(self, context: PCVRTrainContext) -> PCVRTrainDataBundle:
        ...


class BuildTrainModelHook(Protocol):
    def __call__(self, context: PCVRTrainContext, data_bundle: PCVRTrainDataBundle) -> torch.nn.Module:
        ...


class BuildTrainTrainerHook(Protocol):
    def __call__(self, context: PCVRTrainContext, data_bundle: PCVRTrainDataBundle, model: torch.nn.Module) -> Any:
        ...


class RunTrainingHook(Protocol):
    def __call__(self, context: PCVRTrainContext, trainer: Any) -> None:
        ...


class BuildTrainSummaryHook(Protocol):
    def __call__(self, context: PCVRTrainContext, trainer: Any) -> dict[str, Any] | None:
        ...


class TrainReporter(Protocol):
    def train_step(self, *, step: int, loss: float, loss_components: Mapping[str, float], dense_lr: float) -> None:
        ...

    def validation_step(
        self,
        *,
        step: int,
        auc: float,
        logloss: float,
        metrics: Mapping[str, float],
        score_diagnostics: Mapping[str, float | int],
        probe_metrics: Mapping[str, float],
        probe_score_diagnostics: Mapping[str, float | int],
    ) -> None:
        ...

    def should_collect_model_scalars(self, *, phase: str, step: int | None, trainer: Any) -> bool:
        ...

    def set_model_diagnostics_enabled(self, model: torch.nn.Module, enabled: bool) -> None:
        ...

    def consume_model_scalars(self, model: torch.nn.Module, *, phase: str) -> Mapping[str, float]:
        ...

    def model_scalars(self, *, phase: str, step: int, scalars: Mapping[str, float]) -> None:
        ...

    def close(self) -> None:
        ...


class NoopTrainReporter:
    def train_step(self, *, step: int, loss: float, loss_components: Mapping[str, float], dense_lr: float) -> None:
        pass

    def validation_step(
        self,
        *,
        step: int,
        auc: float,
        logloss: float,
        metrics: Mapping[str, float],
        score_diagnostics: Mapping[str, float | int],
        probe_metrics: Mapping[str, float],
        probe_score_diagnostics: Mapping[str, float | int],
    ) -> None:
        pass

    def should_collect_model_scalars(self, *, phase: str, step: int | None, trainer: Any) -> bool:
        return False

    def set_model_diagnostics_enabled(self, model: torch.nn.Module, enabled: bool) -> None:
        pass

    def consume_model_scalars(self, model: torch.nn.Module, *, phase: str) -> Mapping[str, float]:
        return {}

    def model_scalars(self, *, phase: str, step: int, scalars: Mapping[str, float]) -> None:
        pass

    def close(self) -> None:
        pass


class TensorBoardTrainReporter:
    def __init__(self, log_dir: Path) -> None:
        from torch.utils.tensorboard import SummaryWriter

        self.writer = SummaryWriter(log_dir)

    def train_step(self, *, step: int, loss: float, loss_components: Mapping[str, float], dense_lr: float) -> None:
        self.writer.add_scalar("Loss/train", float(loss), int(step))
        for name, value in loss_components.items():
            self.writer.add_scalar(f"Loss/train/{name}", float(value), int(step))
        self.writer.add_scalar("LR/dense", float(dense_lr), int(step))

    def validation_step(
        self,
        *,
        step: int,
        auc: float,
        logloss: float,
        metrics: Mapping[str, float],
        score_diagnostics: Mapping[str, float | int],
        probe_metrics: Mapping[str, float],
        probe_score_diagnostics: Mapping[str, float | int],
    ) -> None:
        del metrics
        self.writer.add_scalar("AUC/valid", float(auc), int(step))
        self.writer.add_scalar("LogLoss/valid", float(logloss), int(step))
        for metric_name, value in score_diagnostics.items():
            self.writer.add_scalar(f"score/{metric_name}", float(value), int(step))
        for metric_name, value in probe_metrics.items():
            self.writer.add_scalar(f"Probe/{metric_name}", float(value), int(step))
        for metric_name, value in probe_score_diagnostics.items():
            self.writer.add_scalar(f"Probe/score/{metric_name}", float(value), int(step))
        self.writer.flush()

    def should_collect_model_scalars(self, *, phase: str, step: int | None, trainer: Any) -> bool:
        del phase
        if step is None:
            return False
        interval = int(trainer.runtime_execution.progress_log_interval_steps)
        return (
            step == 1
            or (interval > 0 and step % interval == 0)
            or (trainer.eval_every_n_steps > 0 and step % trainer.eval_every_n_steps == 0)
            or (trainer.max_steps > 0 and step == trainer.max_steps)
        )

    def set_model_diagnostics_enabled(self, model: torch.nn.Module, enabled: bool) -> None:
        set_enabled = getattr(model, "set_training_diagnostics_enabled", None)
        if not callable(set_enabled):
            set_enabled = getattr(model, "set_tensorboard_diagnostics_enabled", None)
        if callable(set_enabled):
            set_enabled(enabled)

    def consume_model_scalars(self, model: torch.nn.Module, *, phase: str) -> Mapping[str, float]:
        consume_scalars = getattr(model, "consume_training_scalars", None)
        if not callable(consume_scalars):
            consume_scalars = getattr(model, "consume_tensorboard_scalars", None)
        if not callable(consume_scalars):
            return {}
        return consume_scalars(phase=phase)

    def model_scalars(self, *, phase: str, step: int, scalars: Mapping[str, float]) -> None:
        del phase
        for tag, value in scalars.items():
            self.writer.add_scalar(str(tag), float(value), int(step))

    def close(self) -> None:
        self.writer.close()


class BuildTrainReporterHook(Protocol):
    def __call__(self, context: PCVRTrainContext) -> TrainReporter:
        ...


def default_build_train_reporter(context: PCVRTrainContext) -> TrainReporter:
    return TensorBoardTrainReporter(context.tf_events_dir)


@dataclass(frozen=True, slots=True)
class PCVRTrainHooks:
    build_data: BuildTrainDataHook = default_build_train_data
    build_model: BuildTrainModelHook = default_build_train_model
    build_reporter: BuildTrainReporterHook = default_build_train_reporter
    build_trainer: BuildTrainTrainerHook = default_build_train_trainer
    run_training: RunTrainingHook = default_run_training
    build_summary: BuildTrainSummaryHook = default_build_train_summary


DEFAULT_PCVR_TRAIN_HOOKS = PCVRTrainHooks()


def build_pcvr_train_hooks(**overrides: Any) -> PCVRTrainHooks:
    return replace(DEFAULT_PCVR_TRAIN_HOOKS, **overrides)


__all__ = [
    "DEFAULT_PCVR_TRAIN_HOOKS",
    "BuildTrainDataHook",
    "BuildTrainModelHook",
    "BuildTrainReporterHook",
    "BuildTrainSummaryHook",
    "BuildTrainTrainerHook",
    "NoopTrainReporter",
    "PCVRTrainContext",
    "PCVRTrainDataBundle",
    "PCVRTrainHooks",
    "RunTrainingHook",
    "TensorBoardTrainReporter",
    "TrainReporter",
    "build_pcvr_train_hooks",
    "default_build_train_data",
    "default_build_train_model",
    "default_build_train_reporter",
    "default_build_train_summary",
    "default_build_train_trainer",
    "default_run_training",
]
