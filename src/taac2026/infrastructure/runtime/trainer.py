"""Shared PCVR pointwise trainer."""

from __future__ import annotations

import sys
import time
import math
from collections.abc import Mapping
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from taac2026.domain.metrics import binary_auc, binary_score_diagnostics
from taac2026.domain.config import (
    DENSE_LR_SCHEDULER_TYPE_CHOICES,
    PCVR_EARLY_STOPPING_METRIC_CHOICES,
    PCVR_VALIDATION_PROBE_MODE_CHOICES,
)
from taac2026.infrastructure.modeling.model_contract import batch_to_model_input
from taac2026.domain.runtime_config import DENSE_OPTIMIZER_TYPE_CHOICES, PCVRLossConfig, RuntimeExecutionConfig
from taac2026.infrastructure.logging import logger
from taac2026.infrastructure.modeling.tensors import sigmoid_probabilities_numpy
from taac2026.infrastructure.runtime.checkpoint_io import PCVRTrainerSupportMixin
from taac2026.infrastructure.runtime.execution import (
    EarlyStopping,
    compute_pcvr_loss,
    create_grad_scaler,
    maybe_compile_callable,
    maybe_prepare_internal_compile,
    runtime_autocast_context,
    runtime_execution_summary,
)
from taac2026.infrastructure.runtime.protocols import SparseParameterModel


class _NoopReporter:
    def train_step(self, **kwargs: Any) -> None:
        pass

    def validation_step(self, **kwargs: Any) -> None:
        pass

    def should_collect_model_scalars(self, **kwargs: Any) -> bool:
        return False

    def set_model_diagnostics_enabled(self, model: nn.Module, enabled: bool) -> None:
        pass

    def consume_model_scalars(self, model: nn.Module, *, phase: str) -> dict[str, float]:
        return {}

    def model_scalars(self, **kwargs: Any) -> None:
        pass


class _ScalarWriterReporter:
    def __init__(self, writer: Any) -> None:
        self.writer = writer

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
        flush = getattr(self.writer, "flush", None)
        if callable(flush):
            flush()

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

    def set_model_diagnostics_enabled(self, model: nn.Module, enabled: bool) -> None:
        set_enabled = getattr(model, "set_training_diagnostics_enabled", None)
        if not callable(set_enabled):
            set_enabled = getattr(model, "set_tensorboard_diagnostics_enabled", None)
        if callable(set_enabled):
            set_enabled(enabled)

    def consume_model_scalars(self, model: nn.Module, *, phase: str) -> Mapping[str, float]:
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


def _use_interactive_progress() -> bool:
    isatty = getattr(sys.stderr, "isatty", None)
    return bool(isatty and isatty())


def _should_log_progress(current_batch: int, total_batches: int, interval: int) -> bool:
    return current_batch == 1 or current_batch == total_batches or current_batch % interval == 0


def _format_duration(seconds: float) -> str:
    return str(timedelta(seconds=max(0, round(seconds))))


class PCVRPointwiseTrainer(PCVRTrainerSupportMixin):
    """PCVR trainer for binary pointwise classification with AUC monitoring."""

    def __init__(
        self,
        model: nn.Module,
        model_input_type: Any,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        lr: float,
        max_steps: int,
        device: str,
        save_dir: str | Path,
        early_stopping: EarlyStopping,
        dense_optimizer_type: str = "adamw",
        scheduler_type: str = "none",
        warmup_steps: int = 0,
        min_lr_ratio: float = 0.0,
        loss_terms: Any | None = None,
        sparse_lr: float = 0.05,
        sparse_weight_decay: float = 0.0,
        reinit_sparse_every_n_steps: int = 0,
        reinit_cardinality_threshold: int = 0,
        ckpt_params: dict[str, Any] | None = None,
        writer: Any | None = None,
        reporter: Any | None = None,
        schema_path: str | Path | None = None,
        eval_every_n_steps: int = 5_000,
        validation_probe_mode: str = "none",
        early_stopping_metric: str = "auc",
        train_config: dict[str, Any] | None = None,
        runtime_execution: RuntimeExecutionConfig | None = None,
    ) -> None:
        self.model = model
        self.model_input_type = model_input_type
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.writer = writer
        self.reporter = reporter or (_ScalarWriterReporter(writer) if writer is not None else _NoopReporter())
        self.schema_path = Path(schema_path).expanduser().resolve() if schema_path else None
        self.dense_optimizer_type = str(dense_optimizer_type).strip().lower()
        if self.dense_optimizer_type not in DENSE_OPTIMIZER_TYPE_CHOICES:
            raise ValueError(f"unsupported dense optimizer type: {dense_optimizer_type}")
        self.scheduler_type = str(scheduler_type).strip().lower()
        if self.scheduler_type not in DENSE_LR_SCHEDULER_TYPE_CHOICES:
            raise ValueError(f"unsupported scheduler type: {scheduler_type}")
        self.warmup_steps = int(warmup_steps)
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative")
        self.min_lr_ratio = float(min_lr_ratio)
        if not 0.0 <= self.min_lr_ratio <= 1.0:
            raise ValueError("min_lr_ratio must be between 0.0 and 1.0")
        self.base_dense_lr = float(lr)
        self.current_dense_lr = self.base_dense_lr
        self.optim_step = 0
        self.dense_params: list[nn.Parameter] = []

        self.sparse_optimizer: torch.optim.Optimizer | None
        if isinstance(model, SparseParameterModel):
            sparse_params = model.get_sparse_params()
            dense_params = model.get_dense_params()
            if not sparse_params:
                logger.info(
                    "Model exposes get_sparse_params but has no embedding parameters; using {} for all params",
                    self._dense_optimizer_display_name(),
                )
                self.sparse_optimizer = None
                self.dense_params = list(model.parameters())
                self.dense_optimizer = self._build_dense_optimizer(self.dense_params, lr)
            else:
                self.dense_params = list(dense_params)
                sparse_param_count = sum(parameter.numel() for parameter in sparse_params)
                dense_param_count = sum(parameter.numel() for parameter in dense_params)
                logger.info(
                    "Sparse params: {} tensors, {} parameters (Adagrad lr={})",
                    len(sparse_params),
                    f"{sparse_param_count:,}",
                    sparse_lr,
                )
                logger.info(
                    "Dense params: {} tensors, {} parameters ({} lr={})",
                    len(dense_params),
                    f"{dense_param_count:,}",
                    self._dense_optimizer_display_name(),
                    lr,
                )
                self.sparse_optimizer = torch.optim.Adagrad(
                    sparse_params, lr=sparse_lr, weight_decay=sparse_weight_decay
                )
                self.dense_optimizer: torch.optim.Optimizer = self._build_dense_optimizer(self.dense_params, lr)
        else:
            self.sparse_optimizer = None
            self.dense_params = list(model.parameters())
            self.dense_optimizer = self._build_dense_optimizer(self.dense_params, lr)

        self.max_steps = int(max_steps)
        self.device = device
        self.save_dir = Path(save_dir).expanduser().resolve()
        self.early_stopping = early_stopping
        self.runtime_execution = runtime_execution or RuntimeExecutionConfig()
        uses_internal_compile = maybe_prepare_internal_compile(
            self.model,
            enabled=self.runtime_execution.compile,
            label="PCVR training model",
        )
        self.forward_model = maybe_compile_callable(
            self.model,
            enabled=self.runtime_execution.compile and not uses_internal_compile,
            label="PCVR training forward",
        )
        self.predict_fn = maybe_compile_callable(
            self.model.predict,
            enabled=self.runtime_execution.compile and not uses_internal_compile,
            label="PCVR trainer predict",
        )
        self.grad_scaler = create_grad_scaler(self.runtime_execution, self.device)
        self.loss_config = PCVRLossConfig.from_value(loss_terms)
        self.last_train_loss_components: dict[str, float] = {}
        self.reinit_sparse_every_n_steps = int(reinit_sparse_every_n_steps)
        self.reinit_cardinality_threshold = reinit_cardinality_threshold
        self.sparse_lr = sparse_lr
        self.sparse_weight_decay = sparse_weight_decay
        self.ckpt_params = ckpt_params or {}
        self.eval_every_n_steps = int(eval_every_n_steps)
        if self.eval_every_n_steps < 0:
            raise ValueError("eval_every_n_steps must be non-negative")
        self.validation_probe_mode = str(validation_probe_mode).strip().lower()
        if self.validation_probe_mode not in PCVR_VALIDATION_PROBE_MODE_CHOICES:
            raise ValueError(f"unsupported validation probe mode: {validation_probe_mode}")
        self.early_stopping_metric = str(early_stopping_metric).strip().lower()
        if self.early_stopping_metric not in PCVR_EARLY_STOPPING_METRIC_CHOICES:
            raise ValueError(f"unsupported early stopping metric: {early_stopping_metric}")
        if self.validation_probe_mode == "none" and self.early_stopping_metric.startswith("probe_"):
            raise ValueError("probe early stopping metrics require validation_probe_mode != 'none'")
        self.train_config = train_config
        self.last_eval_diagnostics: dict[str, float | int] = {}
        self.last_eval_probe_diagnostics: dict[str, float | int] = {}
        self.last_eval_probe_metrics: dict[str, float] = {}
        self.last_eval_metrics: dict[str, float] = {}
        self.last_train_model_scalars: dict[str, float] = {}
        self.last_eval_model_scalars: dict[str, float] = {}

        logger.info(
            "PCVRPointwiseTrainer loss_terms={}, "
            "dense_optimizer_type={}, scheduler_type={}, warmup_steps={}, min_lr_ratio={}, "
            "max_steps={}, reinit_sparse_every_n_steps={}, "
            "validation_probe_mode={}, early_stopping_metric={}",
            self.loss_config.summary(),
            self.dense_optimizer_type,
            self.scheduler_type,
            self.warmup_steps,
            self.min_lr_ratio,
            self.max_steps,
            self.reinit_sparse_every_n_steps,
            self.validation_probe_mode,
            self.early_stopping_metric,
        )
        logger.info("PCVRPointwiseTrainer runtime: {}", runtime_execution_summary(self.runtime_execution, self.device))

    def _log_loop_progress(
        self,
        phase: str,
        current_batch: int,
        total_batches: int,
        *,
        loop_started_at: float | None = None,
        loss: float | None = None,
    ) -> None:
        message = f"{phase} progress {current_batch}/{total_batches} ({current_batch / total_batches:.1%})"
        if loop_started_at is not None and current_batch > 0:
            elapsed_seconds = max(0.0, time.monotonic() - loop_started_at)
            eta_seconds = elapsed_seconds * max(0, total_batches - current_batch) / current_batch
            message = f"{message} | eta={_format_duration(eta_seconds)}"
        if loss is not None:
            message = f"{message} | loss={loss:.4f}"
        logger.info(message)

    def train(self) -> None:
        logger.info("Start Training (PCVR pointwise)")
        self.model.train()

        total_step = 0
        total_train_steps = self.max_steps if self.max_steps > 0 else self._logical_train_sweep_steps()
        use_tqdm = _use_interactive_progress()
        log_interval = self.runtime_execution.progress_log_interval_steps
        loop_started_at = time.monotonic()
        eval_interval = self.eval_every_n_steps if self.eval_every_n_steps > 0 else self._logical_train_sweep_steps()
        train_iter = self._infinite_train_batches()
        train_pbar = tqdm(total=total_train_steps, dynamic_ncols=True) if use_tqdm else None
        window_loss_sum = 0.0
        window_loss_steps = 0
        evaluated_on_last_step = False

        while total_step < total_train_steps:
            batch = next(train_iter)
            loss = self._train_step(batch)
            total_step += 1
            window_loss_sum += loss
            window_loss_steps += 1

            self.reporter.train_step(
                step=total_step,
                loss=loss,
                loss_components=self.last_train_loss_components,
                dense_lr=self.current_dense_lr,
            )
            self._write_model_training_scalars("train", self.last_train_model_scalars, total_step)

            if train_pbar is not None:
                train_pbar.update(1)
                train_pbar.set_postfix({"loss": f"{loss:.4f}"})
            elif _should_log_progress(total_step, total_train_steps, log_interval):
                self._log_loop_progress(
                    "Train",
                    total_step,
                    total_train_steps,
                    loop_started_at=loop_started_at,
                    loss=loss,
                )

            if self.reinit_sparse_every_n_steps > 0 and total_step % self.reinit_sparse_every_n_steps == 0:
                self._rebuild_sparse_optimizer(total_step)

            if total_step % eval_interval != 0 and total_step != total_train_steps:
                continue

            logger.info("Train step {}, Average Loss: {}", total_step, window_loss_sum / max(1, window_loss_steps))
            window_loss_sum = 0.0
            window_loss_steps = 0

            logger.info("Evaluating at step {}", total_step)
            val_auc, val_logloss = self.evaluate(step=total_step)
            self.model.train()
            torch.cuda.empty_cache()

            logger.info("Step {} Validation | AUC: {}, LogLoss: {}", total_step, val_auc, val_logloss)

            self._report_validation(total_step, val_auc, val_logloss)

            self._handle_validation_result(total_step, val_auc, val_logloss)
            evaluated_on_last_step = total_step == total_train_steps

            if self.early_stopping.early_stop:
                logger.info("Early stopping at step {}", total_step)
                if train_pbar is not None:
                    train_pbar.close()
                return

        if train_pbar is not None:
            train_pbar.close()

        if not evaluated_on_last_step:
            logger.info("Evaluating at step {}", total_step)
            val_auc, val_logloss = self.evaluate(step=total_step)
            self.model.train()
            torch.cuda.empty_cache()
            logger.info("Step {} Validation | AUC: {}, LogLoss: {}", total_step, val_auc, val_logloss)
            self._report_validation(total_step, val_auc, val_logloss)
            self._handle_validation_result(total_step, val_auc, val_logloss)

    def _make_model_input(self, device_batch: dict[str, Any]) -> Any:
        return batch_to_model_input(device_batch, self.model_input_type, torch.device(self.device))

    def _train_step(self, batch: dict[str, Any]) -> float:
        device_batch = self._batch_to_device(batch)
        label = device_batch["label"].float()
        self._set_dense_learning_rate(self.optim_step + 1)

        self.dense_optimizer.zero_grad()
        if self.sparse_optimizer is not None:
            self.sparse_optimizer.zero_grad()

        collect_model_scalars = self._should_collect_train_model_scalars(self.optim_step + 1)
        self._set_model_training_diagnostics_enabled(collect_model_scalars)
        model_input = self._make_model_input(device_batch)
        with runtime_autocast_context(self.runtime_execution, self.device):
            logits = self.forward_model(model_input).squeeze(-1)
            loss, loss_components = compute_pcvr_loss(logits, label, self.loss_config, model=self.model)

        # Detect NaN/inf in logits and loss before backward to prevent silent
        # model corruption, especially under AMP where reduced precision can
        # produce NaN that GradScaler (float16) skips silently or bfloat16
        # propagates directly.
        if not torch.isfinite(loss).all():
            n_bad_logits = int((~torch.isfinite(logits)).sum())
            loss_value = loss.detach().float().item()
            logger.warning(
                "Train step skipped: non-finite loss={:.6f}, non-finite logits={}/{}. "
                "Skipping backward and optimizer step to avoid parameter corruption.",
                loss_value,
                n_bad_logits,
                logits.numel(),
            )
            self.last_train_loss_components = {
                name: float("nan") for name in loss_components
            }
            self.last_train_model_scalars = {}
            self._set_model_training_diagnostics_enabled(False)
            # Do NOT increment optim_step; this step produced no valid gradient.
            return float("nan")

        self.last_train_loss_components = {name: float(value.detach().float().cpu()) for name, value in loss_components.items()}

        if collect_model_scalars:
            self.last_train_model_scalars = self._consume_model_training_scalars("train")
        else:
            self.last_train_model_scalars = {}
        self._set_model_training_diagnostics_enabled(False)

        if self.grad_scaler is not None:
            self.grad_scaler.scale(loss).backward()
            self.grad_scaler.unscale_(self.dense_optimizer)
            if self.sparse_optimizer is not None:
                self.grad_scaler.unscale_(self.sparse_optimizer)

            # After unscale, check whether any gradient is inf/nan.  GradScaler
            # will skip the optimizer step internally when this happens, but we
            # log it so the event is visible instead of silent.
            scale_before_step = self.grad_scaler.get_scale()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0, foreach=False)
            self._orthogonalize_dense_gradients()

            self.grad_scaler.step(self.dense_optimizer)
            if self.sparse_optimizer is not None:
                self.grad_scaler.step(self.sparse_optimizer)
            self.grad_scaler.update()

            scale_after_update = self.grad_scaler.get_scale()
            if scale_after_update < scale_before_step:
                logger.warning(
                    "Train step: GradScaler reduced scale {:.1e} -> {:.1e}, "
                    "indicating inf/nan gradients were found and optimizer step was skipped.",
                    scale_before_step,
                    scale_after_update,
                )
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0, foreach=False)
            self._orthogonalize_dense_gradients()

            self.dense_optimizer.step()
            if self.sparse_optimizer is not None:
                self.sparse_optimizer.step()

        self.optim_step += 1

        return loss.item()

    def evaluate(self, step: int | None = None) -> tuple[float, float]:
        logger.info("Start Evaluation (PCVR pointwise) - validation")
        self.model.eval()

        total_valid_batches = len(self.valid_loader)
        use_tqdm = _use_interactive_progress()
        log_interval = self.runtime_execution.progress_log_interval_steps
        loop_started_at = time.monotonic()
        valid_iter = enumerate(self.valid_loader)
        pbar = (
            tqdm(valid_iter, total=total_valid_batches, dynamic_ncols=True)
            if use_tqdm
            else valid_iter
        )
        all_logits_list = []
        all_probe_logits_list = [] if self.validation_probe_mode != "none" else None
        all_labels_list = []
        model_scalar_sums: dict[str, float] = {}
        model_scalar_counts: dict[str, int] = {}
        collect_model_scalars = self.reporter.should_collect_model_scalars(phase="valid", step=step, trainer=self)

        with torch.no_grad():
            for step_index, batch in pbar:
                self._set_model_training_diagnostics_enabled(collect_model_scalars)
                logits, labels = self._evaluate_step(batch)
                if collect_model_scalars:
                    self._accumulate_model_training_scalars("valid", model_scalar_sums, model_scalar_counts)
                self._set_model_training_diagnostics_enabled(False)
                all_logits_list.append(logits.detach())
                all_labels_list.append(labels.detach())
                if all_probe_logits_list is not None:
                    probe_logits, _probe_labels = self._evaluate_step(batch, probe_mode=self.validation_probe_mode)
                    all_probe_logits_list.append(probe_logits.detach())

                current_batch = step_index + 1
                if not use_tqdm and _should_log_progress(current_batch, total_valid_batches, log_interval):
                    self._log_loop_progress(
                        "Validation",
                        current_batch,
                        total_valid_batches,
                        loop_started_at=loop_started_at,
                    )

        if use_tqdm:
            pbar.close()

        all_logits = torch.cat(all_logits_list, dim=0).float()
        all_labels = torch.cat(all_labels_list, dim=0).long()
        auc, logloss, diagnostics = self._compute_validation_metrics(all_logits, all_labels, label="Evaluate")

        self.last_eval_diagnostics = diagnostics
        self.last_eval_metrics = {"auc": auc, "logloss": logloss}
        self.last_eval_probe_diagnostics = {}
        self.last_eval_probe_metrics = {}
        self.last_eval_model_scalars = {
            tag: model_scalar_sums[tag] / model_scalar_counts[tag]
            for tag in model_scalar_sums
            if model_scalar_counts[tag] > 0
        }
        logger.info(
            "Validation score diagnostics | pos={} neg={} pos_mean={:.6f} neg_mean={:.6f} margin={:.6f} score_std={:.6f}",
            self.last_eval_diagnostics["positive_count"],
            self.last_eval_diagnostics["negative_count"],
            self.last_eval_diagnostics["positive_score_mean"],
            self.last_eval_diagnostics["negative_score_mean"],
            self.last_eval_diagnostics["score_margin_mean"],
            self.last_eval_diagnostics["score_std"],
        )

        if all_probe_logits_list is not None:
            all_probe_logits = torch.cat(all_probe_logits_list, dim=0).float()
            probe_auc, probe_logloss, probe_diagnostics = self._compute_validation_metrics(
                all_probe_logits,
                all_labels,
                label="Evaluate probe",
            )
            auc_retention = self._auc_retention(auc, probe_auc)
            self.last_eval_probe_diagnostics = probe_diagnostics
            self.last_eval_probe_metrics = {
                "auc": probe_auc,
                "logloss": probe_logloss,
                "auc_retention": auc_retention,
            }
            self.last_eval_metrics.update(
                {
                    "probe_auc": probe_auc,
                    "probe_logloss": probe_logloss,
                    "probe_auc_retention": auc_retention,
                }
            )
            logger.info(
                "Validation probe ({}) | AUC: {}, LogLoss: {}, retention: {}",
                self.validation_probe_mode,
                probe_auc,
                probe_logloss,
                auc_retention,
            )

        return auc, logloss

    def _report_validation(self, total_step: int, val_auc: float, val_logloss: float) -> None:
        self.reporter.validation_step(
            step=total_step,
            auc=val_auc,
            logloss=val_logloss,
            metrics=self.last_eval_metrics,
            score_diagnostics=self.last_eval_diagnostics,
            probe_metrics=self.last_eval_probe_metrics,
            probe_score_diagnostics=self.last_eval_probe_diagnostics,
        )
        self._write_model_training_scalars("valid", self.last_eval_model_scalars, total_step)

    def _set_model_training_diagnostics_enabled(self, enabled: bool) -> None:
        self.reporter.set_model_diagnostics_enabled(self.model, enabled)

    def _should_collect_train_model_scalars(self, step: int) -> bool:
        return self.reporter.should_collect_model_scalars(phase="train", step=step, trainer=self)

    def _consume_model_training_scalars(self, phase: str) -> dict[str, float]:
        scalars = self.reporter.consume_model_scalars(self.model, phase=phase)
        if not isinstance(scalars, Mapping):
            return {}
        cleaned: dict[str, float] = {}
        for tag, value in scalars.items():
            try:
                numeric_value = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(numeric_value):
                cleaned[str(tag)] = numeric_value
        return cleaned

    def _accumulate_model_training_scalars(
        self,
        phase: str,
        scalar_sums: dict[str, float],
        scalar_counts: dict[str, int],
    ) -> None:
        for tag, value in self._consume_model_training_scalars(phase).items():
            scalar_sums[tag] = scalar_sums.get(tag, 0.0) + value
            scalar_counts[tag] = scalar_counts.get(tag, 0) + 1

    def _write_model_training_scalars(self, phase: str, scalars: dict[str, float], total_step: int) -> None:
        self.reporter.model_scalars(phase=phase, step=total_step, scalars=scalars)

    def _compute_validation_metrics(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        *,
        label: str,
    ) -> tuple[float, float, dict[str, float | int]]:
        valid_logit_mask = ~torch.isnan(logits)
        valid_logits = logits[valid_logit_mask]
        valid_labels = labels[valid_logit_mask]
        if len(valid_logits) > 0:
            logloss = F.binary_cross_entropy_with_logits(valid_logits, valid_labels.float()).item()
        else:
            logloss = float("inf")

        probabilities = sigmoid_probabilities_numpy(logits)
        labels_np = labels.detach().cpu().numpy()
        nan_mask = np.isnan(probabilities)
        if nan_mask.any():
            n_nan = int(nan_mask.sum())
            logger.warning("[{}] {}/{} predictions are NaN, filtering them out", label, n_nan, len(probabilities))
            valid_mask = ~nan_mask
            probabilities = probabilities[valid_mask]
            labels_np = labels_np[valid_mask]

        auc = binary_auc(labels_np, probabilities)
        diagnostics = binary_score_diagnostics(labels_np, probabilities)
        return auc, logloss, diagnostics

    def _auc_retention(self, full_auc: float, probe_auc: float) -> float:
        full_uplift = float(full_auc) - 0.5
        if full_uplift <= 1.0e-8:
            return 0.0
        return float((float(probe_auc) - 0.5) / full_uplift)

    def _apply_validation_probe(self, device_batch: dict[str, Any], probe_mode: str) -> dict[str, Any]:
        if probe_mode == "none":
            return device_batch
        probed_batch = dict(device_batch)
        for feature_key in ("user_int_feats", "item_int_feats"):
            value = probed_batch.get(feature_key)
            if isinstance(value, torch.Tensor):
                probed_batch[feature_key] = torch.zeros_like(value)
        if probe_mode == "drop_all_sparse":
            for domain in probed_batch.get("_seq_domains", ()):
                value = probed_batch.get(domain)
                if isinstance(value, torch.Tensor):
                    probed_batch[domain] = torch.zeros_like(value)
        return probed_batch

    def validation_metric_score(self, metric_name: str, val_auc: float, val_logloss: float) -> float:
        if metric_name == "auc":
            return float(val_auc)
        if metric_name == "logloss":
            return -float(val_logloss)
        if metric_name == "probe_auc":
            return float(self.last_eval_probe_metrics["auc"])
        if metric_name == "probe_logloss":
            return -float(self.last_eval_probe_metrics["logloss"])
        if metric_name == "probe_auc_retention":
            return float(self.last_eval_probe_metrics["auc_retention"])
        raise ValueError(f"unsupported validation metric: {metric_name}")

    def validation_early_stopping_score(self, val_auc: float, val_logloss: float) -> float:
        return self.validation_metric_score(self.early_stopping_metric, val_auc, val_logloss)

    def _evaluate_step(self, batch: dict[str, Any], probe_mode: str = "none") -> tuple[torch.Tensor, torch.Tensor]:
        device_batch = self._batch_to_device(batch)
        device_batch = self._apply_validation_probe(device_batch, probe_mode)
        label = device_batch["label"]

        model_input = self._make_model_input(device_batch)
        with runtime_autocast_context(self.runtime_execution, self.device):
            logits, _embeddings = self.predict_fn(model_input)
        logits = logits.squeeze(-1)

        return logits, label
