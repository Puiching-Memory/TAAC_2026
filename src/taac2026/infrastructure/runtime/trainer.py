"""Shared PCVR pointwise trainer."""

from __future__ import annotations

import sys
import time
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from taac2026.domain.metrics import binary_score_diagnostics
from taac2026.domain.config import DENSE_LR_SCHEDULER_TYPE_CHOICES
from taac2026.domain.model_contract import batch_to_model_input
from taac2026.infrastructure.logging import logger
from taac2026.infrastructure.modeling.tensors import sigmoid_probabilities_numpy
from taac2026.infrastructure.runtime.checkpoint_io import PCVRTrainerSupportMixin
from taac2026.infrastructure.runtime.execution import (
    BinaryClassificationLossConfig,
    DENSE_OPTIMIZER_TYPE_CHOICES,
    EarlyStopping,
    RuntimeExecutionConfig,
    compute_binary_classification_loss,
    create_grad_scaler,
    maybe_compile_callable,
)


def _use_interactive_progress() -> bool:
    isatty = getattr(sys.stderr, "isatty", None)
    return bool(isatty and isatty())


def _progress_log_interval(total_batches: int) -> int:
    if total_batches <= 0:
        return 1
    return max(1, total_batches // 20)


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
        loss_type: str = "bce",
        focal_alpha: float = 0.1,
        focal_gamma: float = 2.0,
        pairwise_auc_weight: float = 0.0,
        pairwise_auc_temperature: float = 1.0,
        sparse_lr: float = 0.05,
        sparse_weight_decay: float = 0.0,
        reinit_sparse_every_n_steps: int = 0,
        reinit_cardinality_threshold: int = 0,
        ckpt_params: dict[str, Any] | None = None,
        writer: Any | None = None,
        schema_path: str | Path | None = None,
        eval_every_n_steps: int = 0,
        train_config: dict[str, Any] | None = None,
        runtime_execution: RuntimeExecutionConfig | None = None,
    ) -> None:
        self.model = model
        self.model_input_type = model_input_type
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.writer = writer
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
        if hasattr(model, "get_sparse_params"):
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
        self.forward_model = maybe_compile_callable(
            self.model,
            enabled=self.runtime_execution.compile,
            label="PCVR training forward",
        )
        self.predict_fn = maybe_compile_callable(
            self.model.predict,
            enabled=self.runtime_execution.compile,
            label="PCVR trainer predict",
        )
        self.grad_scaler = create_grad_scaler(self.runtime_execution, self.device)
        self.loss_config = BinaryClassificationLossConfig(
            loss_type=loss_type,
            focal_alpha=focal_alpha,
            focal_gamma=focal_gamma,
            pairwise_auc_weight=pairwise_auc_weight,
            pairwise_auc_temperature=pairwise_auc_temperature,
        )
        self.loss_type = self.loss_config.loss_type
        self.focal_alpha = self.loss_config.focal_alpha
        self.focal_gamma = self.loss_config.focal_gamma
        self.reinit_sparse_every_n_steps = int(reinit_sparse_every_n_steps)
        self.reinit_cardinality_threshold = reinit_cardinality_threshold
        self.sparse_lr = sparse_lr
        self.sparse_weight_decay = sparse_weight_decay
        self.ckpt_params = ckpt_params or {}
        self.eval_every_n_steps = eval_every_n_steps
        self.train_config = train_config
        self.last_eval_diagnostics: dict[str, float | int] = {}

        logger.info(
            "PCVRPointwiseTrainer loss_type={}, focal_alpha={}, focal_gamma={}, "
            "pairwise_auc_weight={}, pairwise_auc_temperature={}, "
            "dense_optimizer_type={}, scheduler_type={}, warmup_steps={}, min_lr_ratio={}, "
            "max_steps={}, reinit_sparse_every_n_steps={}",
            self.loss_type,
            self.focal_alpha,
            self.focal_gamma,
            self.loss_config.pairwise_auc_weight,
            self.loss_config.pairwise_auc_temperature,
            self.dense_optimizer_type,
            self.scheduler_type,
            self.warmup_steps,
            self.min_lr_ratio,
            self.max_steps,
            self.reinit_sparse_every_n_steps,
        )
        logger.info("PCVRPointwiseTrainer runtime: {}", self.runtime_execution.summary(self.device))

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
        log_interval = _progress_log_interval(total_train_steps)
        loop_started_at = time.monotonic()
        eval_interval = self.eval_every_n_steps if self.eval_every_n_steps > 0 else self._logical_train_sweep_steps()
        if self.early_stopping.patience_unit == "steps":
            self.early_stopping.configure_step_scale(eval_interval)
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

            if self.writer:
                self.writer.add_scalar("Loss/train", loss, total_step)
                self.writer.add_scalar("LR/dense", self.current_dense_lr, total_step)

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

            if self.writer:
                self.writer.add_scalar("AUC/valid", val_auc, total_step)
                self.writer.add_scalar("LogLoss/valid", val_logloss, total_step)
                self._write_eval_diagnostics(total_step)

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
            if self.writer:
                self.writer.add_scalar("AUC/valid", val_auc, total_step)
                self.writer.add_scalar("LogLoss/valid", val_logloss, total_step)
                self._write_eval_diagnostics(total_step)
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

        model_input = self._make_model_input(device_batch)
        with self.runtime_execution.autocast_context(self.device):
            logits = self.forward_model(model_input).squeeze(-1)
            loss = compute_binary_classification_loss(logits, label, self.loss_config)

        if self.grad_scaler is not None:
            self.grad_scaler.scale(loss).backward()
            self.grad_scaler.unscale_(self.dense_optimizer)
            if self.sparse_optimizer is not None:
                self.grad_scaler.unscale_(self.sparse_optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0, foreach=False)
            self._orthogonalize_dense_gradients()

            self.grad_scaler.step(self.dense_optimizer)
            if self.sparse_optimizer is not None:
                self.grad_scaler.step(self.sparse_optimizer)
            self.grad_scaler.update()
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
        log_interval = _progress_log_interval(total_valid_batches)
        loop_started_at = time.monotonic()
        valid_iter = enumerate(self.valid_loader)
        pbar = (
            tqdm(valid_iter, total=total_valid_batches, dynamic_ncols=True)
            if use_tqdm
            else valid_iter
        )
        all_logits_list = []
        all_labels_list = []

        with torch.no_grad():
            for step_index, batch in pbar:
                logits, labels = self._evaluate_step(batch)
                all_logits_list.append(logits.detach().cpu())
                all_labels_list.append(labels.detach().cpu())

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

        probabilities = sigmoid_probabilities_numpy(all_logits)
        labels_np = all_labels.numpy()
        nan_mask = np.isnan(probabilities)
        if nan_mask.any():
            n_nan = int(nan_mask.sum())
            logger.warning("[Evaluate] {}/{} predictions are NaN, filtering them out", n_nan, len(probabilities))
            valid_mask = ~nan_mask
            probabilities = probabilities[valid_mask]
            labels_np = labels_np[valid_mask]

        if len(probabilities) == 0 or len(np.unique(labels_np)) < 2:
            auc = 0.0
        else:
            auc = float(roc_auc_score(labels_np, probabilities))

        valid_logits = all_logits[~torch.isnan(all_logits)]
        valid_labels = all_labels[~torch.isnan(all_logits)]
        if len(valid_logits) > 0:
            logloss = F.binary_cross_entropy_with_logits(valid_logits, valid_labels.float()).item()
        else:
            logloss = float("inf")

        self.last_eval_diagnostics = binary_score_diagnostics(labels_np, probabilities)
        logger.info(
            "Validation score diagnostics | pos={} neg={} pos_mean={:.6f} neg_mean={:.6f} margin={:.6f} score_std={:.6f}",
            self.last_eval_diagnostics["positive_count"],
            self.last_eval_diagnostics["negative_count"],
            self.last_eval_diagnostics["positive_score_mean"],
            self.last_eval_diagnostics["negative_score_mean"],
            self.last_eval_diagnostics["score_margin_mean"],
            self.last_eval_diagnostics["score_std"],
        )

        return auc, logloss

    def _evaluate_step(self, batch: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        device_batch = self._batch_to_device(batch)
        label = device_batch["label"]

        model_input = self._make_model_input(device_batch)
        with self.runtime_execution.autocast_context(self.device):
            logits, _embeddings = self.predict_fn(model_input)
        logits = logits.squeeze(-1)

        return logits, label