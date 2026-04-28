"""Shared PCVR pointwise trainer."""

from __future__ import annotations

import logging
import shutil
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
from taac2026.infrastructure.checkpoints import build_checkpoint_dir_name, write_checkpoint_sidecars
from taac2026.infrastructure.pcvr.protocol import batch_to_model_input
from taac2026.infrastructure.pcvr.tensors import sigmoid_probabilities_numpy
from taac2026.infrastructure.training.runtime import (
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


class PCVRPointwiseTrainer:
    """PCVR trainer for binary pointwise classification with AUC monitoring."""

    def __init__(
        self,
        model: nn.Module,
        model_input_type: Any,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        lr: float,
        num_epochs: int,
        device: str,
        save_dir: str | Path,
        early_stopping: EarlyStopping,
        dense_optimizer_type: str = "adamw",
        loss_type: str = "bce",
        focal_alpha: float = 0.1,
        focal_gamma: float = 2.0,
        pairwise_auc_weight: float = 0.0,
        pairwise_auc_temperature: float = 1.0,
        sparse_lr: float = 0.05,
        sparse_weight_decay: float = 0.0,
        reinit_sparse_after_epoch: int = 1,
        reinit_cardinality_threshold: int = 0,
        ckpt_params: dict[str, Any] | None = None,
        writer: Any | None = None,
        schema_path: str | Path | None = None,
        ns_groups_path: str | Path | None = None,
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
        self.ns_groups_path = Path(ns_groups_path).expanduser().resolve() if ns_groups_path else None
        self.dense_optimizer_type = str(dense_optimizer_type).strip().lower()
        if self.dense_optimizer_type not in DENSE_OPTIMIZER_TYPE_CHOICES:
            raise ValueError(f"unsupported dense optimizer type: {dense_optimizer_type}")
        self.dense_params: list[nn.Parameter] = []

        self.sparse_optimizer: torch.optim.Optimizer | None
        if hasattr(model, "get_sparse_params"):
            sparse_params = model.get_sparse_params()
            dense_params = model.get_dense_params()
            if not sparse_params:
                logging.info("Model exposes get_sparse_params but has no embedding parameters; using AdamW for all params")
                self.sparse_optimizer = None
                self.dense_params = list(model.parameters())
                self.dense_optimizer = self._build_dense_optimizer(self.dense_params, lr)
            else:
                self.dense_params = list(dense_params)
                sparse_param_count = sum(parameter.numel() for parameter in sparse_params)
                dense_param_count = sum(parameter.numel() for parameter in dense_params)
                logging.info(
                    "Sparse params: %s tensors, %s parameters (Adagrad lr=%s)",
                    len(sparse_params),
                    f"{sparse_param_count:,}",
                    sparse_lr,
                )
                logging.info(
                    "Dense params: %s tensors, %s parameters (AdamW lr=%s)",
                    len(dense_params),
                    f"{dense_param_count:,}",
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

        self.num_epochs = num_epochs
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
        self.reinit_sparse_after_epoch = reinit_sparse_after_epoch
        self.reinit_cardinality_threshold = reinit_cardinality_threshold
        self.sparse_lr = sparse_lr
        self.sparse_weight_decay = sparse_weight_decay
        self.ckpt_params = ckpt_params or {}
        self.eval_every_n_steps = eval_every_n_steps
        self.train_config = train_config
        self.last_eval_diagnostics: dict[str, float | int] = {}

        logging.info(
            "PCVRPointwiseTrainer loss_type=%s, focal_alpha=%s, focal_gamma=%s, "
            "pairwise_auc_weight=%s, pairwise_auc_temperature=%s, "
            "dense_optimizer_type=%s, reinit_sparse_after_epoch=%s",
            self.loss_type,
            self.focal_alpha,
            self.focal_gamma,
            self.loss_config.pairwise_auc_weight,
            self.loss_config.pairwise_auc_temperature,
            self.dense_optimizer_type,
            reinit_sparse_after_epoch,
        )
        logging.info("PCVRPointwiseTrainer runtime: %s", self.runtime_execution.summary(self.device))

    def _build_dense_optimizer(self, parameters: list[nn.Parameter], lr: float) -> torch.optim.Optimizer:
        return torch.optim.AdamW(parameters, lr=lr, betas=(0.9, 0.98))

    def _orthogonalize_dense_gradients(self) -> None:
        if self.dense_optimizer_type != "orthogonal_adamw":
            return
        for parameter in self.dense_params:
            gradient = parameter.grad
            if gradient is None or gradient.ndim < 2:
                continue
            with torch.no_grad():
                original_dtype = gradient.dtype
                original_norm = gradient.norm().clamp_min(1e-12)
                matrix = gradient.float().reshape(gradient.shape[0], -1)
                transposed = matrix.shape[0] > matrix.shape[1]
                if transposed:
                    matrix = matrix.t()
                matrix = matrix / matrix.norm().clamp_min(1e-12)
                for _ in range(2):
                    matrix = 1.5 * matrix - 0.5 * matrix @ (matrix.t() @ matrix)
                if transposed:
                    matrix = matrix.t()
                gradient.copy_((matrix.reshape_as(gradient) * original_norm).to(original_dtype))

    def _build_step_dir_name(self, global_step: int, is_best: bool = False) -> str:
        return build_checkpoint_dir_name(global_step, self.ckpt_params, is_best=is_best)

    def _write_sidecar_files(self, checkpoint_dir: Path) -> None:
        write_checkpoint_sidecars(
            checkpoint_dir,
            schema_path=self.schema_path,
            ns_groups_path=self.ns_groups_path,
            train_config=self.train_config,
        )

    def _save_step_checkpoint(
        self,
        global_step: int,
        is_best: bool = False,
        skip_model_file: bool = False,
    ) -> Path:
        checkpoint_dir = self.save_dir / self._build_step_dir_name(global_step, is_best=is_best)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        if not skip_model_file:
            torch.save(self.model.state_dict(), checkpoint_dir / "model.pt")
        self._write_sidecar_files(checkpoint_dir)
        logging.info("Saved checkpoint to %s", checkpoint_dir / "model.pt")
        return checkpoint_dir

    def _remove_old_best_dirs(self) -> None:
        for old_dir in self.save_dir.glob("global_step*.best_model"):
            shutil.rmtree(old_dir)
            logging.info("Removed old best_model dir: %s", old_dir)

    def _batch_to_device(self, batch: dict[str, Any]) -> dict[str, Any]:
        device_batch: dict[str, Any] = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                device_batch[key] = value.to(self.device, non_blocking=True)
            else:
                device_batch[key] = value
        return device_batch

    def _handle_validation_result(
        self,
        total_step: int,
        val_auc: float,
        val_logloss: float,
    ) -> None:
        old_best = self.early_stopping.best_score
        is_likely_new_best = (
            old_best is None
            or val_auc > old_best + self.early_stopping.delta
        )
        if not is_likely_new_best:
            self.early_stopping(val_auc, self.model, {
                "best_val_AUC": val_auc,
                "best_val_logloss": val_logloss,
                "best_val_score_diagnostics": self.last_eval_diagnostics,
            })
            return

        best_dir = self.save_dir / self._build_step_dir_name(total_step, is_best=True)
        self.early_stopping.checkpoint_path = str(best_dir / "model.pt")
        self._remove_old_best_dirs()

        self.early_stopping(val_auc, self.model, {
            "best_val_AUC": val_auc,
            "best_val_logloss": val_logloss,
            "best_val_score_diagnostics": self.last_eval_diagnostics,
        })

        if self.early_stopping.best_score != old_best and Path(self.early_stopping.checkpoint_path).exists():
            self._save_step_checkpoint(total_step, is_best=True, skip_model_file=True)

    def _log_loop_progress(
        self,
        phase: str,
        current_batch: int,
        total_batches: int,
        *,
        epoch: int | None = None,
        loop_started_at: float | None = None,
        loss: float | None = None,
    ) -> None:
        prefix = f"{phase} progress"
        if epoch is not None and epoch > 0:
            prefix = f"{phase} epoch {epoch} progress"

        message = f"{prefix} {current_batch}/{total_batches} ({current_batch / total_batches:.1%})"
        if loop_started_at is not None and current_batch > 0:
            elapsed_seconds = max(0.0, time.monotonic() - loop_started_at)
            eta_seconds = elapsed_seconds * max(0, total_batches - current_batch) / current_batch
            message = f"{message} | eta={_format_duration(eta_seconds)}"
        if loss is not None:
            message = f"{message} | loss={loss:.4f}"
        logging.info(message)

    def _write_eval_diagnostics(self, total_step: int) -> None:
        if self.writer is None:
            return
        for name, value in self.last_eval_diagnostics.items():
            numeric_value = float(value)
            if np.isfinite(numeric_value):
                self.writer.add_scalar(f"Diagnostics/valid/{name}", numeric_value, total_step)

    def train(self) -> None:
        print("Start training (PCVR pointwise)")
        self.model.train()
        total_step = 0

        for epoch in range(1, self.num_epochs + 1):
            total_train_batches = len(self.train_loader)
            use_tqdm = _use_interactive_progress()
            log_interval = _progress_log_interval(total_train_batches)
            loop_started_at = time.monotonic()
            train_iter = enumerate(self.train_loader)
            train_pbar = (
                tqdm(train_iter, total=total_train_batches, dynamic_ncols=True)
                if use_tqdm
                else train_iter
            )
            loss_sum = 0.0

            for step_index, batch in train_pbar:
                loss = self._train_step(batch)
                total_step += 1
                loss_sum += loss

                if self.writer:
                    self.writer.add_scalar("Loss/train", loss, total_step)

                current_batch = step_index + 1
                if use_tqdm:
                    train_pbar.set_postfix({"loss": f"{loss:.4f}"})
                elif _should_log_progress(current_batch, total_train_batches, log_interval):
                    self._log_loop_progress(
                        "Train",
                        current_batch,
                        total_train_batches,
                        epoch=epoch,
                        loop_started_at=loop_started_at,
                        loss=loss,
                    )

                if self.eval_every_n_steps > 0 and total_step % self.eval_every_n_steps == 0:
                    logging.info("Evaluating at step %s", total_step)
                    val_auc, val_logloss = self.evaluate(epoch=epoch)
                    self.model.train()
                    torch.cuda.empty_cache()

                    logging.info("Step %s Validation | AUC: %s, LogLoss: %s", total_step, val_auc, val_logloss)

                    if self.writer:
                        self.writer.add_scalar("AUC/valid", val_auc, total_step)
                        self.writer.add_scalar("LogLoss/valid", val_logloss, total_step)
                        self._write_eval_diagnostics(total_step)

                    self._handle_validation_result(total_step, val_auc, val_logloss)

                    if self.early_stopping.early_stop:
                        logging.info("Early stopping at step %s", total_step)
                        return

            if use_tqdm:
                train_pbar.close()

            logging.info("Epoch %s, Average Loss: %s", epoch, loss_sum / len(self.train_loader))

            val_auc, val_logloss = self.evaluate(epoch=epoch)
            self.model.train()
            torch.cuda.empty_cache()

            logging.info("Epoch %s Validation | AUC: %s, LogLoss: %s", epoch, val_auc, val_logloss)

            if self.writer:
                self.writer.add_scalar("AUC/valid", val_auc, total_step)
                self.writer.add_scalar("LogLoss/valid", val_logloss, total_step)
                self._write_eval_diagnostics(total_step)

            self._handle_validation_result(total_step, val_auc, val_logloss)

            if self.early_stopping.early_stop:
                logging.info("Early stopping at epoch %s", epoch)
                break

            if epoch >= self.reinit_sparse_after_epoch and self.sparse_optimizer is not None:
                old_state: dict[int, Any] = {}
                for group in self.sparse_optimizer.param_groups:
                    for parameter in group["params"]:
                        if parameter.data_ptr() in self.sparse_optimizer.state:
                            old_state[parameter.data_ptr()] = self.sparse_optimizer.state[parameter]

                reinit_ptrs = self.model.reinit_high_cardinality_params(self.reinit_cardinality_threshold)
                sparse_params = self.model.get_sparse_params()
                self.sparse_optimizer = torch.optim.Adagrad(
                    sparse_params, lr=self.sparse_lr, weight_decay=self.sparse_weight_decay
                )
                restored = 0
                for parameter in sparse_params:
                    if parameter.data_ptr() not in reinit_ptrs and parameter.data_ptr() in old_state:
                        self.sparse_optimizer.state[parameter] = old_state[parameter.data_ptr()]
                        restored += 1
                logging.info(
                    "Rebuilt Adagrad optimizer after epoch %s, restored optimizer state for "
                    "%s low-cardinality params",
                    epoch,
                    restored,
                )

    def _make_model_input(self, device_batch: dict[str, Any]) -> Any:
        return batch_to_model_input(device_batch, self.model_input_type, torch.device(self.device))

    def _train_step(self, batch: dict[str, Any]) -> float:
        device_batch = self._batch_to_device(batch)
        label = device_batch["label"].float()

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

        return loss.item()

    def evaluate(self, epoch: int | None = None) -> tuple[float, float]:
        print("Start Evaluation (PCVR pointwise) - validation")
        self.model.eval()
        if epoch is None:
            epoch = -1

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
                        epoch=epoch,
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
            logging.warning("[Evaluate] %s/%s predictions are NaN, filtering them out", n_nan, len(probabilities))
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
        logging.info(
            "Validation score diagnostics | pos=%s neg=%s pos_mean=%.6f neg_mean=%.6f margin=%.6f score_std=%.6f",
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