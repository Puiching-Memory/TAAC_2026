"""Support mixin extracted from the shared PCVR pointwise trainer."""

from __future__ import annotations

import logging
import shutil
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from taac2026.infrastructure.checkpoints import (
    build_checkpoint_dir_name,
    preferred_checkpoint_path,
    save_checkpoint_state_dict,
    write_checkpoint_sidecars,
)
from taac2026.domain.sidecar import build_pcvr_train_config_sidecar
from taac2026.infrastructure.optimization.registry import build_dense_optimizer, dense_optimizer_display_name
from taac2026.infrastructure.optimization.schedules import dense_lr_multiplier
from taac2026.infrastructure.optimization.transforms import orthogonalize_gradient


class PCVRTrainerSupportMixin:
    def _dense_optimizer_display_name(self) -> str:
        return dense_optimizer_display_name(self.dense_optimizer_type)

    def _build_dense_optimizer(self, parameters: list[nn.Parameter], lr: float) -> torch.optim.Optimizer:
        return build_dense_optimizer(parameters, dense_optimizer_type=self.dense_optimizer_type, lr=lr)

    def _dense_lr_multiplier(self, step: int) -> float:
        return dense_lr_multiplier(
            step,
            scheduler_type=self.scheduler_type,
            max_steps=self.max_steps,
            warmup_steps=self.warmup_steps,
            min_lr_ratio=self.min_lr_ratio,
        )

    def _set_dense_learning_rate(self, step: int) -> float:
        lr_multiplier = self._dense_lr_multiplier(step)
        self.current_dense_lr = self.base_dense_lr * lr_multiplier
        for group in self.dense_optimizer.param_groups:
            group["lr"] = self.current_dense_lr
        return self.current_dense_lr

    def _orthogonalize_dense_gradients(self) -> None:
        if self.dense_optimizer_type != "orthogonal_adamw":
            return
        for parameter in self.dense_params:
            gradient = parameter.grad
            if gradient is None or gradient.ndim < 2:
                continue
            orthogonalize_gradient(gradient)

    def _build_step_dir_name(self, global_step: int, is_best: bool = False) -> str:
        return build_checkpoint_dir_name(global_step, self.ckpt_params, is_best=is_best)

    def _write_sidecar_files(self, checkpoint_dir: Path) -> None:
        write_checkpoint_sidecars(
            checkpoint_dir,
            schema_path=self.schema_path,
            train_config=build_pcvr_train_config_sidecar(self.train_config),
        )

    def _save_step_checkpoint(
        self,
        global_step: int,
        is_best: bool = False,
        skip_model_file: bool = False,
    ) -> Path:
        checkpoint_dir = self.save_dir / self._build_step_dir_name(global_step, is_best=is_best)
        if not skip_model_file:
            save_checkpoint_state_dict(self.model.state_dict(), checkpoint_dir)
        self._write_sidecar_files(checkpoint_dir)
        logging.info("Saved checkpoint to %s", preferred_checkpoint_path(checkpoint_dir))
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
            }, step=total_step)
            return

        best_dir = self.save_dir / self._build_step_dir_name(total_step, is_best=True)
        self.early_stopping.checkpoint_path = str(preferred_checkpoint_path(best_dir))
        self._remove_old_best_dirs()

        self.early_stopping(val_auc, self.model, {
            "best_val_AUC": val_auc,
            "best_val_logloss": val_logloss,
            "best_val_score_diagnostics": self.last_eval_diagnostics,
        }, step=total_step)

        if self.early_stopping.best_score != old_best and Path(self.early_stopping.checkpoint_path).exists():
            self._save_step_checkpoint(total_step, is_best=True, skip_model_file=True)

    def _infinite_train_batches(self) -> Iterator[dict[str, Any]]:
        while True:
            yielded = False
            for batch in self.train_loader:
                yielded = True
                yield batch
            if not yielded:
                raise RuntimeError("train_loader produced no batches")

    def _logical_train_sweep_steps(self) -> int:
        try:
            return max(1, len(self.train_loader))
        except TypeError:
            return 1

    def _rebuild_sparse_optimizer(self, total_step: int) -> None:
        if self.sparse_optimizer is None:
            return
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
            "Reinitialized sparse optimizer at step %d (%d params reset, %d preserved)",
            total_step,
            len(reinit_ptrs),
            restored,
        )

    def _write_eval_diagnostics(self, total_step: int) -> None:
        if self.writer is None or not self.last_eval_diagnostics:
            return
        for metric_name, value in self.last_eval_diagnostics.items():
            self.writer.add_scalar(f"score/{metric_name}", float(value), total_step)
        self.writer.flush()


__all__ = ["PCVRTrainerSupportMixin"]