"""Support mixin extracted from the shared PCVR pointwise trainer."""

from __future__ import annotations

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
from taac2026.infrastructure.logging import logger
from taac2026.infrastructure.optimization.registry import build_dense_optimizer, dense_optimizer_display_name
from taac2026.infrastructure.optimization.schedules import dense_lr_multiplier
from taac2026.infrastructure.optimization.transforms import orthogonalize_gradient
from taac2026.infrastructure.runtime.protocols import ReinitializableSparseParameterModel


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

    def _build_step_dir_name(
        self,
        global_step: int,
        checkpoint_params: dict[str, Any] | None = None,
    ) -> str:
        merged_params = dict(self.ckpt_params)
        if checkpoint_params:
            merged_params.update(checkpoint_params)
        return build_checkpoint_dir_name(global_step, merged_params)

    def _write_sidecar_files(self, checkpoint_dir: Path) -> None:
        write_checkpoint_sidecars(
            checkpoint_dir,
            schema_path=self.schema_path,
            train_config=self.train_config,
        )

    def _save_step_checkpoint(
        self,
        global_step: int,
        checkpoint_params: dict[str, Any] | None = None,
    ) -> Path:
        checkpoint_dir = self.save_dir / self._build_step_dir_name(global_step, checkpoint_params)
        save_checkpoint_state_dict(self._checkpoint_state_dict(), checkpoint_dir)
        self._write_sidecar_files(checkpoint_dir)
        logger.info("Saved checkpoint to {}", preferred_checkpoint_path(checkpoint_dir))
        return checkpoint_dir

    def _checkpoint_state_dict(self) -> dict[str, torch.Tensor]:
        ema = getattr(self, "ema", None)
        if ema is not None:
            return ema.state_dict()
        return self.model.state_dict()

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
        stop_score = self.validation_early_stopping_score(val_auc, val_logloss)
        logger.info(
            "Validation early stopping monitor | metric={} score={}",
            self.early_stopping_metric,
            stop_score,
        )
        extra_metrics = {
            "early_stopping_metric": self.early_stopping_metric,
            "early_stopping_score": stop_score,
            "best_val_AUC": val_auc,
            "best_val_logloss": val_logloss,
            "best_val_score_diagnostics": self.last_eval_diagnostics,
            "best_val_probe_metrics": getattr(self, "last_eval_probe_metrics", {}),
            "best_val_probe_score_diagnostics": getattr(self, "last_eval_probe_diagnostics", {}),
        }
        self.early_stopping(stop_score, self.model, extra_metrics, step=total_step)
        self._save_step_checkpoint(total_step, {"auc": val_auc})

    def _infinite_train_batches(self) -> Iterator[dict[str, Any]]:
        start_step = 0
        while True:
            self._set_train_loader_start_step(start_step)
            yielded = False
            yielded_steps = 0
            for batch in self.train_loader:
                yielded = True
                yielded_steps += 1
                yield batch
            if not yielded:
                raise RuntimeError("train_loader produced no batches")
            start_step += yielded_steps

    def _set_train_loader_start_step(self, start_step: int) -> None:
        sampler = getattr(self.train_loader, "sampler", None)
        sampler_set_start_step = getattr(sampler, "set_start_step", None)
        if callable(sampler_set_start_step):
            sampler_set_start_step(start_step)
            return
        dataset = getattr(self.train_loader, "dataset", None)
        dataset_set_start_step = getattr(dataset, "set_start_step", None)
        if callable(dataset_set_start_step):
            dataset_set_start_step(start_step)

    def _logical_train_sweep_steps(self) -> int:
        dataset = getattr(self.train_loader, "dataset", None)
        dataset_sweep_steps = getattr(dataset, "logical_sweep_steps", None)
        if callable(dataset_sweep_steps):
            return max(1, int(dataset_sweep_steps()))
        try:
            return max(1, len(self.train_loader))
        except TypeError:
            return 1

    def _rebuild_sparse_optimizer(self, total_step: int) -> None:
        if self.sparse_optimizer is None:
            return
        if not isinstance(self.model, ReinitializableSparseParameterModel):
            raise TypeError(
                "sparse optimizer reinitialization requires get_sparse_params() "
                "and reinit_high_cardinality_params()"
            )
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
        logger.info(
            "Reinitialized sparse optimizer at step {} ({} params reset, {} preserved)",
            total_step,
            len(reinit_ptrs),
            restored,
        )
        sync_ema = getattr(self, "_sync_ema_after_model_reinit", None)
        if callable(sync_ema):
            sync_ema()

    def _write_eval_diagnostics(self, total_step: int) -> None:
        del total_step


__all__ = ["PCVRTrainerSupportMixin"]
