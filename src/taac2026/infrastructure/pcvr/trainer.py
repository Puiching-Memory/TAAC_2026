"""Shared PCVR pointwise trainer."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from taac2026.infrastructure.checkpoints import build_checkpoint_dir_name, write_checkpoint_sidecars
from taac2026.infrastructure.pcvr.protocol import batch_to_model_input
from taac2026.infrastructure.training.runtime import EarlyStopping, sigmoid_focal_loss


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
        loss_type: str = "bce",
        focal_alpha: float = 0.1,
        focal_gamma: float = 2.0,
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
    ) -> None:
        self.model = model
        self.model_input_type = model_input_type
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.writer = writer
        self.schema_path = Path(schema_path).expanduser().resolve() if schema_path else None
        self.ns_groups_path = Path(ns_groups_path).expanduser().resolve() if ns_groups_path else None

        self.sparse_optimizer: torch.optim.Optimizer | None
        if hasattr(model, "get_sparse_params"):
            sparse_params = model.get_sparse_params()
            dense_params = model.get_dense_params()
            if not sparse_params:
                logging.info("Model exposes get_sparse_params but has no embedding parameters; using AdamW for all params")
                self.sparse_optimizer = None
                self.dense_optimizer = torch.optim.AdamW(
                    model.parameters(), lr=lr, betas=(0.9, 0.98)
                )
            else:
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
                self.dense_optimizer: torch.optim.Optimizer = torch.optim.AdamW(
                    dense_params, lr=lr, betas=(0.9, 0.98)
                )
        else:
            self.sparse_optimizer = None
            self.dense_optimizer = torch.optim.AdamW(
                model.parameters(), lr=lr, betas=(0.9, 0.98)
            )

        self.num_epochs = num_epochs
        self.device = device
        self.save_dir = Path(save_dir).expanduser().resolve()
        self.early_stopping = early_stopping
        self.loss_type = loss_type
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.reinit_sparse_after_epoch = reinit_sparse_after_epoch
        self.reinit_cardinality_threshold = reinit_cardinality_threshold
        self.sparse_lr = sparse_lr
        self.sparse_weight_decay = sparse_weight_decay
        self.ckpt_params = ckpt_params or {}
        self.eval_every_n_steps = eval_every_n_steps
        self.train_config = train_config

        logging.info(
            "PCVRPointwiseTrainer loss_type=%s, focal_alpha=%s, focal_gamma=%s, "
            "reinit_sparse_after_epoch=%s",
            loss_type,
            focal_alpha,
            focal_gamma,
            reinit_sparse_after_epoch,
        )

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
            })
            return

        best_dir = self.save_dir / self._build_step_dir_name(total_step, is_best=True)
        self.early_stopping.checkpoint_path = str(best_dir / "model.pt")
        self._remove_old_best_dirs()

        self.early_stopping(val_auc, self.model, {
            "best_val_AUC": val_auc,
            "best_val_logloss": val_logloss,
        })

        if self.early_stopping.best_score != old_best and Path(self.early_stopping.checkpoint_path).exists():
            self._save_step_checkpoint(total_step, is_best=True, skip_model_file=True)

    def train(self) -> None:
        print("Start training (PCVR pointwise)")
        self.model.train()
        total_step = 0

        for epoch in range(1, self.num_epochs + 1):
            train_pbar = tqdm(enumerate(self.train_loader), total=len(self.train_loader), dynamic_ncols=True)
            loss_sum = 0.0

            for _step, batch in train_pbar:
                loss = self._train_step(batch)
                total_step += 1
                loss_sum += loss

                if self.writer:
                    self.writer.add_scalar("Loss/train", loss, total_step)

                train_pbar.set_postfix({"loss": f"{loss:.4f}"})

                if self.eval_every_n_steps > 0 and total_step % self.eval_every_n_steps == 0:
                    logging.info("Evaluating at step %s", total_step)
                    val_auc, val_logloss = self.evaluate(epoch=epoch)
                    self.model.train()
                    torch.cuda.empty_cache()

                    logging.info("Step %s Validation | AUC: %s, LogLoss: %s", total_step, val_auc, val_logloss)

                    if self.writer:
                        self.writer.add_scalar("AUC/valid", val_auc, total_step)
                        self.writer.add_scalar("LogLoss/valid", val_logloss, total_step)

                    self._handle_validation_result(total_step, val_auc, val_logloss)

                    if self.early_stopping.early_stop:
                        logging.info("Early stopping at step %s", total_step)
                        return

            logging.info("Epoch %s, Average Loss: %s", epoch, loss_sum / len(self.train_loader))

            val_auc, val_logloss = self.evaluate(epoch=epoch)
            self.model.train()
            torch.cuda.empty_cache()

            logging.info("Epoch %s Validation | AUC: %s, LogLoss: %s", epoch, val_auc, val_logloss)

            if self.writer:
                self.writer.add_scalar("AUC/valid", val_auc, total_step)
                self.writer.add_scalar("LogLoss/valid", val_logloss, total_step)

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
        logits = self.model(model_input).squeeze(-1)

        if self.loss_type == "focal":
            loss = sigmoid_focal_loss(logits, label, alpha=self.focal_alpha, gamma=self.focal_gamma)
        else:
            loss = F.binary_cross_entropy_with_logits(logits, label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0, foreach=False)

        self.dense_optimizer.step()
        if self.sparse_optimizer is not None:
            self.sparse_optimizer.step()

        return loss.item()

    def evaluate(self, epoch: int | None = None) -> tuple[float, float]:
        print("Start Evaluation (PCVR pointwise) - validation")
        self.model.eval()
        if epoch is None:
            epoch = -1

        pbar = tqdm(enumerate(self.valid_loader), total=len(self.valid_loader))
        all_logits_list = []
        all_labels_list = []

        with torch.no_grad():
            for _step, batch in pbar:
                logits, labels = self._evaluate_step(batch)
                all_logits_list.append(logits.detach().cpu())
                all_labels_list.append(labels.detach().cpu())

        all_logits = torch.cat(all_logits_list, dim=0)
        all_labels = torch.cat(all_labels_list, dim=0).long()

        probabilities = torch.sigmoid(all_logits).numpy()
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

        return auc, logloss

    def _evaluate_step(self, batch: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        device_batch = self._batch_to_device(batch)
        label = device_batch["label"]

        model_input = self._make_model_input(device_batch)
        logits, _embeddings = self.model.predict(model_input)
        logits = logits.squeeze(-1)

        return logits, label