"""Shared training runtime helpers."""

from __future__ import annotations

import copy
import logging
import os
import random
import time
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class LogFormatter(logging.Formatter):
    """Log formatter that includes wall-clock and elapsed run time."""

    def __init__(self) -> None:
        super().__init__()
        self.start_time = time.time()

    def format(self, record: logging.LogRecord) -> str:
        elapsed_seconds = round(record.created - self.start_time)
        prefix = f"{time.strftime('%x %X')} - {timedelta(seconds=elapsed_seconds)}"
        message = record.getMessage().replace("\n", "\n" + " " * (len(prefix) + 3))
        return f"{prefix} - {message}"


def create_logger(filepath: str | Path) -> logging.Logger:
    """Configure the root logger for a training or evaluation process."""

    log_path = Path(filepath)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_formatter = LogFormatter()

    file_handler = logging.FileHandler(log_path, "w")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(log_formatter)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(log_formatter)

    logger = logging.getLogger()
    logger.handlers = []
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    def reset_time() -> None:
        log_formatter.start_time = time.time()

    logger.reset_time = reset_time  # type: ignore[attr-defined]
    return logger


class EarlyStopping:
    """Early-stop training when a higher-is-better validation metric plateaus."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        label: str = "",
        patience: int = 5,
        verbose: bool = False,
        delta: float = 0,
    ) -> None:
        self.checkpoint_path = str(checkpoint_path)
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score: float | None = None
        self.early_stop = False
        self.delta = delta
        self.best_model: dict[str, torch.Tensor] | None = None
        self.best_saved_score = 0.0
        self.best_extra_metrics: dict[str, Any] | None = None
        self.label = f"{label} " if label else ""

    def _is_not_improved(self, score: float) -> bool:
        assert self.best_score is not None, "call __call__ first to seed best_score"
        return score <= self.best_score + self.delta

    def __call__(
        self,
        score: float,
        model: nn.Module,
        extra_metrics: dict[str, Any] | None = None,
    ) -> None:
        if self.best_score is None:
            self.best_score = score
            self.best_extra_metrics = extra_metrics
            self.best_saved_score = 0.0
            self.save_checkpoint(score, model)
            self.best_model = copy.deepcopy(model.state_dict())
        elif self._is_not_improved(score):
            self.counter += 1
            logging.info("%searlyStopping counter: %s / %s", self.label, self.counter, self.patience)
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            logging.info("%searlyStopping counter reset!", self.label)
            self.best_score = score
            self.best_model = copy.deepcopy(model.state_dict())
            self.best_extra_metrics = extra_metrics
            self.save_checkpoint(score, model)
            self.counter = 0

    def save_checkpoint(self, score: float, model: nn.Module) -> None:
        if self.verbose:
            logging.info("Validation score increased. Saving model ...")
        checkpoint_path = Path(self.checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), checkpoint_path)
        self.best_saved_score = score


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch RNGs for reproducible training."""

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def sigmoid_focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.1,
    gamma: float = 2.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """Compute binary sigmoid focal loss from raw logits."""

    probabilities = torch.sigmoid(logits)
    bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = probabilities * targets + (1 - probabilities) * (1 - targets)
    focal_weight = (1 - p_t) ** gamma
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    loss = alpha_t * focal_weight * bce_loss
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    return loss