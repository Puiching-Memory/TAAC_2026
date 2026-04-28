"""Shared training runtime helpers."""

from __future__ import annotations

import copy
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
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


AMP_DTYPE_CHOICES: tuple[str, ...] = ("bfloat16", "float16")
BINARY_LOSS_TYPE_CHOICES: tuple[str, ...] = ("bce", "focal")
DENSE_OPTIMIZER_TYPE_CHOICES: tuple[str, ...] = ("adamw", "orthogonal_adamw")
_AMP_DTYPE_ALIASES = {
    "bf16": "bfloat16",
    "bfloat16": "bfloat16",
    "fp16": "float16",
    "float16": "float16",
    "half": "float16",
}


def normalize_amp_dtype(value: str | None) -> str:
    if value is None:
        return "bfloat16"
    normalized = str(value).strip().lower()
    try:
        return _AMP_DTYPE_ALIASES[normalized]
    except KeyError as error:
        raise ValueError(f"unsupported amp dtype: {value}") from error


def amp_dtype_to_torch_dtype(value: str | None) -> torch.dtype:
    normalized = normalize_amp_dtype(value)
    if normalized == "bfloat16":
        return torch.bfloat16
    return torch.float16


def _device_type(device: str | torch.device) -> str:
    return device.type if isinstance(device, torch.device) else torch.device(device).type


@dataclass(frozen=True, slots=True)
class RuntimeExecutionConfig:
    amp: bool = False
    amp_dtype: str = "bfloat16"
    compile: bool = False

    def normalized_amp_dtype(self) -> str:
        return normalize_amp_dtype(self.amp_dtype)

    def torch_amp_dtype(self) -> torch.dtype:
        return amp_dtype_to_torch_dtype(self.amp_dtype)

    def amp_enabled_for(self, device: str | torch.device) -> bool:
        return self.amp and _device_type(device) == "cuda" and torch.cuda.is_available()

    def grad_scaler_enabled_for(self, device: str | torch.device) -> bool:
        return self.amp_enabled_for(device) and self.torch_amp_dtype() == torch.float16

    def autocast_context(self, device: str | torch.device) -> AbstractContextManager[None]:
        if not self.amp_enabled_for(device):
            return nullcontext()
        return torch.autocast(device_type="cuda", dtype=self.torch_amp_dtype())

    def summary(self, device: str | torch.device) -> str:
        return (
            f"amp={self.amp} (effective={self.amp_enabled_for(device)}), "
            f"amp_dtype={self.normalized_amp_dtype()}, compile={self.compile}"
        )


@dataclass(frozen=True, slots=True)
class BinaryClassificationLossConfig:
    loss_type: str = "bce"
    focal_alpha: float = 0.1
    focal_gamma: float = 2.0
    pairwise_auc_weight: float = 0.0
    pairwise_auc_temperature: float = 1.0

    def __post_init__(self) -> None:
        normalized_loss_type = str(self.loss_type).strip().lower()
        if normalized_loss_type not in BINARY_LOSS_TYPE_CHOICES:
            raise ValueError(f"unsupported loss_type: {self.loss_type}")

        focal_alpha = float(self.focal_alpha)
        if not 0.0 <= focal_alpha <= 1.0:
            raise ValueError(f"focal_alpha must be between 0 and 1, got {self.focal_alpha}")

        focal_gamma = float(self.focal_gamma)
        if focal_gamma < 0.0:
            raise ValueError(f"focal_gamma must be >= 0, got {self.focal_gamma}")

        pairwise_auc_weight = float(self.pairwise_auc_weight)
        if pairwise_auc_weight < 0.0:
            raise ValueError(f"pairwise_auc_weight must be >= 0, got {self.pairwise_auc_weight}")

        pairwise_auc_temperature = float(self.pairwise_auc_temperature)
        if pairwise_auc_temperature <= 0.0:
            raise ValueError(f"pairwise_auc_temperature must be > 0, got {self.pairwise_auc_temperature}")

        object.__setattr__(self, "loss_type", normalized_loss_type)
        object.__setattr__(self, "focal_alpha", focal_alpha)
        object.__setattr__(self, "focal_gamma", focal_gamma)
        object.__setattr__(self, "pairwise_auc_weight", pairwise_auc_weight)
        object.__setattr__(self, "pairwise_auc_temperature", pairwise_auc_temperature)


DEFAULT_BINARY_CLASSIFICATION_LOSS_CONFIG = BinaryClassificationLossConfig()


def create_grad_scaler(
    runtime_execution: RuntimeExecutionConfig,
    device: str | torch.device,
) -> torch.amp.GradScaler | None:
    if not runtime_execution.grad_scaler_enabled_for(device):
        return None
    return torch.amp.GradScaler(device="cuda", enabled=True)


def maybe_compile_callable(callable_obj, *, enabled: bool, label: str):
    if not enabled:
        return callable_obj
    try:
        return torch.compile(callable_obj)
    except Exception as error:  # pragma: no cover - exercised via monkeypatched tests.
        logging.warning("Failed to compile %s; falling back to eager execution: %s", label, error)
        return callable_obj


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


def compute_binary_classification_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    loss_config: BinaryClassificationLossConfig | None = None,
    reduction: str = "mean",
) -> torch.Tensor:
    """Compute the shared binary classification loss from raw logits."""

    resolved_loss_config = loss_config or DEFAULT_BINARY_CLASSIFICATION_LOSS_CONFIG
    if resolved_loss_config.loss_type == "focal":
        loss = sigmoid_focal_loss(
            logits,
            targets,
            alpha=resolved_loss_config.focal_alpha,
            gamma=resolved_loss_config.focal_gamma,
            reduction=reduction,
        )
    else:
        loss = F.binary_cross_entropy_with_logits(logits, targets, reduction=reduction)
    if reduction == "none" or resolved_loss_config.pairwise_auc_weight <= 0.0:
        return loss
    return loss + resolved_loss_config.pairwise_auc_weight * binary_pairwise_auc_loss(
        logits,
        targets,
        temperature=resolved_loss_config.pairwise_auc_temperature,
    )


def binary_pairwise_auc_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Pairwise softplus ranking loss for improving positive-negative ordering."""

    flat_logits = logits.reshape(-1)
    flat_targets = targets.reshape(-1)
    positive_logits = flat_logits[flat_targets > 0.5]
    negative_logits = flat_logits[flat_targets <= 0.5]
    if positive_logits.numel() == 0 or negative_logits.numel() == 0:
        return flat_logits.sum() * 0.0
    pairwise_margin = (positive_logits[:, None] - negative_logits[None, :]) / float(temperature)
    return F.softplus(-pairwise_margin).mean()