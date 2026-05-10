"""Shared training runtime helpers."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from contextlib import AbstractContextManager, nullcontext
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from taac2026.domain.runtime_config import (
    AMP_DTYPE_CHOICES,
    DEFAULT_PCVR_LOSS_CONFIG,
    DEFAULT_PROGRESS_LOG_INTERVAL_STEPS,
    DENSE_OPTIMIZER_TYPE_CHOICES,
    PCVR_LOSS_TERM_KIND_CHOICES,
    PCVRLossConfig,
    PCVRLossTermConfig,
    RuntimeExecutionConfig,
    normalize_amp_dtype,
    parse_pcvr_loss_config_arg,
)
from taac2026.infrastructure.logging import configure_logging, logger
from taac2026.infrastructure.checkpoints import save_checkpoint_state_dict


def amp_dtype_to_torch_dtype(value: str | None) -> torch.dtype:
    normalized = normalize_amp_dtype(value)
    if normalized == "bfloat16":
        return torch.bfloat16
    return torch.float16


def runtime_device_type(device: str | torch.device) -> str:
    return device.type if isinstance(device, torch.device) else torch.device(device).type


def runtime_torch_amp_dtype(runtime_execution: RuntimeExecutionConfig) -> torch.dtype:
    return amp_dtype_to_torch_dtype(runtime_execution.amp_dtype)


def runtime_amp_enabled(runtime_execution: RuntimeExecutionConfig, device: str | torch.device) -> bool:
    return bool(runtime_execution.amp) and runtime_device_type(device) == "cuda" and torch.cuda.is_available()


def runtime_grad_scaler_enabled(runtime_execution: RuntimeExecutionConfig, device: str | torch.device) -> bool:
    return runtime_amp_enabled(runtime_execution, device) and runtime_torch_amp_dtype(runtime_execution) == torch.float16


def runtime_autocast_context(
    runtime_execution: RuntimeExecutionConfig,
    device: str | torch.device,
) -> AbstractContextManager[None]:
    if not runtime_amp_enabled(runtime_execution, device):
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=runtime_torch_amp_dtype(runtime_execution))


def runtime_execution_summary(runtime_execution: RuntimeExecutionConfig, device: str | torch.device) -> str:
    return (
        f"amp={runtime_execution.amp} (effective={runtime_amp_enabled(runtime_execution, device)}), "
        f"amp_dtype={runtime_execution.normalized_amp_dtype()}, compile={runtime_execution.compile}, "
        f"progress_log_interval_steps={runtime_execution.progress_log_interval_steps}, "
        f"deterministic={runtime_execution.deterministic}"
    )


def create_grad_scaler(
    runtime_execution: RuntimeExecutionConfig,
    device: str | torch.device,
) -> torch.amp.GradScaler | None:
    if not runtime_grad_scaler_enabled(runtime_execution, device):
        return None
    return torch.amp.GradScaler(device="cuda", enabled=True)


def maybe_compile_callable(callable_obj, *, enabled: bool, label: str):
    if not enabled:
        return callable_obj
    try:
        return torch.compile(callable_obj)
    except Exception as error:  # pragma: no cover - exercised via monkeypatched tests.
        logger.warning("Failed to compile {}; falling back to eager execution: {}", label, error)
        return callable_obj


def maybe_prepare_internal_compile(model: nn.Module, *, enabled: bool, label: str) -> bool:
    if not enabled or not bool(getattr(model, "uses_internal_compile", False)):
        return False
    prepare = getattr(model, "prepare_for_runtime_compile", None)
    if not callable(prepare):
        return False
    try:
        prepare()
    except Exception as error:  # pragma: no cover - exercised via monkeypatched tests.
        logger.warning("Failed to prepare internal compile for {}; falling back to eager execution: {}", label, error)
    return True


def create_logger(filepath: str | Path):
    """Configure shared loguru sinks for a training or evaluation process."""

    configure_logging(filepath)
    return logger


class EarlyStopping:
    """Early-stop training when a higher-is-better validation metric plateaus by step."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        label: str = "",
        patience_steps: int = 25_000,
        verbose: bool = False,
        delta: float = 0,
    ) -> None:
        self.checkpoint_path = str(checkpoint_path)
        self.patience_steps = int(patience_steps)
        if self.patience_steps < 0:
            raise ValueError("patience_steps must be non-negative")
        self.verbose = verbose
        self.counter = 0
        self.best_score: float | None = None
        self.early_stop = False
        self.delta = delta
        self.best_model: dict[str, torch.Tensor] | None = None
        self.best_saved_score = 0.0
        self.best_extra_metrics: dict[str, Any] | None = None
        self.label = f"{label} " if label else ""
        self.best_step: int | None = None

    @property
    def resolved_patience(self) -> int:
        return self.patience_steps

    def _resolve_current_step(self, step: int | None) -> int:
        if step is None:
            raise ValueError("step is required for step-based early stopping")
        return int(step)

    def _is_not_improved(self, score: float) -> bool:
        assert self.best_score is not None, "call __call__ first to seed best_score"
        return score <= self.best_score + self.delta

    def __call__(
        self,
        score: float,
        model: nn.Module,
        extra_metrics: dict[str, Any] | None = None,
        step: int | None = None,
    ) -> None:
        current_step = self._resolve_current_step(step)
        if self.best_score is None:
            self.best_score = score
            self.best_extra_metrics = extra_metrics
            self.best_saved_score = 0.0
            self.best_step = current_step
            self.save_checkpoint(score, model)
            self.best_model = copy.deepcopy(model.state_dict())
        elif self._is_not_improved(score):
            assert self.best_step is not None
            self.counter = max(0, current_step - self.best_step)
            logger.info(
                "{}earlyStopping counter: {} / {} steps",
                self.label,
                self.counter,
                self.resolved_patience,
            )
            if self.counter >= self.resolved_patience:
                self.early_stop = True
        else:
            logger.info("{}earlyStopping counter reset!", self.label)
            self.best_score = score
            self.best_model = copy.deepcopy(model.state_dict())
            self.best_extra_metrics = extra_metrics
            self.best_step = current_step
            self.save_checkpoint(score, model)
            self.counter = 0

    def save_checkpoint(self, score: float, model: nn.Module) -> None:
        if self.verbose:
            logger.info("Validation score increased. Saving model ...")
        checkpoint_path = Path(self.checkpoint_path)
        save_checkpoint_state_dict(model.state_dict(), checkpoint_path)
        self.best_saved_score = score


def set_seed(seed: int, *, deterministic: bool = True) -> None:
    """Seed Python, NumPy, and PyTorch RNGs for reproducible training."""

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = bool(deterministic)


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


def _zero_loss_like(logits: torch.Tensor) -> torch.Tensor:
    return logits.reshape(-1).sum() * 0.0


def _resolve_model_loss_terms(model: nn.Module | None) -> Mapping[str, torch.Tensor]:
    if model is None:
        return {}
    loss_terms = getattr(model, "pcvr_loss_terms", None)
    if not callable(loss_terms):
        return {}
    resolved = loss_terms()
    if resolved is None:
        return {}
    if not isinstance(resolved, Mapping):
        raise TypeError(f"pcvr_loss_terms() must return a mapping, got {type(resolved).__name__}")
    return resolved


def compute_pcvr_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    loss_config: PCVRLossConfig | None = None,
    *,
    model: nn.Module | None = None,
    reduction: str = "mean",
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute weighted PCVR loss terms from raw logits and optional model losses."""

    resolved_loss_config = loss_config or DEFAULT_PCVR_LOSS_CONFIG
    model_terms: Mapping[str, torch.Tensor] | None = None
    total_loss = _zero_loss_like(logits)
    components: dict[str, torch.Tensor] = {}

    for term in resolved_loss_config.terms:
        if term.weight == 0.0:
            components[term.name] = _zero_loss_like(logits).detach()
            continue
        if term.kind == "bce":
            raw_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction=reduction)
        elif term.kind == "focal":
            raw_loss = sigmoid_focal_loss(
                logits,
                targets,
                alpha=term.focal_alpha,
                gamma=term.focal_gamma,
                reduction=reduction,
            )
        elif term.kind == "pairwise_auc":
            if reduction == "none":
                raise ValueError("pairwise_auc loss does not support reduction='none'")
            raw_loss = binary_pairwise_auc_loss(logits, targets, temperature=term.temperature)
        elif term.kind == "model":
            if model_terms is None:
                model_terms = _resolve_model_loss_terms(model)
            try:
                raw_loss = model_terms[term.name]
            except KeyError as error:
                available = ", ".join(sorted(model_terms)) or "<none>"
                message = f"model did not provide configured loss term {term.name!r}; available: {available}"
                raise KeyError(message) from error
            if not isinstance(raw_loss, torch.Tensor):
                raw_loss = logits.new_tensor(float(raw_loss))
        else:  # pragma: no cover - PCVRLossTermConfig validates this.
            raise ValueError(f"unsupported PCVR loss term kind: {term.kind}")

        weighted_loss = raw_loss * term.weight
        total_loss = total_loss + weighted_loss
        components[term.name] = raw_loss.detach()

    return total_loss, components


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


__all__ = [
    "AMP_DTYPE_CHOICES",
    "DEFAULT_PCVR_LOSS_CONFIG",
    "DEFAULT_PROGRESS_LOG_INTERVAL_STEPS",
    "DENSE_OPTIMIZER_TYPE_CHOICES",
    "PCVR_LOSS_TERM_KIND_CHOICES",
    "EarlyStopping",
    "PCVRLossConfig",
    "PCVRLossTermConfig",
    "RuntimeExecutionConfig",
    "amp_dtype_to_torch_dtype",
    "binary_pairwise_auc_loss",
    "compute_pcvr_loss",
    "create_grad_scaler",
    "create_logger",
    "maybe_compile_callable",
    "maybe_prepare_internal_compile",
    "normalize_amp_dtype",
    "parse_pcvr_loss_config_arg",
    "runtime_amp_enabled",
    "runtime_autocast_context",
    "runtime_device_type",
    "runtime_execution_summary",
    "runtime_grad_scaler_enabled",
    "runtime_torch_amp_dtype",
    "set_seed",
    "sigmoid_focal_loss",
]
