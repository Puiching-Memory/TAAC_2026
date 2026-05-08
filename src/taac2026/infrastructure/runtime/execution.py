"""Shared training runtime helpers."""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
import json
import math
import os
import random
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from taac2026.infrastructure.logging import configure_logging, logger
from taac2026.infrastructure.checkpoints import save_checkpoint_state_dict


AMP_DTYPE_CHOICES: tuple[str, ...] = ("bfloat16", "float16")
PCVR_LOSS_TERM_KIND_CHOICES: tuple[str, ...] = ("bce", "focal", "pairwise_auc", "model")
DENSE_OPTIMIZER_TYPE_CHOICES: tuple[str, ...] = (
    "adamw",
    "fused_adamw",
    "orthogonal_adamw",
    "muon",
)
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
class PCVRLossTermConfig:
    name: str
    kind: str = "bce"
    weight: float = 1.0
    focal_alpha: float = 0.1
    focal_gamma: float = 2.0
    temperature: float = 1.0

    def __post_init__(self) -> None:
        name = str(self.name).strip()
        if not name:
            raise ValueError("loss term name must be non-empty")

        kind = str(self.kind).strip().lower()
        if kind not in PCVR_LOSS_TERM_KIND_CHOICES:
            raise ValueError(f"unsupported PCVR loss term kind: {self.kind}")

        weight = float(self.weight)
        if not math.isfinite(weight) or weight < 0.0:
            raise ValueError(f"loss term weight must be finite and >= 0, got {self.weight}")

        focal_alpha = float(self.focal_alpha)
        if not 0.0 <= focal_alpha <= 1.0:
            raise ValueError(f"focal_alpha must be between 0 and 1, got {self.focal_alpha}")

        focal_gamma = float(self.focal_gamma)
        if focal_gamma < 0.0:
            raise ValueError(f"focal_gamma must be >= 0, got {self.focal_gamma}")

        temperature = float(self.temperature)
        if temperature <= 0.0:
            raise ValueError(f"loss term temperature must be > 0, got {self.temperature}")

        object.__setattr__(self, "name", name)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "weight", weight)
        object.__setattr__(self, "focal_alpha", focal_alpha)
        object.__setattr__(self, "focal_gamma", focal_gamma)
        object.__setattr__(self, "temperature", temperature)

    @classmethod
    def from_value(cls, value: PCVRLossTermConfig | Mapping[str, Any]) -> PCVRLossTermConfig:
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise TypeError(f"loss term must be a mapping, got {type(value).__name__}")
        return cls(
            name=str(value["name"]),
            kind=str(value.get("kind", "bce")),
            weight=float(value.get("weight", 1.0)),
            focal_alpha=float(value.get("focal_alpha", 0.1)),
            focal_gamma=float(value.get("focal_gamma", 2.0)),
            temperature=float(value.get("temperature", value.get("pairwise_auc_temperature", 1.0))),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "weight": self.weight,
            "focal_alpha": self.focal_alpha,
            "focal_gamma": self.focal_gamma,
            "temperature": self.temperature,
        }


def _default_pcvr_loss_terms() -> tuple[PCVRLossTermConfig, ...]:
    return (PCVRLossTermConfig(name="bce", kind="bce", weight=1.0),)


@dataclass(frozen=True, slots=True)
class PCVRLossConfig:
    terms: tuple[PCVRLossTermConfig, ...] = _default_pcvr_loss_terms()

    def __post_init__(self) -> None:
        terms = tuple(PCVRLossTermConfig.from_value(term) for term in self.terms)
        if not terms:
            raise ValueError("PCVR loss config must define at least one loss term")
        names = [term.name for term in terms]
        duplicate_names = sorted({name for name in names if names.count(name) > 1})
        if duplicate_names:
            joined = ", ".join(duplicate_names)
            raise ValueError(f"PCVR loss term names must be unique; duplicates: {joined}")
        object.__setattr__(self, "terms", terms)

    @classmethod
    def from_value(cls, value: PCVRLossConfig | Mapping[str, Any] | Sequence[Any] | str | None) -> PCVRLossConfig:
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            value = _parse_loss_config_string(value)
        if isinstance(value, Mapping):
            value = value.get("terms", value.get("loss_terms"))
            if value is None:
                raise KeyError("PCVR loss config mapping must include 'terms' or 'loss_terms'")
        if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray, str)):
            return cls(terms=tuple(PCVRLossTermConfig.from_value(term) for term in value))
        raise TypeError(f"unsupported PCVR loss config value: {type(value).__name__}")

    def to_list(self) -> list[dict[str, Any]]:
        return [term.to_dict() for term in self.terms]

    def with_weight_overrides(self, raw_overrides: str | Mapping[str, float] | None) -> PCVRLossConfig:
        if raw_overrides in (None, ""):
            return self
        overrides = _parse_loss_weight_overrides(raw_overrides)
        known_names = {term.name for term in self.terms}
        unknown_names = sorted(set(overrides) - known_names)
        if unknown_names:
            joined = ", ".join(unknown_names)
            raise KeyError(f"loss weight override references unknown term(s): {joined}")
        return PCVRLossConfig(
            terms=tuple(
                PCVRLossTermConfig(
                    name=term.name,
                    kind=term.kind,
                    weight=overrides.get(term.name, term.weight),
                    focal_alpha=term.focal_alpha,
                    focal_gamma=term.focal_gamma,
                    temperature=term.temperature,
                )
                for term in self.terms
            )
        )

    def summary(self) -> str:
        return ", ".join(f"{term.name}:{term.kind}*{term.weight:g}" for term in self.terms)


DEFAULT_PCVR_LOSS_CONFIG = PCVRLossConfig()


def _parse_loss_config_string(value: str) -> Any:
    stripped = value.strip()
    if not stripped:
        raise ValueError("loss config string must be non-empty")
    if stripped[0] in "[{":
        return json.loads(stripped)
    terms: list[dict[str, Any]] = []
    for chunk in stripped.split(","):
        token = chunk.strip()
        if not token:
            continue
        parts = [part.strip() for part in token.split(":")]
        if len(parts) == 1:
            name = parts[0]
            kind = parts[0]
            weight = 1.0
        elif len(parts) == 2:
            name, weight_raw = parts
            kind = name
            weight = float(weight_raw)
        elif len(parts) == 3:
            name, kind, weight_raw = parts
            weight = float(weight_raw)
        else:
            raise ValueError(f"invalid loss term spec: {token!r}")
        terms.append({"name": name, "kind": kind, "weight": weight})
    return terms


def _parse_loss_weight_overrides(raw_overrides: str | Mapping[str, float]) -> dict[str, float]:
    if isinstance(raw_overrides, Mapping):
        pairs = raw_overrides.items()
    else:
        pairs = []
        for chunk in str(raw_overrides).split(","):
            token = chunk.strip()
            if not token:
                continue
            if "=" not in token:
                raise ValueError(f"loss weight override must use name=weight syntax: {token!r}")
            name, raw_weight = token.split("=", 1)
            pairs.append((name.strip(), raw_weight.strip()))

    overrides: dict[str, float] = {}
    for name, raw_weight in pairs:
        normalized_name = str(name).strip()
        if not normalized_name:
            raise ValueError("loss weight override name must be non-empty")
        weight = float(raw_weight)
        if not math.isfinite(weight) or weight < 0.0:
            raise ValueError(f"loss weight override must be finite and >= 0, got {raw_weight}")
        overrides[normalized_name] = weight
    return overrides


def parse_pcvr_loss_config_arg(value: str) -> list[dict[str, Any]]:
    return PCVRLossConfig.from_value(value).to_list()


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
        logger.warning("Failed to compile {}; falling back to eager execution: {}", label, error)
        return callable_obj


def create_logger(filepath: str | Path):
    """Configure shared loguru sinks for a training or evaluation process."""

    configure_logging(filepath)
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
        patience_unit: Literal["evaluations", "steps"] = "evaluations",
        step_scale: int = 1,
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
        if patience_unit not in {"evaluations", "steps"}:
            raise ValueError(f"unsupported patience_unit: {patience_unit}")
        if step_scale <= 0:
            raise ValueError("step_scale must be positive")
        self.patience_unit = patience_unit
        self.step_scale = int(step_scale)
        self.best_step: int | None = None

    @property
    def resolved_patience(self) -> int:
        if self.patience_unit == "steps":
            return self.patience * self.step_scale
        return self.patience

    def configure_step_scale(self, step_scale: int) -> None:
        if step_scale <= 0:
            raise ValueError("step_scale must be positive")
        self.step_scale = int(step_scale)

    def _resolve_current_step(self, step: int | None) -> int:
        if self.patience_unit != "steps":
            return 0
        if step is None:
            raise ValueError("step is required when patience_unit='steps'")
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
            self.best_step = current_step if self.patience_unit == "steps" else None
            self.save_checkpoint(score, model)
            self.best_model = copy.deepcopy(model.state_dict())
        elif self._is_not_improved(score):
            if self.patience_unit == "steps":
                assert self.best_step is not None
                self.counter = max(0, current_step - self.best_step)
            else:
                self.counter += 1
            logger.info(
                "{}earlyStopping counter: {} / {} {}",
                self.label,
                self.counter,
                self.resolved_patience,
                self.patience_unit,
            )
            if self.counter >= self.resolved_patience:
                self.early_stop = True
        else:
            logger.info("{}earlyStopping counter reset!", self.label)
            self.best_score = score
            self.best_model = copy.deepcopy(model.state_dict())
            self.best_extra_metrics = extra_metrics
            self.best_step = current_step if self.patience_unit == "steps" else None
            self.save_checkpoint(score, model)
            self.counter = 0

    def save_checkpoint(self, score: float, model: nn.Module) -> None:
        if self.verbose:
            logger.info("Validation score increased. Saving model ...")
        checkpoint_path = Path(self.checkpoint_path)
        save_checkpoint_state_dict(model.state_dict(), checkpoint_path)
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
