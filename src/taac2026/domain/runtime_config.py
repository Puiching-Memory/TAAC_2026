"""Runtime-facing PCVR configuration boundary objects."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
import math
from typing import Any


AMP_DTYPE_CHOICES: tuple[str, ...] = ("bfloat16", "float16")
PCVR_LOSS_TERM_KIND_CHOICES: tuple[str, ...] = ("bce", "focal", "pairwise_auc", "model")
DENSE_OPTIMIZER_TYPE_CHOICES: tuple[str, ...] = (
    "adamw",
    "fused_adamw",
    "orthogonal_adamw",
    "muon",
)
DEFAULT_PROGRESS_LOG_INTERVAL_STEPS = 100
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


@dataclass(frozen=True, slots=True)
class RuntimeExecutionConfig:
    amp: bool = False
    amp_dtype: str = "bfloat16"
    compile: bool = False
    progress_log_interval_steps: int = DEFAULT_PROGRESS_LOG_INTERVAL_STEPS
    deterministic: bool = True

    def __post_init__(self) -> None:
        interval = int(self.progress_log_interval_steps)
        if interval <= 0:
            raise ValueError("progress_log_interval_steps must be positive")
        object.__setattr__(self, "progress_log_interval_steps", interval)
        object.__setattr__(self, "amp_dtype", normalize_amp_dtype(self.amp_dtype))
        object.__setattr__(self, "deterministic", bool(self.deterministic))

    def normalized_amp_dtype(self) -> str:
        return normalize_amp_dtype(self.amp_dtype)


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


__all__ = [
    "AMP_DTYPE_CHOICES",
    "DEFAULT_PCVR_LOSS_CONFIG",
    "DEFAULT_PROGRESS_LOG_INTERVAL_STEPS",
    "DENSE_OPTIMIZER_TYPE_CHOICES",
    "PCVR_LOSS_TERM_KIND_CHOICES",
    "PCVRLossConfig",
    "PCVRLossTermConfig",
    "RuntimeExecutionConfig",
    "normalize_amp_dtype",
    "parse_pcvr_loss_config_arg",
]
