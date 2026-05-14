"""Checkpoint discovery and platform naming helpers."""

from __future__ import annotations

import math
import re
import shutil
from pathlib import Path
from typing import Any

from safetensors.torch import load_file as load_safetensors_file, save_file as save_safetensors_file
import torch

from taac2026.domain.sidecar import build_pcvr_train_config_sidecar
from taac2026.infrastructure.io.json import write_path

_GLOBAL_STEP_PATTERN = re.compile(r"^global_step(?P<step>\d+)(?:[A-Za-z0-9_.=\-]*)$")
PRIMARY_CHECKPOINT_FILENAME = "model.safetensors"
_CHECKPOINT_SUFFIX = ".safetensors"


def _is_supported_checkpoint_file(path: Path) -> bool:
    return path.suffix.lower() == _CHECKPOINT_SUFFIX


def _find_checkpoint_file(checkpoint_dir: Path) -> Path | None:
    candidate = preferred_checkpoint_path(checkpoint_dir)
    return candidate if candidate.exists() else None


def preferred_checkpoint_path(checkpoint_dir: Path) -> Path:
    return checkpoint_dir / PRIMARY_CHECKPOINT_FILENAME


def _checkpoint_dir_from_path(checkpoint_path: Path) -> Path:
    if _is_supported_checkpoint_file(checkpoint_path):
        return checkpoint_path.parent
    return checkpoint_path


def _serialize_state_dict_for_safetensors(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    serialized: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"checkpoint state_dict entry must be a tensor: {key}")
        serialized[key] = value.detach().cpu().contiguous()
    return serialized


def save_checkpoint_state_dict(
    state_dict: dict[str, torch.Tensor],
    checkpoint_path: Path,
) -> Path:
    checkpoint_dir = _checkpoint_dir_from_path(checkpoint_path.expanduser())
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    primary_path = preferred_checkpoint_path(checkpoint_dir)
    save_safetensors_file(_serialize_state_dict_for_safetensors(state_dict), str(primary_path))

    return primary_path


def load_checkpoint_state_dict(
    checkpoint_path: Path,
    *,
    map_location: torch.device | str | None = None,
) -> dict[str, torch.Tensor]:
    resolved_checkpoint_path = checkpoint_path.expanduser().resolve()
    if not _is_supported_checkpoint_file(resolved_checkpoint_path):
        raise ValueError(f"unsupported checkpoint format: {resolved_checkpoint_path}")
    device = "cpu" if map_location is None else str(map_location)
    return load_safetensors_file(str(resolved_checkpoint_path), device=device)


def validate_checkpoint_dir_name(name: str) -> None:
    if len(name) > 300:
        raise ValueError("checkpoint directory name exceeds the platform 300 character limit")
    if not _GLOBAL_STEP_PATTERN.match(name):
        raise ValueError(
            "checkpoint directory must start with global_step and only contain letters, "
            "numbers, underscores, hyphens, equals signs, and dots"
        )


def checkpoint_step(path: Path) -> int:
    match = _GLOBAL_STEP_PATTERN.match(path.parent.name if _is_supported_checkpoint_file(path) else path.name)
    if not match:
        return -1
    return int(match.group("step"))


def _format_checkpoint_auc(value: Any) -> str | None:
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric_value):
        return None
    formatted = f"{numeric_value:.6f}".rstrip("0").rstrip(".")
    return formatted or "0"


def resolve_checkpoint_path(run_dir: Path, checkpoint_path: Path | None = None) -> Path:
    if checkpoint_path is not None:
        candidate = checkpoint_path.expanduser().resolve()
        if candidate.is_dir():
            resolved_candidate = _find_checkpoint_file(candidate)
            if resolved_candidate is None:
                raise FileNotFoundError(f"checkpoint not found in {candidate}; expected: {PRIMARY_CHECKPOINT_FILENAME}")
            return resolved_candidate
        if not _is_supported_checkpoint_file(candidate):
            raise ValueError(f"unsupported checkpoint format: {candidate}")
        if not candidate.exists():
            raise FileNotFoundError(f"checkpoint not found: {candidate}")
        return candidate

    resolved_run_dir = run_dir.expanduser().resolve()
    all_candidates = sorted(
        resolved_run_dir.glob(f"global_step*/{PRIMARY_CHECKPOINT_FILENAME}"),
        key=checkpoint_step,
    )
    if all_candidates:
        return all_candidates[-1]

    direct_candidate = _find_checkpoint_file(resolved_run_dir)
    if direct_candidate is not None:
        return direct_candidate

    raise FileNotFoundError(f"no {PRIMARY_CHECKPOINT_FILENAME} checkpoint found under {resolved_run_dir}")


def build_checkpoint_dir_name(
    global_step: int,
    checkpoint_params: dict[str, Any] | None = None,
) -> str:
    if global_step < 0:
        raise ValueError("global_step must be non-negative")
    params = checkpoint_params or {}
    parts = [f"global_step{global_step}"]
    formatted_auc = _format_checkpoint_auc(params.get("auc", params.get("AUC")))
    if formatted_auc is not None:
        parts.append(f"AUC={formatted_auc}")
    name = ".".join(parts)
    validate_checkpoint_dir_name(name)
    return name


def write_checkpoint_sidecars(
    checkpoint_dir: Path,
    *,
    schema_path: Path | None = None,
    train_config: dict[str, Any] | None = None,
) -> dict[str, Path]:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    if schema_path is not None and schema_path.exists():
        target = checkpoint_dir / "schema.json"
        shutil.copy2(schema_path, target)
        written["schema"] = target

    if train_config is not None:
        target = checkpoint_dir / "train_config.json"
        write_path(target, build_pcvr_train_config_sidecar(train_config), indent=2, trailing_newline=True)
        written["train_config"] = target

    return written
