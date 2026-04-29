"""Checkpoint discovery and platform naming helpers."""

from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Any

from taac2026.infrastructure.io.json_utils import write_path


_GLOBAL_STEP_PATTERN = re.compile(r"^global_step(?P<step>\d+)(?:[A-Za-z0-9_.=\-]*)$")


def validate_checkpoint_dir_name(name: str) -> None:
    if len(name) > 300:
        raise ValueError("checkpoint directory name exceeds the platform 300 character limit")
    if not _GLOBAL_STEP_PATTERN.match(name):
        raise ValueError(
            "checkpoint directory must start with global_step and only contain letters, "
            "numbers, underscores, hyphens, equals signs, and dots"
        )


def checkpoint_step(path: Path) -> int:
    match = _GLOBAL_STEP_PATTERN.match(path.parent.name if path.name == "model.pt" else path.name)
    if not match:
        return -1
    return int(match.group("step"))


def resolve_checkpoint_path(run_dir: Path, checkpoint_path: Path | None = None) -> Path:
    if checkpoint_path is not None:
        candidate = checkpoint_path.expanduser().resolve()
        if candidate.is_dir():
            candidate = candidate / "model.pt"
        if not candidate.exists():
            raise FileNotFoundError(f"checkpoint not found: {candidate}")
        return candidate

    resolved_run_dir = run_dir.expanduser().resolve()
    best_candidates = sorted(
        resolved_run_dir.glob("global_step*.best_model/model.pt"),
        key=checkpoint_step,
    )
    if best_candidates:
        return best_candidates[-1]

    all_candidates = sorted(resolved_run_dir.glob("global_step*/model.pt"), key=checkpoint_step)
    if all_candidates:
        return all_candidates[-1]

    direct_candidate = resolved_run_dir / "model.pt"
    if direct_candidate.exists():
        return direct_candidate

    raise FileNotFoundError(f"no model.pt checkpoint found under {resolved_run_dir}")


def build_checkpoint_dir_name(global_step: int, checkpoint_params: dict[str, Any] | None = None, *, is_best: bool = False) -> str:
    if global_step < 0:
        raise ValueError("global_step must be non-negative")
    params = checkpoint_params or {}
    parts = [f"global_step{global_step}"]
    for key in ("layer", "head", "hidden"):
        if key in params:
            parts.append(f"{key}={params[key]}")
    name = ".".join(parts)
    if is_best:
        name += ".best_model"
    validate_checkpoint_dir_name(name)
    return name


def write_checkpoint_sidecars(
    checkpoint_dir: Path,
    *,
    schema_path: Path | None = None,
    ns_groups_path: Path | None = None,
    train_config: dict[str, Any] | None = None,
) -> dict[str, Path]:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    if schema_path is not None and schema_path.exists():
        target = checkpoint_dir / "schema.json"
        shutil.copy2(schema_path, target)
        written["schema"] = target

    ns_groups_copied = False
    if ns_groups_path is not None and ns_groups_path.exists():
        target = checkpoint_dir / "ns_groups.json"
        shutil.copy2(ns_groups_path, target)
        written["ns_groups"] = target
        ns_groups_copied = True

    if train_config is not None:
        config_to_dump = dict(train_config)
        if ns_groups_copied:
            config_to_dump["ns_groups_json"] = "ns_groups.json"
        target = checkpoint_dir / "train_config.json"
        write_path(target, config_to_dump, indent=2, trailing_newline=True)
        written["train_config"] = target

    return written
