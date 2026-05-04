"""Shared experiment package discovery helpers."""

from __future__ import annotations

from pathlib import Path


def discover_experiment_paths(
    experiment_root: Path,
    *,
    required_files: tuple[str, ...] = ("__init__.py", "model.py"),
) -> list[str]:
    experiment_paths: list[str] = []
    root = experiment_root.expanduser().resolve()
    base_root = root.parent.parent if root.parent.name == "experiments" else root.parent
    for child in sorted(root.iterdir()):
        if not child.is_dir() or child.name.startswith("__"):
            continue
        if all((child / name).exists() for name in required_files):
            experiment_paths.append(child.relative_to(base_root).as_posix())
    return experiment_paths
