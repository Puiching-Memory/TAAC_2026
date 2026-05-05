from __future__ import annotations

from pathlib import Path


def locate_repo_root(anchor: Path) -> Path:
    resolved = anchor.resolve()
    candidates = (resolved, *resolved.parents) if resolved.is_dir() else resolved.parents
    for parent in candidates:
        if (parent / "pyproject.toml").is_file() and (parent / "experiments").is_dir():
            return parent
    raise RuntimeError(f"could not locate TAAC_2026 repository root from {anchor}")
