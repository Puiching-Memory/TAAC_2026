"""Shared experiment metadata helpers for CLI entrypoints."""

from __future__ import annotations

import os


def experiment_requires_dataset(experiment: object) -> bool:
    metadata = getattr(experiment, "metadata", {})
    if not isinstance(metadata, dict):
        return True
    requires_dataset = metadata.get("requires_dataset", True)
    return bool(requires_dataset) if isinstance(requires_dataset, bool) else True


def experiment_kind(experiment: object) -> str | None:
    metadata = getattr(experiment, "metadata", {})
    if not isinstance(metadata, dict):
        return None
    kind = metadata.get("kind")
    return kind if isinstance(kind, str) else None


def is_bundle_mode() -> bool:
    return os.environ.get("TAAC_BUNDLE_MODE") == "1"
