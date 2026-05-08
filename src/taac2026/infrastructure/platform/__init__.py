"""Platform runtime helpers for local and online TAAC entrypoints."""

from __future__ import annotations

from taac2026.infrastructure.platform.env import (
    DOCKER_GPU_PLATFORM,
    LOCAL_UV_PLATFORM,
    ONLINE_INFERENCE_BUNDLE_PLATFORM,
    ONLINE_TRAINING_BUNDLE_PLATFORM,
    RuntimePlatform,
)


__all__ = [
    "DOCKER_GPU_PLATFORM",
    "LOCAL_UV_PLATFORM",
    "ONLINE_INFERENCE_BUNDLE_PLATFORM",
    "ONLINE_TRAINING_BUNDLE_PLATFORM",
    "RuntimePlatform",
]