"""Named runtime platform adapters used by TAAC entrypoints."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


RunnerMode = Literal["python", "uv"]


@dataclass(frozen=True, slots=True)
class RuntimePlatform:
    name: str
    default_runner: RunnerMode
    pip_extras_env: str
    install_project_deps_by_default: bool
    bundle_kind: Literal["training", "inference"] | None = None


LOCAL_UV_PLATFORM = RuntimePlatform(
    name="local-uv",
    default_runner="uv",
    pip_extras_env="TAAC_PIP_EXTRAS",
    install_project_deps_by_default=False,
)
DOCKER_GPU_PLATFORM = RuntimePlatform(
    name="docker-gpu",
    default_runner="uv",
    pip_extras_env="TAAC_PIP_EXTRAS",
    install_project_deps_by_default=False,
)
ONLINE_TRAINING_BUNDLE_PLATFORM = RuntimePlatform(
    name="online-training-bundle",
    default_runner="python",
    pip_extras_env="TAAC_BUNDLE_PIP_EXTRAS",
    install_project_deps_by_default=True,
    bundle_kind="training",
)
ONLINE_INFERENCE_BUNDLE_PLATFORM = RuntimePlatform(
    name="online-inference-bundle",
    default_runner="python",
    pip_extras_env="TAAC_BUNDLE_PIP_EXTRAS",
    install_project_deps_by_default=True,
    bundle_kind="inference",
)


def select_run_sh_platform(*, bundle_mode: bool, platform_name: str | None = None) -> RuntimePlatform:
    if platform_name == "docker-gpu":
        return DOCKER_GPU_PLATFORM
    if bundle_mode:
        return ONLINE_TRAINING_BUNDLE_PLATFORM
    return LOCAL_UV_PLATFORM


__all__ = [
    "DOCKER_GPU_PLATFORM",
    "LOCAL_UV_PLATFORM",
    "ONLINE_INFERENCE_BUNDLE_PLATFORM",
    "ONLINE_TRAINING_BUNDLE_PLATFORM",
    "RunnerMode",
    "RuntimePlatform",
    "select_run_sh_platform",
]