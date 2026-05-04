"""Versioned bundle manifest helpers."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from taac2026 import __version__


BundleKind = Literal["training", "inference"]

BUNDLE_MANIFEST_VERSION = 1
TRAINING_BUNDLE_FORMAT = "taac2026-training-v2"
TRAINING_BUNDLE_FORMAT_VERSION = 2
INFERENCE_BUNDLE_FORMAT = "taac2026-inference-v1"
INFERENCE_BUNDLE_FORMAT_VERSION = 1

_ENTRYPOINTS: dict[BundleKind, str] = {
    "training": "run.sh",
    "inference": "infer.py",
}
_BUNDLE_FORMATS: dict[BundleKind, tuple[str, int]] = {
    "training": (TRAINING_BUNDLE_FORMAT, TRAINING_BUNDLE_FORMAT_VERSION),
    "inference": (INFERENCE_BUNDLE_FORMAT, INFERENCE_BUNDLE_FORMAT_VERSION),
}
_RUNTIME_ENV: dict[BundleKind, dict[str, str]] = {
    "training": {
        "dataset_path": "TRAIN_DATA_PATH",
        "schema_path": "TAAC_SCHEMA_PATH",
        "checkpoint_path": "TRAIN_CKPT_PATH",
        "cuda_profile": "TAAC_CUDA_PROFILE",
        "pip_extras": "TAAC_BUNDLE_PIP_EXTRAS (optional; defaults to runtime-only install with no dev extra)",
    },
    "inference": {
        "model_path": "MODEL_OUTPUT_PATH",
        "dataset_path": "EVAL_DATA_PATH",
        "result_path": "EVAL_RESULT_PATH",
        "schema_path": "TAAC_SCHEMA_PATH",
        "pip_extras": "TAAC_BUNDLE_PIP_EXTRAS (optional; defaults to runtime-only install with no dev extra)",
    },
}


class FrameworkMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = "taac2026"
    version: str


class BundleCompatibility(BaseModel):
    model_config = ConfigDict(extra="forbid")

    python: str = ">=3.10,<3.14"
    local_runner: str = "uv"
    online_runner: str = "python"
    requires_uv_online: bool = False
    package_root: str = "project"
    supports_taac_experiment_override: bool = True


class BundleManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    manifest_version: int = BUNDLE_MANIFEST_VERSION
    kind: BundleKind = Field(alias="bundle_kind")
    bundle_format: str
    bundle_format_version: int
    framework: FrameworkMetadata
    bundled_experiment_path: str
    entrypoint: str
    code_package: str
    runtime_env: dict[str, str]
    compatibility: BundleCompatibility

    def to_dict(self) -> dict[str, object]:
        return self.model_dump(mode="python", by_alias=True)


def build_bundle_manifest(*, kind: BundleKind, experiment_path: Path, root: Path) -> dict[str, object]:
    bundle_format, bundle_format_version = _BUNDLE_FORMATS[kind]
    manifest = BundleManifest(
        kind=kind,
        bundle_format=bundle_format,
        bundle_format_version=bundle_format_version,
        framework=FrameworkMetadata(version=__version__),
        bundled_experiment_path=str(experiment_path.relative_to(root)),
        entrypoint=_ENTRYPOINTS[kind],
        code_package="code_package.zip",
        runtime_env=dict(_RUNTIME_ENV[kind]),
        compatibility=BundleCompatibility(),
    )
    return validate_bundle_manifest(manifest.to_dict(), kind=kind)


def validate_bundle_manifest(manifest: Mapping[str, Any], *, kind: BundleKind | None = None) -> dict[str, object]:
    model = BundleManifest.model_validate(dict(manifest))
    payload = model.to_dict()
    if model.manifest_version != BUNDLE_MANIFEST_VERSION:
        raise ValueError(f"unsupported bundle manifest version: {model.manifest_version}")

    manifest_kind = model.kind
    if kind is not None and manifest_kind != kind:
        raise ValueError(f"bundle manifest kind mismatch: expected {kind}, got {manifest_kind}")

    expected_format, expected_format_version = _BUNDLE_FORMATS[manifest_kind]
    if model.bundle_format != expected_format:
        raise ValueError(f"unsupported {manifest_kind} bundle format: {model.bundle_format}")
    if model.bundle_format_version != expected_format_version:
        raise ValueError(f"unsupported {manifest_kind} bundle format version: {model.bundle_format_version}")

    if not model.bundled_experiment_path.startswith("experiments/"):
        raise ValueError("bundle manifest must contain an experiments/... bundled_experiment_path")
    if model.entrypoint != _ENTRYPOINTS[manifest_kind]:
        raise ValueError(f"invalid {manifest_kind} bundle entrypoint: {model.entrypoint}")
    if model.code_package != "code_package.zip":
        raise ValueError(f"invalid bundle code_package: {model.code_package}")

    missing_env = sorted(set(_RUNTIME_ENV[manifest_kind]) - set(model.runtime_env))
    if missing_env:
        raise KeyError(f"bundle manifest runtime_env is missing key(s): {', '.join(missing_env)}")

    return payload


__all__ = [
    "BUNDLE_MANIFEST_VERSION",
    "INFERENCE_BUNDLE_FORMAT",
    "INFERENCE_BUNDLE_FORMAT_VERSION",
    "TRAINING_BUNDLE_FORMAT",
    "TRAINING_BUNDLE_FORMAT_VERSION",
    "BundleCompatibility",
    "BundleKind",
    "BundleManifest",
    "FrameworkMetadata",
    "build_bundle_manifest",
    "validate_bundle_manifest",
]
