from __future__ import annotations

from typing import Literal

import torch


GpuArchitecture = Literal["unknown", "ampere", "ada", "hopper", "blackwell"]
RecommendedPrecision = Literal["fp16", "bf16", "fp8", "mxfp8", "nvfp4"]

_PRECISION_PRIORITY: tuple[RecommendedPrecision, ...] = ("nvfp4", "mxfp8", "fp8", "bf16", "fp16")
_AUTO_RECIPE_BY_PRECISION: dict[RecommendedPrecision, str | None] = {
    "fp16": None,
    "bf16": None,
    "fp8": "delayed_scaling",
    "mxfp8": "mxfp8_block_scaling",
    "nvfp4": "nvfp4_block_scaling",
}


def classify_gpu_architecture(capability: tuple[int, int] | None) -> GpuArchitecture:
    if capability is None:
        return "unknown"
    major, minor = capability
    if major >= 10:
        return "blackwell"
    if major >= 9:
        return "hopper"
    if major == 8 and minor >= 9:
        return "ada"
    if major >= 8:
        return "ampere"
    return "unknown"


def detect_compute_capability(device: torch.device | int | None = None) -> tuple[int, int] | None:
    if not torch.cuda.is_available():
        return None

    if device is None:
        return torch.cuda.get_device_capability()

    resolved_device = torch.device(device)
    if resolved_device.type != "cuda":
        return None
    return torch.cuda.get_device_capability(resolved_device)


def detect_precision_support(device: torch.device | int | None = None) -> dict[str, object]:
    capability = detect_compute_capability(device)
    if capability is None:
        return {
            "compute_capability": None,
            "architecture": "unknown",
            "fp16": False,
            "bf16": False,
            "fp8": False,
            "fp8_block_scaling": False,
            "mxfp8": False,
            "nvfp4": False,
            "supported_precisions": [],
            "recommended_precision": None,
            "recommended_recipe": None,
        }

    major, minor = capability
    support: dict[str, object] = {
        "compute_capability": [major, minor],
        "architecture": classify_gpu_architecture(capability),
        "fp16": True,
        "bf16": major >= 8,
        "fp8": (major > 8) or (major == 8 and minor >= 9),
        "fp8_block_scaling": major >= 10,
        "mxfp8": major >= 10,
        "nvfp4": major >= 10,
    }

    supported_precisions = [
        precision for precision in _PRECISION_PRIORITY if bool(support.get(precision, False))
    ]
    recommended_precision = supported_precisions[0] if supported_precisions else None

    support["supported_precisions"] = supported_precisions
    support["recommended_precision"] = recommended_precision
    support["recommended_recipe"] = (
        _AUTO_RECIPE_BY_PRECISION[recommended_precision] if recommended_precision is not None else None
    )
    return support


__all__ = [
    "GpuArchitecture",
    "RecommendedPrecision",
    "classify_gpu_architecture",
    "detect_compute_capability",
    "detect_precision_support",
]