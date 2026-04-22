from __future__ import annotations

import copy
from functools import lru_cache
from typing import Any
import warnings

import torch
from torch import nn


SUPPORTED_QUANTIZATION_MODES = ("none", "int8")


def normalize_quantization_mode(mode: str | None) -> str:
    if mode is None:
        return "none"
    normalized = str(mode).strip().lower()
    if normalized not in SUPPORTED_QUANTIZATION_MODES:
        supported = ", ".join(SUPPORTED_QUANTIZATION_MODES)
        raise ValueError(f"Unsupported quantization mode '{mode}'. Expected one of: {supported}")
    return normalized


def _resolve_model_device(model: nn.Module) -> torch.device:
    first_parameter = next(model.parameters(), None)
    if first_parameter is not None:
        return first_parameter.device
    first_buffer = next(model.buffers(), None)
    if first_buffer is not None:
        return first_buffer.device
    return torch.device("cpu")


def _count_linear_layers(model: nn.Module) -> int:
    return sum(1 for module in model.modules() if isinstance(module, nn.Linear))


def _count_torchao_dynamic_quantized_linear_layers(model: nn.Module) -> int:
    return sum(
        1
        for module in model.modules()
        if isinstance(module, nn.Linear)
        and type(module.weight).__name__ == "Int8Tensor"
        and "torchao.quantization" in type(module.weight).__module__
    )


def _count_torchrec_embedding_bag_collections(model: nn.Module) -> int:
    return sum(
        1
        for module in model.modules()
        if type(module).__name__ == "EmbeddingBagCollection" and "torchrec.modules.embedding_modules" in type(module).__module__
    )


@lru_cache(maxsize=1)
def _torchao_dynamic_quantization_components() -> tuple[Any, type[Any]]:
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"invalid escape sequence .*",
            category=SyntaxWarning,
            module=r"torchao\..*",
        )
        from torchao.quantization import Int8DynamicActivationInt8WeightConfig, quantize_

    return quantize_, Int8DynamicActivationInt8WeightConfig


def _quantize_linear_modules(model: nn.Module) -> nn.Module:
    quantize_, config_type = _torchao_dynamic_quantization_components()
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"Deprecation: .*Layout is deprecated and will be removed in a future release of torchao.*",
            category=UserWarning,
            module=r"torchao\..*",
        )
        quantize_(model, config_type(version=2))
    return model

def quantize_model_for_inference(model: nn.Module, mode: str | None) -> tuple[nn.Module, dict[str, Any]]:
    resolved_mode = normalize_quantization_mode(mode)
    if resolved_mode == "none":
        return model, {
            "mode": resolved_mode,
            "active": False,
            "reason": None,
            "device": str(_resolve_model_device(model)),
            "quantized_linear_layers": 0,
        }

    torchrec_embedding_bag_collection_count = _count_torchrec_embedding_bag_collections(model)
    if torchrec_embedding_bag_collection_count > 0:
        raise ValueError(
            "dynamic int8 inference does not support TorchRec EmbeddingBagCollection modules; "
            f"use quantization mode 'none' for models with {torchrec_embedding_bag_collection_count} collection(s)"
        )

    cpu_model = copy.deepcopy(model).cpu().eval()
    quantizable_linear_layers = _count_linear_layers(cpu_model)
    if quantizable_linear_layers == 0:
        return cpu_model, {
            "mode": resolved_mode,
            "active": False,
            "reason": "model has no nn.Linear modules eligible for dynamic int8 quantization",
            "device": "cpu",
            "quantized_linear_layers": 0,
        }

    quantized_model = _quantize_linear_modules(cpu_model)
    quantized_linear_layers = _count_torchao_dynamic_quantized_linear_layers(quantized_model)

    reason_parts: list[str] = []
    if quantized_linear_layers > 0:
        reason_parts.append("dynamic int8 inference quantized nn.Linear modules via torchao on cpu")
    else:
        reason_parts.append(
            "dynamic int8 inference targeted nn.Linear modules via torchao on cpu but no layers reported an Int8Tensor weight"
        )

    return quantized_model, {
        "mode": resolved_mode,
        "active": quantized_linear_layers > 0,
        "reason": "; ".join(reason_parts),
        "device": "cpu",
        "quantized_linear_layers": quantized_linear_layers,
    }


__all__ = [
    "SUPPORTED_QUANTIZATION_MODES",
    "normalize_quantization_mode",
    "quantize_model_for_inference",
]