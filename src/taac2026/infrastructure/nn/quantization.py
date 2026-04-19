from __future__ import annotations

from functools import lru_cache
from typing import Any

import torch
from torch import nn


SUPPORTED_QUANTIZATION_MODES = ("none", "int8")
_QUANTIZATION_ALIASES = {
    "off": "none",
    "false": "none",
    "dynamic-int8": "int8",
    "linear-int8": "int8",
}


def normalize_quantization_mode(mode: str | None) -> str:
    if mode is None:
        return "none"
    normalized = str(mode).strip().lower()
    normalized = _QUANTIZATION_ALIASES.get(normalized, normalized)
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


def _count_dynamic_quantized_linear_layers(model: nn.Module) -> int:
    return sum(
        1
        for module in model.modules()
        if type(module).__name__ == "Linear" and "quantized.dynamic" in type(module).__module__
    )


def _count_torchrec_embedding_bag_collections(model: nn.Module) -> int:
    return sum(
        1
        for module in model.modules()
        if type(module).__name__ == "EmbeddingBagCollection" and "torchrec.modules.embedding_modules" in type(module).__module__
    )


def _count_quantized_torchrec_embedding_bag_collections(model: nn.Module) -> int:
    return sum(
        1
        for module in model.modules()
        if type(module).__name__ == "EmbeddingBagCollection" and "torchrec.quant.embedding_modules" in type(module).__module__
    )


@lru_cache(maxsize=1)
def _torchrec_dynamic_quantization_types():
    from torchrec import EmbeddingBagCollection as TorchRecEmbeddingBagCollection
    from torchrec.quant import EmbeddingBagCollection as QuantizedEmbeddingBagCollection

    quantization = torch.ao.quantization
    qconfig = quantization.QConfig(
        activation=quantization.PlaceholderObserver,
        weight=quantization.PlaceholderObserver.with_args(dtype=torch.qint8),
    )
    return TorchRecEmbeddingBagCollection, QuantizedEmbeddingBagCollection, qconfig


def _quantize_dynamic_modules(model: nn.Module) -> nn.Module:
    quantization = torch.ao.quantization
    qconfig_spec: dict[type[nn.Module], Any] = {nn.Linear: quantization.default_dynamic_qconfig}
    mapping: dict[type[nn.Module], type[nn.Module]] = {}

    torchrec_embedding_bag_collection_count = _count_torchrec_embedding_bag_collections(model)
    if torchrec_embedding_bag_collection_count > 0:
        embedding_bag_collection_type, quantized_embedding_bag_collection_type, torchrec_qconfig = _torchrec_dynamic_quantization_types()
        qconfig_spec[embedding_bag_collection_type] = torchrec_qconfig
        mapping[embedding_bag_collection_type] = quantized_embedding_bag_collection_type

    quantize_kwargs: dict[str, Any] = {
        "qconfig_spec": qconfig_spec,
        "inplace": False,
    }
    if mapping:
        quantize_kwargs["mapping"] = mapping
    return quantization.quantize_dynamic(model, **quantize_kwargs)


def quantize_model_for_inference(model: nn.Module, mode: str | None) -> tuple[nn.Module, dict[str, Any]]:
    resolved_mode = normalize_quantization_mode(mode)
    quantizable_linear_layers = _count_linear_layers(model)
    quantizable_embedding_collections = _count_torchrec_embedding_bag_collections(model)
    if resolved_mode == "none":
        return model, {
            "requested_mode": resolved_mode,
            "mode": resolved_mode,
            "active": False,
            "reason": None,
            "device": str(_resolve_model_device(model)),
            "quantizable_linear_layers": quantizable_linear_layers,
            "quantizable_embedding_collections": quantizable_embedding_collections,
            "quantized_linear_layers": 0,
            "quantized_embedding_collections": 0,
        }

    cpu_model = model.cpu().eval()
    if quantizable_linear_layers == 0 and quantizable_embedding_collections == 0:
        return cpu_model, {
            "requested_mode": resolved_mode,
            "mode": resolved_mode,
            "active": False,
            "reason": "model has no nn.Linear or TorchRec EmbeddingBagCollection modules eligible for dynamic int8 quantization",
            "device": "cpu",
            "quantizable_linear_layers": quantizable_linear_layers,
            "quantizable_embedding_collections": quantizable_embedding_collections,
            "quantized_linear_layers": 0,
            "quantized_embedding_collections": 0,
        }

    quantized_model = _quantize_dynamic_modules(cpu_model)
    quantized_linear_layers = _count_dynamic_quantized_linear_layers(quantized_model)
    quantized_embedding_collections = _count_quantized_torchrec_embedding_bag_collections(quantized_model)
    quantized_targets = []
    if quantizable_linear_layers > 0:
        quantized_targets.append("nn.Linear")
    if quantizable_embedding_collections > 0:
        quantized_targets.append("TorchRec EmbeddingBagCollection")
    return quantized_model, {
        "requested_mode": resolved_mode,
        "mode": resolved_mode,
        "active": quantized_linear_layers > 0 or quantized_embedding_collections > 0,
        "reason": "dynamic int8 inference currently quantizes " + " and ".join(quantized_targets) + " modules on cpu",
        "device": "cpu",
        "quantizable_linear_layers": quantizable_linear_layers,
        "quantizable_embedding_collections": quantizable_embedding_collections,
        "quantized_linear_layers": quantized_linear_layers,
        "quantized_embedding_collections": quantized_embedding_collections,
    }


__all__ = [
    "SUPPORTED_QUANTIZATION_MODES",
    "normalize_quantization_mode",
    "quantize_model_for_inference",
]