from __future__ import annotations

from contextlib import ExitStack, nullcontext
from functools import lru_cache
import importlib
import importlib.util
from typing import Any, Literal

import torch
from torch import nn

from .gpu_capability import detect_precision_support
from .triton_attention import resolve_attention_mask


TransformerEnginePrecision = Literal["auto", "fp16", "bf16", "fp8", "mxfp8", "nvfp4"]
TransformerEngineRecipeMode = Literal[
    "auto",
    "none",
    "delayed_scaling",
    "current_scaling",
    "fp8_block_scaling",
    "mxfp8_block_scaling",
    "nvfp4_block_scaling",
]

_CONCRETE_PRECISIONS = ("fp16", "bf16", "fp8", "mxfp8", "nvfp4")
_AUTO_PRECISION_PRIORITY = ("nvfp4", "mxfp8", "fp8", "bf16", "fp16")
_RECIPE_PRECISION_REQUIREMENTS: dict[str, str] = {
    "delayed_scaling": "fp8",
    "current_scaling": "fp8",
    "fp8_block_scaling": "fp8",
    "mxfp8_block_scaling": "mxfp8",
    "nvfp4_block_scaling": "nvfp4",
}
_AUTO_RECIPE_FOR_PRECISION: dict[str, str] = {
    "fp8": "delayed_scaling",
    "mxfp8": "mxfp8_block_scaling",
    "nvfp4": "nvfp4_block_scaling",
}


def is_transformer_engine_installed() -> bool:
    return importlib.util.find_spec("transformer_engine") is not None


@lru_cache(maxsize=1)
def _load_transformer_engine_root():
    return importlib.import_module("transformer_engine")


@lru_cache(maxsize=1)
def _load_transformer_engine_pytorch():
    return importlib.import_module("transformer_engine.pytorch")


@lru_cache(maxsize=1)
def _load_transformer_engine_recipe():
    return importlib.import_module("transformer_engine.common.recipe")


def _normalize_precision(name: TransformerEnginePrecision | str) -> TransformerEnginePrecision:
    normalized = str(name).strip().lower()
    if normalized not in {"auto", *_CONCRETE_PRECISIONS}:
        raise ValueError(f"Unsupported Transformer Engine precision '{name}'")
    return normalized  # type: ignore[return-value]


def _normalize_recipe_mode(name: TransformerEngineRecipeMode | str) -> TransformerEngineRecipeMode:
    normalized = str(name).strip().lower()
    if normalized not in {"auto", "none", *tuple(_RECIPE_PRECISION_REQUIREMENTS)}:
        raise ValueError(f"Unsupported Transformer Engine recipe mode '{name}'")
    return normalized  # type: ignore[return-value]


def _normalize_activation_name(name: str) -> str:
    normalized = str(name).strip().lower()
    if normalized not in {"gelu", "silu", "swiglu"}:
        raise ValueError(f"Unsupported Transformer Engine activation '{name}'")
    return normalized


def _normalize_normalization_name(name: str) -> str:
    normalized = str(name).strip().lower()
    if normalized == "layernorm":
        return "LayerNorm"
    if normalized == "rmsnorm":
        return "RMSNorm"
    raise ValueError(f"Unsupported normalization '{name}'")


def _normalize_availability_result(result: object, *, fallback_reason: str = "") -> dict[str, object]:
    if isinstance(result, tuple) and len(result) == 2:
        available, reason = result
    else:
        available, reason = bool(result), ""
    resolved_reason = str(reason or fallback_reason)
    return {
        "available": bool(available),
        "reason": resolved_reason,
    }


def _call_availability_probe(te_module: Any, attribute_name: str, *, missing_reason: str) -> dict[str, object]:
    probe = getattr(te_module, attribute_name, None)
    if probe is None:
        return {
            "available": False,
            "reason": missing_reason,
        }
    return _normalize_availability_result(probe(return_reason=True))


def detect_transformer_engine_availability(device: torch.device | int | None = None) -> dict[str, object]:
    report: dict[str, object] = {
        "installed": is_transformer_engine_installed(),
    }
    if not report["installed"]:
        return report

    te_module = _load_transformer_engine_pytorch()
    te_root = _load_transformer_engine_root()
    report["version"] = getattr(te_root, "__version__", "unknown")

    resolved_device = torch.device(device) if device is not None else torch.device("cuda:0")
    if resolved_device.type != "cuda" or not torch.cuda.is_available():
        unavailable_reason = "CUDA device not available"
        report.update(
            {
                "compute_capability": None,
                "cudnn_version": None,
                "bf16": _normalize_availability_result(False, fallback_reason=unavailable_reason),
                "fp8": _normalize_availability_result(False, fallback_reason=unavailable_reason),
                "fp8_block_scaling": _normalize_availability_result(False, fallback_reason=unavailable_reason),
                "mxfp8": _normalize_availability_result(False, fallback_reason=unavailable_reason),
                "nvfp4": _normalize_availability_result(False, fallback_reason=unavailable_reason),
            }
        )
        return report

    report["compute_capability"] = list(torch.cuda.get_device_capability(resolved_device))
    get_cudnn_version = getattr(te_module, "get_cudnn_version", None)
    report["cudnn_version"] = list(get_cudnn_version()) if get_cudnn_version is not None else None
    report["bf16"] = _call_availability_probe(
        te_module,
        "is_bf16_available",
        missing_reason="Installed Transformer Engine does not expose BF16 availability probing",
    )
    report["fp8"] = _call_availability_probe(
        te_module,
        "is_fp8_available",
        missing_reason="Installed Transformer Engine does not expose FP8 availability probing",
    )
    report["fp8_block_scaling"] = _call_availability_probe(
        te_module,
        "is_fp8_block_scaling_available",
        missing_reason="Installed Transformer Engine does not expose FP8 block scaling availability probing",
    )
    report["mxfp8"] = _call_availability_probe(
        te_module,
        "is_mxfp8_available",
        missing_reason="Installed Transformer Engine does not expose MXFP8 availability probing",
    )
    report["nvfp4"] = _call_availability_probe(
        te_module,
        "is_nvfp4_available",
        missing_reason="Installed Transformer Engine does not expose NVFP4 availability probing",
    )
    return report


def _precision_support_reason(
    precision: str,
    *,
    device: torch.device,
    support: dict[str, object],
    te_report: dict[str, object] | None,
) -> tuple[bool, str]:
    if not bool(support.get(precision, False)):
        return False, f"Current GPU does not support Transformer Engine precision '{precision}'"
    if precision == "fp16" or te_report is None:
        return True, ""

    te_entry = te_report.get(precision)
    if not isinstance(te_entry, dict):
        return True, ""
    if bool(te_entry.get("available", False)):
        return True, ""
    return False, str(te_entry.get("reason") or f"Transformer Engine does not support precision '{precision}'")


def resolve_transformer_engine_precision(
    device: torch.device | str,
    *,
    requested_precision: TransformerEnginePrecision | str = "auto",
    requested_recipe_mode: TransformerEngineRecipeMode | str = "auto",
) -> str | None:
    resolved_device = torch.device(device)
    support = detect_precision_support(resolved_device)
    te_report = detect_transformer_engine_availability(resolved_device) if is_transformer_engine_installed() else None

    normalized_precision = _normalize_precision(requested_precision)
    normalized_recipe_mode = _normalize_recipe_mode(requested_recipe_mode)
    if normalized_recipe_mode not in {"auto", "none"}:
        required_precision = _RECIPE_PRECISION_REQUIREMENTS[normalized_recipe_mode]
        if normalized_precision not in {"auto", required_precision}:
            raise ValueError(
                f"Transformer Engine recipe mode '{normalized_recipe_mode}' requires precision '{required_precision}', "
                f"but got '{requested_precision}'"
            )
        normalized_precision = required_precision  # type: ignore[assignment]

    if normalized_precision == "auto":
        for candidate in _AUTO_PRECISION_PRIORITY:
            is_supported, _ = _precision_support_reason(
                candidate,
                device=resolved_device,
                support=support,
                te_report=te_report,
            )
            if is_supported:
                return candidate
        return None

    is_supported, reason = _precision_support_reason(
        normalized_precision,
        device=resolved_device,
        support=support,
        te_report=te_report,
    )
    if not is_supported:
        raise ValueError(reason)
    return normalized_precision


def resolve_transformer_engine_recipe_mode(
    device: torch.device | str,
    *,
    requested_precision: TransformerEnginePrecision | str = "auto",
    requested_recipe_mode: TransformerEngineRecipeMode | str = "auto",
) -> str:
    normalized_recipe_mode = _normalize_recipe_mode(requested_recipe_mode)
    resolved_precision = resolve_transformer_engine_precision(
        device,
        requested_precision=requested_precision,
        requested_recipe_mode=normalized_recipe_mode,
    )
    if resolved_precision is None:
        return "none"

    if normalized_recipe_mode == "auto":
        return _AUTO_RECIPE_FOR_PRECISION.get(resolved_precision, "none")
    if normalized_recipe_mode == "none":
        return normalized_recipe_mode

    required_precision = _RECIPE_PRECISION_REQUIREMENTS[normalized_recipe_mode]
    if required_precision != resolved_precision:
        raise ValueError(
            f"Transformer Engine recipe mode '{normalized_recipe_mode}' is incompatible with precision '{resolved_precision}'"
        )

    te_report = detect_transformer_engine_availability(torch.device(device)) if is_transformer_engine_installed() else None
    if normalized_recipe_mode == "fp8_block_scaling" and te_report is not None:
        entry = te_report.get("fp8_block_scaling")
        if isinstance(entry, dict) and not bool(entry.get("available", False)):
            raise ValueError(str(entry.get("reason") or "Transformer Engine FP8 block scaling is unavailable"))
    return normalized_recipe_mode


def build_transformer_engine_recipe(
    device: torch.device | str,
    *,
    requested_precision: TransformerEnginePrecision | str = "auto",
    requested_recipe_mode: TransformerEngineRecipeMode | str = "auto",
    amax_history_len: int = 16,
):
    resolved_recipe_mode = resolve_transformer_engine_recipe_mode(
        device,
        requested_precision=requested_precision,
        requested_recipe_mode=requested_recipe_mode,
    )
    if resolved_recipe_mode == "none":
        return None

    recipe_module = _load_transformer_engine_recipe()
    if resolved_recipe_mode == "delayed_scaling":
        return recipe_module.DelayedScaling(
            fp8_format=recipe_module.Format.HYBRID,
            amax_history_len=amax_history_len,
            amax_compute_algo="max",
        )
    if resolved_recipe_mode == "current_scaling":
        return recipe_module.Float8CurrentScaling(fp8_format=recipe_module.Format.HYBRID)
    if resolved_recipe_mode == "fp8_block_scaling":
        return recipe_module.Float8BlockScaling(fp8_format=recipe_module.Format.E4M3)
    if resolved_recipe_mode == "mxfp8_block_scaling":
        return recipe_module.MXFP8BlockScaling(fp8_format=recipe_module.Format.E4M3)
    if resolved_recipe_mode == "nvfp4_block_scaling":
        return recipe_module.NVFP4BlockScaling()
    raise ValueError(f"Unsupported Transformer Engine recipe mode '{resolved_recipe_mode}'")


def _resolve_autocast_dtype(device: torch.device, precision: str) -> torch.dtype:
    support = detect_precision_support(device)
    if precision == "fp16":
        return torch.float16
    if precision == "bf16":
        return torch.bfloat16
    return torch.bfloat16 if bool(support.get("bf16", False)) else torch.float16


def build_transformer_engine_context(
    device: torch.device | str,
    *,
    requested_precision: TransformerEnginePrecision | str = "auto",
    requested_recipe_mode: TransformerEngineRecipeMode | str = "auto",
    amax_history_len: int = 16,
):
    resolved_device = torch.device(device)
    if resolved_device.type != "cuda":
        return nullcontext()

    resolved_precision = resolve_transformer_engine_precision(
        resolved_device,
        requested_precision=requested_precision,
        requested_recipe_mode=requested_recipe_mode,
    )
    if resolved_precision is None:
        return nullcontext()

    stack = ExitStack()
    stack.enter_context(torch.autocast(device_type="cuda", dtype=_resolve_autocast_dtype(resolved_device, resolved_precision)))
    recipe = build_transformer_engine_recipe(
        resolved_device,
        requested_precision=resolved_precision,
        requested_recipe_mode=requested_recipe_mode,
        amax_history_len=amax_history_len,
    )
    if recipe is not None:
        te_module = _load_transformer_engine_pytorch()
        stack.enter_context(te_module.autocast(enabled=True, recipe=recipe))
    return stack


def adapt_mask_for_te(mask: torch.Tensor | None) -> torch.Tensor | None:
    if mask is None:
        return None
    return ~mask.to(dtype=torch.bool)


def _resolve_te_attention_mask(
    *,
    query_length: int,
    key_length: int,
    batch_size: int,
    device: torch.device,
    attention_mask: torch.Tensor | None,
    query_mask: torch.Tensor | None,
    key_mask: torch.Tensor | None,
    is_causal: bool,
) -> tuple[torch.Tensor | None, str]:
    # Let TE use its native no-mask or causal path when no explicit padding or bias mask is present.
    if attention_mask is None and query_mask is None and key_mask is None:
        return None, "causal" if is_causal else "no_mask"

    combined_mask = resolve_attention_mask(
        query_length=query_length,
        key_length=key_length,
        batch_size=batch_size,
        device=device,
        attention_mask=attention_mask,
        query_mask=query_mask,
        key_mask=key_mask,
        is_causal=is_causal,
    )
    te_mask = adapt_mask_for_te(combined_mask)
    if te_mask is None:
        return None, "causal" if is_causal else "no_mask"
    return te_mask.unsqueeze(1), "arbitrary"


def _build_module_execution_context(
    device: torch.device,
    *,
    requested_precision: TransformerEnginePrecision,
    requested_recipe_mode: TransformerEngineRecipeMode,
    amax_history_len: int,
    use_external_context: bool,
):
    if use_external_context:
        return nullcontext()
    return build_transformer_engine_context(
        device,
        requested_precision=requested_precision,
        requested_recipe_mode=requested_recipe_mode,
        amax_history_len=amax_history_len,
    )


class TransformerEngineAttention(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        *,
        dropout: float = 0.0,
        attention_type: str = "self",
        te_precision: TransformerEnginePrecision = "auto",
        te_recipe_mode: TransformerEngineRecipeMode = "auto",
        te_amax_history_len: int = 16,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        normalized_attention_type = str(attention_type).strip().lower()
        if normalized_attention_type not in {"self", "cross"}:
            raise ValueError(f"Unsupported Transformer Engine attention_type '{attention_type}'")

        te_module = _load_transformer_engine_pytorch()
        self.attention_type = normalized_attention_type
        self.te_precision = _normalize_precision(te_precision)
        self.te_recipe_mode = _normalize_recipe_mode(te_recipe_mode)
        self.te_amax_history_len = te_amax_history_len
        self.attention = te_module.MultiheadAttention(
            hidden_size=hidden_dim,
            num_attention_heads=num_heads,
            attention_dropout=dropout,
            attention_type=normalized_attention_type,
            attn_mask_type="arbitrary",
            input_layernorm=False,
            qkv_format="bshd",
            bias=False,
            device="cpu",
        )
        self.output_dropout = nn.Dropout(dropout)

    def forward(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
        query_mask: torch.Tensor | None = None,
        key_mask: torch.Tensor | None = None,
        additive_bias: torch.Tensor | None = None,
        is_causal: bool = False,
        use_external_context: bool = False,
    ) -> torch.Tensor:
        if query_states.device.type != "cuda":
            raise RuntimeError("Transformer Engine attention backend requires CUDA tensors")
        te_mask, attn_mask_type = _resolve_te_attention_mask(
            query_length=query_states.shape[1],
            key_length=key_states.shape[1],
            batch_size=query_states.shape[0],
            device=query_states.device,
            attention_mask=attention_mask,
            query_mask=query_mask,
            key_mask=key_mask,
            is_causal=is_causal,
        )

        core_attention_bias = None
        core_attention_bias_type = "no_bias"
        if additive_bias is not None:
            core_attention_bias = additive_bias.to(device=query_states.device, dtype=query_states.dtype)
            if core_attention_bias.ndim == 3:
                core_attention_bias = core_attention_bias.unsqueeze(1)
            elif core_attention_bias.ndim != 4:
                raise ValueError("Transformer Engine additive_bias must have shape [batch, query, key] or [batch, heads, query, key]")
            core_attention_bias_type = "post_scale_bias"

        with _build_module_execution_context(
            query_states.device,
            requested_precision=self.te_precision,
            requested_recipe_mode=self.te_recipe_mode,
            amax_history_len=self.te_amax_history_len,
            use_external_context=use_external_context,
        ):
            if self.attention_type == "self":
                if key_states is not query_states or value_states is not query_states:
                    raise ValueError("Transformer Engine self attention expects query_states, key_states, and value_states to share the same tensor")
                output = self.attention(
                    query_states,
                    attention_mask=te_mask,
                    attn_mask_type=attn_mask_type,
                    core_attention_bias_type=core_attention_bias_type,
                    core_attention_bias=core_attention_bias,
                )
            else:
                if value_states is not key_states:
                    raise ValueError("Transformer Engine cross attention expects key_states and value_states to share the same tensor")
                output = self.attention(
                    query_states,
                    attention_mask=te_mask,
                    attn_mask_type=attn_mask_type,
                    encoder_output=key_states,
                    core_attention_bias_type=core_attention_bias_type,
                    core_attention_bias=core_attention_bias,
                )
        return self.output_dropout(output)


class TransformerEngineFeedForward(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        ffn_dim: int,
        *,
        norm_type: str,
        activation: str = "swiglu",
        dropout: float = 0.0,
        te_precision: TransformerEnginePrecision = "auto",
        te_recipe_mode: TransformerEngineRecipeMode = "auto",
        te_amax_history_len: int = 16,
    ) -> None:
        super().__init__()
        te_module = _load_transformer_engine_pytorch()
        self.te_precision = _normalize_precision(te_precision)
        self.te_recipe_mode = _normalize_recipe_mode(te_recipe_mode)
        self.te_amax_history_len = te_amax_history_len
        self.feed_forward = te_module.LayerNormMLP(
            hidden_size=hidden_dim,
            ffn_hidden_size=ffn_dim,
            normalization=_normalize_normalization_name(norm_type),
            activation=_normalize_activation_name(activation),
            bias=True,
            device="cpu",
        )
        self.output_dropout = nn.Dropout(dropout)

    def forward(self, hidden_states: torch.Tensor, *, use_external_context: bool = False) -> torch.Tensor:
        if hidden_states.device.type != "cuda":
            raise RuntimeError("Transformer Engine feed-forward backend requires CUDA tensors")
        with _build_module_execution_context(
            hidden_states.device,
            requested_precision=self.te_precision,
            requested_recipe_mode=self.te_recipe_mode,
            amax_history_len=self.te_amax_history_len,
            use_external_context=use_external_context,
        ):
            output = self.feed_forward(hidden_states)
        return self.output_dropout(output)


__all__ = [
    "TransformerEngineAttention",
    "TransformerEngineFeedForward",
    "TransformerEnginePrecision",
    "TransformerEngineRecipeMode",
    "adapt_mask_for_te",
    "build_transformer_engine_context",
    "build_transformer_engine_recipe",
    "detect_transformer_engine_availability",
    "is_transformer_engine_installed",
    "resolve_transformer_engine_precision",
    "resolve_transformer_engine_recipe_mode",
]