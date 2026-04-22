from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

from ...domain.config import TrainConfig
from ...infrastructure.nn.quantization import normalize_quantization_mode, quantize_model_for_inference
from ..training.runtime_optimization import RuntimeExecution, prepare_runtime_execution


SUPPORTED_INFERENCE_EXPORT_MODES = ("none", "torch-export")


def normalize_inference_export_mode(mode: str | None) -> str:
    if mode is None:
        return "none"
    normalized = str(mode).strip().lower()
    if normalized not in SUPPORTED_INFERENCE_EXPORT_MODES:
        supported = ", ".join(SUPPORTED_INFERENCE_EXPORT_MODES)
        raise ValueError(f"Unsupported export mode '{mode}'. Expected one of: {supported}")
    return normalized


def export_model_for_inference(
    model: torch.nn.Module,
    example_batch: Any | None,
    *,
    mode: str | None,
    output_path: str | Path,
) -> dict[str, Any]:
    resolved_mode = normalize_inference_export_mode(mode)
    if resolved_mode == "none":
        return {
            "mode": resolved_mode,
            "active": False,
            "reason": None,
            "artifact_path": None,
        }

    if example_batch is None:
        raise ValueError("example_batch is required when export mode is active")

    resolved_output_path = Path(output_path)
    if resolved_output_path.suffix != ".pt2":
        resolved_output_path = resolved_output_path.with_suffix(".pt2")
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)

    exported_program = torch.export.export(model.eval(), (example_batch,))
    torch.export.save(exported_program, resolved_output_path)
    return {
        "mode": resolved_mode,
        "active": True,
        "reason": "captured an example-shape evaluation graph with torch.export",
        "artifact_path": str(resolved_output_path),
    }


def prepare_evaluation_inference(
    model: torch.nn.Module,
    train_config: TrainConfig,
    device: torch.device | str,
    *,
    quantization_mode: str | None = None,
) -> tuple[RuntimeExecution, dict[str, Any], TrainConfig]:
    resolved_quantization_mode = normalize_quantization_mode(quantization_mode)
    inference_train_config = replace(train_config)
    resolved_device = torch.device(device)
    inference_model = model.eval()

    compile_disabled = False
    amp_disabled = False
    if resolved_quantization_mode != "none":
        compile_disabled = bool(inference_train_config.enable_torch_compile)
        amp_disabled = bool(inference_train_config.enable_amp)
        inference_train_config.device = "cpu"
        inference_train_config.enable_torch_compile = False
        inference_train_config.torch_compile_backend = None
        inference_train_config.torch_compile_mode = None
        inference_train_config.enable_amp = False
        inference_model, quantization_summary = quantize_model_for_inference(inference_model, resolved_quantization_mode)
        resolved_device = torch.device("cpu")
    else:
        inference_model, quantization_summary = quantize_model_for_inference(inference_model, resolved_quantization_mode)

    quantization_summary["runtime_overrides"] = {
        "compile_disabled": compile_disabled,
        "amp_disabled": amp_disabled,
        "forced_device": str(resolved_device) if resolved_quantization_mode != "none" else None,
    }
    runtime_execution = prepare_runtime_execution(inference_model, inference_train_config, resolved_device)
    return runtime_execution, quantization_summary, inference_train_config


__all__ = [
    "SUPPORTED_INFERENCE_EXPORT_MODES",
    "export_model_for_inference",
    "normalize_inference_export_mode",
    "prepare_evaluation_inference",
]