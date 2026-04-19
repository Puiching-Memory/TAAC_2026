from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import torch

from ...domain.config import TrainConfig
from ...infrastructure.io.console import logger


AMP_DTYPE_LOOKUP: dict[str, torch.dtype] = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


@dataclass(slots=True)
class OptimizerStepController:
    scaler: Any | None = None

    @property
    def uses_grad_scaler(self) -> bool:
        return self.scaler is not None

    def backward_and_step(
        self,
        loss: torch.Tensor,
        optimizer: Any,
        *,
        model_parameters,
        grad_clip_norm: float | None = None,
    ) -> None:
        if self.scaler is None:
            loss.backward()
            if grad_clip_norm and grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model_parameters, grad_clip_norm)
            optimizer.step()
            return

        self.scaler.scale(loss).backward()
        if grad_clip_norm and grad_clip_norm > 0:
            self.scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model_parameters, grad_clip_norm)
        self.scaler.step(optimizer)
        self.scaler.update()


@dataclass(slots=True)
class RuntimeExecution:
    base_model: torch.nn.Module
    execution_model: torch.nn.Module
    device: torch.device
    compile_requested: bool
    compile_active: bool
    compile_backend: str | None
    compile_mode: str | None
    compile_reason: str | None
    amp_requested: bool
    amp_active: bool
    amp_requested_dtype: str | None
    amp_resolved_dtype: str | None
    amp_reason: str | None
    optimizer_step_controller: OptimizerStepController

    def autocast_context(self):
        if not self.amp_active or self.amp_resolved_dtype is None:
            return nullcontext()
        return torch.autocast(
            device_type=self.device.type,
            dtype=AMP_DTYPE_LOOKUP[self.amp_resolved_dtype],
        )

    def summary(self) -> dict[str, Any]:
        return {
            "device": str(self.device),
            "torch_compile": {
                "requested": self.compile_requested,
                "active": self.compile_active,
                "backend": self.compile_backend,
                "mode": self.compile_mode,
                "reason": self.compile_reason,
            },
            "amp": {
                "requested": self.amp_requested,
                "active": self.amp_active,
                "requested_dtype": self.amp_requested_dtype,
                "resolved_dtype": self.amp_resolved_dtype,
                "gradient_scaler": self.optimizer_step_controller.uses_grad_scaler,
                "reason": self.amp_reason,
            },
        }

    @property
    def uses_grad_scaler(self) -> bool:
        return self.optimizer_step_controller.uses_grad_scaler

    def backward_and_step(
        self,
        loss: torch.Tensor,
        optimizer: Any,
        *,
        model_parameters,
        grad_clip_norm: float | None = None,
    ) -> None:
        self.optimizer_step_controller.backward_and_step(
            loss,
            optimizer,
            model_parameters=model_parameters,
            grad_clip_norm=grad_clip_norm,
        )


def _normalize_amp_dtype_name(name: str) -> str:
    normalized = str(name).strip().lower()
    if normalized not in AMP_DTYPE_LOOKUP:
        supported = ", ".join(sorted(AMP_DTYPE_LOOKUP))
        raise ValueError(f"Unsupported AMP dtype '{name}'. Expected one of: {supported}")
    return normalized


def _build_cuda_grad_scaler() -> Any:
    amp_namespace = getattr(torch, "amp", None)
    if amp_namespace is not None and hasattr(amp_namespace, "GradScaler"):
        return amp_namespace.GradScaler("cuda", enabled=True)
    return torch.cuda.amp.GradScaler(enabled=True)


def prepare_runtime_execution(
    model: torch.nn.Module,
    train_config: TrainConfig,
    device: torch.device | str,
) -> RuntimeExecution:
    resolved_device = torch.device(device)
    execution_model = model

    compile_requested = bool(train_config.enable_torch_compile)
    compile_active = False
    compile_reason: str | None = None
    if compile_requested:
        compile_fn = getattr(torch, "compile", None)
        if compile_fn is None:
            compile_reason = "torch.compile is unavailable in the current torch build"
            logger.warning("torch.compile requested on device={} but unavailable; continuing without compilation", resolved_device)
        else:
            compile_kwargs: dict[str, Any] = {}
            if train_config.torch_compile_backend is not None:
                compile_kwargs["backend"] = train_config.torch_compile_backend
            if train_config.torch_compile_mode is not None:
                compile_kwargs["mode"] = train_config.torch_compile_mode
            try:
                execution_model = compile_fn(model, **compile_kwargs)
                compile_active = True
            except Exception as exc:  # pragma: no cover - exercised via monkeypatch in tests
                raise RuntimeError(
                    "torch.compile failed "
                    f"(backend={train_config.torch_compile_backend!r}, mode={train_config.torch_compile_mode!r})"
                ) from exc

    amp_requested = bool(train_config.enable_amp)
    amp_requested_dtype = _normalize_amp_dtype_name(train_config.amp_dtype) if amp_requested else None
    amp_active = False
    amp_resolved_dtype: str | None = None
    amp_reason: str | None = None
    if amp_requested and amp_requested_dtype is not None:
        if resolved_device.type == "cuda":
            amp_active = True
            amp_resolved_dtype = amp_requested_dtype
        elif resolved_device.type == "cpu" and amp_requested_dtype == "bfloat16":
            amp_active = True
            amp_resolved_dtype = amp_requested_dtype
        else:
            amp_reason = f"AMP dtype {amp_requested_dtype} is not supported on {resolved_device.type}; continuing without autocast"
            logger.warning("{}", amp_reason)

    gradient_scaler = None
    if amp_active and resolved_device.type == "cuda" and amp_resolved_dtype == "float16":
        gradient_scaler = _build_cuda_grad_scaler()

    return RuntimeExecution(
        base_model=model,
        execution_model=execution_model,
        device=resolved_device,
        compile_requested=compile_requested,
        compile_active=compile_active,
        compile_backend=train_config.torch_compile_backend,
        compile_mode=train_config.torch_compile_mode,
        compile_reason=compile_reason,
        amp_requested=amp_requested,
        amp_active=amp_active,
        amp_requested_dtype=amp_requested_dtype,
        amp_resolved_dtype=amp_resolved_dtype,
        amp_reason=amp_reason,
        optimizer_step_controller=OptimizerStepController(scaler=gradient_scaler),
    )


def runtime_optimization_cli_args(train_config: TrainConfig) -> list[str]:
    args: list[str] = []
    if train_config.enable_torch_compile:
        args.append("--compile")
        if train_config.torch_compile_backend is not None:
            args.extend(["--compile-backend", train_config.torch_compile_backend])
        if train_config.torch_compile_mode is not None:
            args.extend(["--compile-mode", train_config.torch_compile_mode])
    if train_config.enable_amp:
        args.extend(["--amp", "--amp-dtype", train_config.amp_dtype])
    return args


__all__ = [
    "AMP_DTYPE_LOOKUP",
    "RuntimeExecution",
    "prepare_runtime_execution",
    "runtime_optimization_cli_args",
]