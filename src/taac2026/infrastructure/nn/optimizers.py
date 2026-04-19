from __future__ import annotations

from collections.abc import Iterator

import torch
from torch import nn
from torch.optim import Optimizer
from torchrec.optim import RowWiseAdagrad


class Muon(Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1.0e-3,
        momentum: float = 0.95,
        ns_steps: int = 5,
        weight_decay: float = 0.0,
    ) -> None:
        defaults = dict(lr=lr, momentum=momentum, ns_steps=ns_steps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @staticmethod
    def _newton_schulz(gradient: torch.Tensor, steps: int = 5) -> torch.Tensor:
        a, b, c = (3.4445, -4.7750, 2.0315)
        matrix = gradient.float()
        transpose = False
        if matrix.shape[0] < matrix.shape[1]:
            matrix = matrix.transpose(0, 1)
            transpose = True
        matrix = matrix / (matrix.norm() + 1.0e-7)
        for _ in range(steps):
            gram = matrix @ matrix.transpose(0, 1)
            poly = b * gram + c * gram @ gram
            matrix = a * matrix + poly @ matrix
        if transpose:
            matrix = matrix.transpose(0, 1)
        return matrix.to(dtype=gradient.dtype)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            ns_steps = group["ns_steps"]
            weight_decay = group["weight_decay"]
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                gradient = parameter.grad
                state = self.state[parameter]
                if weight_decay > 0:
                    parameter.mul_(1.0 - lr * weight_decay)
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(gradient)
                buffer = state["momentum_buffer"]
                buffer.mul_(momentum).add_(gradient)
                if parameter.ndim >= 2:
                    update = self._newton_schulz(buffer.reshape(buffer.shape[0], -1), ns_steps).view_as(parameter)
                else:
                    update = buffer
                parameter.add_(update, alpha=-lr)
        return loss


class CombinedOptimizer:
    def __init__(self, optimizers: list[Optimizer]) -> None:
        self.optimizers = optimizers
        self.param_groups = []
        for optimizer in optimizers:
            self.param_groups.extend(optimizer.param_groups)

    def zero_grad(self, set_to_none: bool = True) -> None:
        for optimizer in self.optimizers:
            optimizer.zero_grad(set_to_none=set_to_none)

    def step(self, closure=None) -> None:
        loss = None
        for index, optimizer in enumerate(self.optimizers):
            loss = optimizer.step(closure if index == 0 else None)
        return loss

    def state_dict(self):
        return [optimizer.state_dict() for optimizer in self.optimizers]

    def load_state_dict(self, state_dicts) -> None:
        for optimizer, state_dict in zip(self.optimizers, state_dicts, strict=False):
            optimizer.load_state_dict(state_dict)


_EMBEDDING_MODULE_TYPES = (nn.Embedding, nn.EmbeddingBag)
_NORM_MODULE_TYPES = (
    nn.LayerNorm,
    nn.BatchNorm1d,
    nn.BatchNorm2d,
    nn.BatchNorm3d,
    nn.GroupNorm,
    nn.InstanceNorm1d,
    nn.InstanceNorm2d,
    nn.InstanceNorm3d,
)


def _iter_trainable_parameters(model: nn.Module) -> Iterator[tuple[str, str, nn.Module, torch.nn.Parameter]]:
    owners: dict[int, tuple[str, str, nn.Module]] = {}
    for module_name, module in model.named_modules():
        for parameter_name, parameter in module.named_parameters(recurse=False):
            full_name = f"{module_name}.{parameter_name}" if module_name else parameter_name
            owners[id(parameter)] = (full_name, parameter_name, module)

    seen: set[int] = set()
    for full_name, parameter in model.named_parameters():
        if not parameter.requires_grad or id(parameter) in seen:
            continue
        seen.add(id(parameter))
        owner_name, parameter_name, module = owners.get(id(parameter), (full_name, full_name.rsplit(".", 1)[-1], model))
        yield owner_name, parameter_name, module, parameter


def _is_norm_module(module: nn.Module) -> bool:
    return isinstance(module, _NORM_MODULE_TYPES) or type(module).__name__.lower().endswith("norm")


def _is_embedding_module(module: nn.Module) -> bool:
    if isinstance(module, _EMBEDDING_MODULE_TYPES):
        return True
    module_name = type(module).__name__.lower()
    return "embedding" in module_name and "norm" not in module_name


def build_hybrid_optimizer(
    model: nn.Module,
    train_config,
    *,
    muon_lr: float | None = None,
    rowwise_adagrad_lr: float | None = None,
    adamw_lr: float | None = None,
    muon_momentum: float = 0.95,
    muon_ns_steps: int = 5,
    use_muon: bool = True,
    use_rowwise_adagrad: bool = True,
):
    resolved_muon_lr = train_config.learning_rate if muon_lr is None else muon_lr
    resolved_rowwise_lr = train_config.learning_rate if rowwise_adagrad_lr is None else rowwise_adagrad_lr
    resolved_adamw_lr = train_config.learning_rate if adamw_lr is None else adamw_lr

    rowwise_params: list[torch.nn.Parameter] = []
    muon_params: list[torch.nn.Parameter] = []
    adamw_decay: list[torch.nn.Parameter] = []
    adamw_no_decay: list[torch.nn.Parameter] = []

    for full_name, parameter_name, module, parameter in _iter_trainable_parameters(model):
        if _is_embedding_module(module) and parameter.ndim == 2:
            if use_rowwise_adagrad:
                rowwise_params.append(parameter)
            else:
                adamw_decay.append(parameter)
            continue

        if parameter_name == "bias" or full_name.endswith(".bias") or _is_norm_module(module):
            adamw_no_decay.append(parameter)
            continue

        if use_muon and parameter.ndim == 2:
            muon_params.append(parameter)
            continue

        adamw_decay.append(parameter)

    optimizers: list[Optimizer] = []
    if rowwise_params:
        optimizers.append(
            RowWiseAdagrad(
                rowwise_params,
                lr=resolved_rowwise_lr,
                weight_decay=train_config.weight_decay,
            )
        )
    if muon_params:
        optimizers.append(
            Muon(
                muon_params,
                lr=resolved_muon_lr,
                momentum=muon_momentum,
                ns_steps=muon_ns_steps,
                weight_decay=train_config.weight_decay,
            )
        )

    adamw_groups = []
    if adamw_decay:
        adamw_groups.append({"params": adamw_decay, "weight_decay": train_config.weight_decay})
    if adamw_no_decay:
        adamw_groups.append({"params": adamw_no_decay, "weight_decay": 0.0})
    if adamw_groups:
        optimizers.append(torch.optim.AdamW(adamw_groups, lr=resolved_adamw_lr))

    if not optimizers:
        raise ValueError("No trainable parameters were available to build an optimizer")
    if len(optimizers) == 1:
        return optimizers[0]
    return CombinedOptimizer(optimizers)


__all__ = ["CombinedOptimizer", "Muon", "build_hybrid_optimizer"]