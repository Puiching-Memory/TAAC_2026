from __future__ import annotations

import torch

from .config import TrainConfig


def zeropower_via_newtonschulz5(gradient: torch.Tensor, steps: int = 5) -> torch.Tensor:
    assert gradient.ndim >= 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    update = gradient.bfloat16()
    if update.size(-2) > update.size(-1):
        update = update.mT

    update = update / (update.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        squared = update @ update.mT
        polynomial = b * squared + c * squared @ squared
        update = a * update + polynomial @ update

    if gradient.size(-2) > gradient.size(-1):
        update = update.mT
    return update


def muon_update(
    gradient: torch.Tensor,
    momentum: torch.Tensor,
    beta: float = 0.95,
    ns_steps: int = 5,
    nesterov: bool = True,
) -> torch.Tensor:
    momentum.lerp_(gradient, 1.0 - beta)
    update = gradient.lerp_(momentum, beta) if nesterov else momentum
    if update.ndim == 4:
        update = update.view(len(update), -1)
    update = zeropower_via_newtonschulz5(update, steps=ns_steps)
    update *= max(1.0, gradient.size(-2) / max(gradient.size(-1), 1)) ** 0.5
    return update


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr: float = 0.02, weight_decay: float = 0.0, momentum: float = 0.95) -> None:
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                state = self.state[parameter]
                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(parameter)
                update = muon_update(parameter.grad, state["momentum_buffer"], beta=group["momentum"])
                parameter.mul_(1 - group["lr"] * group["weight_decay"])
                parameter.add_(update.reshape(parameter.shape), alpha=-group["lr"])
        return loss


class CompositeOptimizer:
    def __init__(self, optimizers: list[torch.optim.Optimizer]) -> None:
        self.optimizers = optimizers

    def zero_grad(self, set_to_none: bool = True) -> None:
        for optimizer in self.optimizers:
            optimizer.zero_grad(set_to_none=set_to_none)

    def step(self) -> None:
        for optimizer in self.optimizers:
            optimizer.step()


def _should_use_muon(name: str, parameter: torch.nn.Parameter) -> bool:
    if parameter.ndim < 2:
        return False
    excluded_keywords = [
        "embedding",
        "position_embedding",
        "source_embedding",
        "sequence_group_embedding",
        "norm",
        "bias",
    ]
    return not any(keyword in name for keyword in excluded_keywords)


def build_optimizer(model: torch.nn.Module, config: TrainConfig) -> torch.optim.Optimizer | CompositeOptimizer:
    normalized = config.optimizer_name.lower()
    if normalized == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

    if normalized == "muon_adamw":
        adamw_parameters: list[torch.nn.Parameter] = []
        muon_parameters: list[torch.nn.Parameter] = []
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            if _should_use_muon(name, parameter):
                muon_parameters.append(parameter)
            else:
                adamw_parameters.append(parameter)

        optimizers: list[torch.optim.Optimizer] = []
        if adamw_parameters:
            optimizers.append(
                torch.optim.AdamW(
                    adamw_parameters,
                    lr=config.learning_rate,
                    weight_decay=config.weight_decay,
                )
            )
        if muon_parameters:
            optimizers.append(
                Muon(
                    muon_parameters,
                    lr=config.muon_learning_rate,
                    weight_decay=config.weight_decay,
                )
            )
        if len(optimizers) == 1:
            return optimizers[0]
        return CompositeOptimizer(optimizers)

    raise ValueError(f"Unsupported optimizer_name: {config.optimizer_name}")


__all__ = ["CompositeOptimizer", "Muon", "build_optimizer"]