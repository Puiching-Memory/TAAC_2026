"""Muon optimizer with AdamW fallback for non-matrix parameters."""

from __future__ import annotations

import math
from collections.abc import Iterable

import torch


def _orthogonalize_update(
    update: torch.Tensor,
    *,
    steps: int,
    eps: float = 1e-12,
) -> torch.Tensor:
    original_dtype = update.dtype
    matrix = update.float().reshape(update.shape[0], -1)
    rows, cols = matrix.shape
    transposed = rows > cols
    if transposed:
        matrix = matrix.t()

    norm = matrix.norm()
    if not torch.isfinite(norm) or norm <= eps:
        return torch.zeros_like(update)

    matrix = matrix / norm.clamp_min(eps)
    for _ in range(max(1, steps)):
        gram = matrix @ matrix.t()
        matrix = 1.5 * matrix - 0.5 * gram @ matrix

    if transposed:
        matrix = matrix.t()
    return matrix.reshape_as(update).to(original_dtype)


class Muon(torch.optim.Optimizer):
    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        *,
        lr: float,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        weight_decay: float = 0.01,
        adamw_betas: tuple[float, float] = (0.9, 0.98),
        adamw_eps: float = 1e-8,
    ) -> None:
        if lr <= 0.0:
            raise ValueError(f"lr must be > 0, got {lr}")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"momentum must be in [0, 1), got {momentum}")
        if ns_steps < 1:
            raise ValueError(f"ns_steps must be >= 1, got {ns_steps}")
        if weight_decay < 0.0:
            raise ValueError(f"weight_decay must be >= 0, got {weight_decay}")
        beta1, beta2 = adamw_betas
        if not 0.0 <= beta1 < 1.0 or not 0.0 <= beta2 < 1.0:
            raise ValueError(f"adamw_betas must be in [0, 1), got {adamw_betas}")
        if adamw_eps <= 0.0:
            raise ValueError(f"adamw_eps must be > 0, got {adamw_eps}")

        defaults = {
            "lr": float(lr),
            "momentum": float(momentum),
            "nesterov": bool(nesterov),
            "ns_steps": int(ns_steps),
            "weight_decay": float(weight_decay),
            "adamw_betas": (float(beta1), float(beta2)),
            "adamw_eps": float(adamw_eps),
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):  # type: ignore[override]
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = float(group["lr"])
            momentum = float(group["momentum"])
            nesterov = bool(group["nesterov"])
            ns_steps = int(group["ns_steps"])
            weight_decay = float(group["weight_decay"])
            beta1, beta2 = group["adamw_betas"]
            adamw_eps = float(group["adamw_eps"])

            for parameter in group["params"]:
                grad = parameter.grad
                if grad is None:
                    continue
                if grad.is_sparse:
                    raise RuntimeError("Muon does not support sparse gradients")

                if parameter.ndim >= 2:
                    state = self.state[parameter]
                    momentum_buffer = state.setdefault("momentum_buffer", torch.zeros_like(parameter))
                    momentum_buffer.mul_(momentum).add_(grad)
                    update = grad.add(momentum_buffer, alpha=momentum) if nesterov else momentum_buffer
                    matrix = update.reshape(update.shape[0], -1)
                    update_scale = math.sqrt(max(1.0, matrix.shape[0] / max(1, matrix.shape[1])))
                    orthogonal_update = _orthogonalize_update(update, steps=ns_steps)
                    if weight_decay != 0.0:
                        parameter.mul_(1.0 - lr * weight_decay)
                    parameter.add_(orthogonal_update, alpha=-lr * update_scale)
                    continue

                state = self.state[parameter]
                if not state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(parameter)
                    state["exp_avg_sq"] = torch.zeros_like(parameter)

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                state["step"] += 1
                step = int(state["step"])

                if weight_decay != 0.0:
                    parameter.mul_(1.0 - lr * weight_decay)

                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                bias_correction1 = 1.0 - beta1**step
                bias_correction2 = 1.0 - beta2**step
                denom = exp_avg_sq.sqrt().div_(math.sqrt(bias_correction2)).add_(adamw_eps)
                step_size = lr / bias_correction1
                parameter.addcdiv_(exp_avg, denom, value=-step_size)

        return loss


__all__ = ["Muon"]