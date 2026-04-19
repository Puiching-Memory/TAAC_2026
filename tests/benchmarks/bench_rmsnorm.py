from __future__ import annotations

import torch


class _RMSNorm(torch.nn.Module):
    def __init__(self, hidden_dim: int, eps: float = 1.0e-6) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(hidden_dim))
        self.eps = eps

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        variance = inputs.pow(2).mean(dim=-1, keepdim=True)
        normalized = inputs * torch.rsqrt(variance + self.eps)
        return normalized * self.weight


def test_rmsnorm_baseline(benchmark, benchmark_device, cuda_timer, performance_recorder) -> None:
    batch_size = 64
    sequence_length = 32
    hidden_dim = 128
    norm = _RMSNorm(hidden_dim).to(benchmark_device)
    tokens = torch.randn(batch_size, sequence_length, hidden_dim, device=benchmark_device)

    def run() -> torch.Tensor:
        return norm(tokens)

    benchmark.extra_info.update({
        "component": "rmsnorm",
        "phase": "baseline",
        "label": "phase-0",
        "metric": "latency",
    })
    benchmark(run)

    stats = cuda_timer(run)
    stats.update({
        "name": "rmsnorm",
        "component": "rmsnorm",
        "phase": "baseline",
        "label": "phase-0",
    })
    performance_recorder("rmsnorm", stats)
