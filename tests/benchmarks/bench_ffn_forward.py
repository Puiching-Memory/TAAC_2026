from __future__ import annotations

import torch


def test_ffn_forward_baseline(benchmark, benchmark_device, cuda_timer, performance_recorder) -> None:
    batch_size = 32
    sequence_length = 32
    hidden_dim = 128
    ffn = torch.nn.Sequential(
        torch.nn.Linear(hidden_dim, hidden_dim * 4),
        torch.nn.GELU(),
        torch.nn.Linear(hidden_dim * 4, hidden_dim),
    ).to(benchmark_device)
    tokens = torch.randn(batch_size, sequence_length, hidden_dim, device=benchmark_device)

    def run() -> torch.Tensor:
        return ffn(tokens)

    benchmark.extra_info.update({
        "component": "ffn",
        "phase": "baseline",
        "label": "phase-0",
        "metric": "latency",
    })
    benchmark(run)

    stats = cuda_timer(run)
    stats.update({
        "name": "ffn_forward",
        "component": "ffn",
        "phase": "baseline",
        "label": "phase-0",
        "throughput": float(batch_size * sequence_length) / max(stats["median_ms"] / 1e3, 1e-9),
    })
    performance_recorder("ffn_forward", stats)
