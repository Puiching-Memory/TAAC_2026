from __future__ import annotations

import torch


def test_attention_forward_baseline(benchmark, benchmark_device, cuda_timer, performance_recorder) -> None:
    batch_size = 16
    sequence_length = 32
    hidden_dim = 128
    attention = torch.nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True, device=benchmark_device)
    tokens = torch.randn(batch_size, sequence_length, hidden_dim, device=benchmark_device)

    def run() -> torch.Tensor:
        return attention(tokens, tokens, tokens, need_weights=False)[0]

    benchmark.extra_info.update({
        "component": "attention",
        "phase": "baseline",
        "label": "phase-0",
        "metric": "latency",
    })
    benchmark(run)

    stats = cuda_timer(run)
    stats.update({
        "name": "attention_forward",
        "component": "attention",
        "phase": "baseline",
        "label": "phase-0",
        "throughput": float(batch_size * sequence_length) / max(stats["median_ms"] / 1e3, 1e-9),
    })
    performance_recorder("attention_forward", stats)
