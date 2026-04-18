from __future__ import annotations

import torch


def test_embedding_lookup_baseline(benchmark, benchmark_device, cuda_timer, performance_recorder) -> None:
    batch_size = 64
    sequence_length = 32
    vocab_size = 131_072
    embedding_dim = 96
    embedding = torch.nn.Embedding(vocab_size, embedding_dim, device=benchmark_device)
    tokens = torch.randint(0, vocab_size, (batch_size, sequence_length), device=benchmark_device)

    def run() -> torch.Tensor:
        return embedding(tokens)

    benchmark.extra_info.update({
        "component": "embedding",
        "phase": "baseline",
        "label": "phase-0",
        "metric": "latency",
    })
    benchmark(run)

    stats = cuda_timer(run)
    stats.update({
        "name": "embedding_lookup",
        "component": "embedding",
        "phase": "baseline",
        "label": "phase-0",
        "throughput": float(tokens.numel()) / max(stats["median_ms"] / 1e3, 1e-9),
    })
    performance_recorder("embedding_lookup", stats)
