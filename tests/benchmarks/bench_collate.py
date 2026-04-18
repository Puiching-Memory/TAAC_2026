from __future__ import annotations

from taac2026.infrastructure.io.default_data_pipeline import load_dataloaders


def test_collate_batch_baseline(benchmark, benchmark_workspace, cuda_timer, performance_recorder) -> None:
    train_loader, _, _ = load_dataloaders(
        config=benchmark_workspace.data_config,
        vocab_size=257,
        batch_size=2,
        eval_batch_size=2,
        num_workers=0,
        seed=7,
    )

    def run():
        return next(iter(train_loader))

    benchmark.extra_info.update({
        "component": "collate",
        "phase": "baseline",
        "label": "phase-0",
        "metric": "latency",
    })
    batch = benchmark(run)

    stats = cuda_timer(run)
    stats.update({
        "name": "collate",
        "component": "collate",
        "phase": "baseline",
        "label": "phase-0",
        "throughput": float(batch.batch_size) / max(stats["median_ms"] / 1e3, 1e-9),
    })
    performance_recorder("collate", stats)
