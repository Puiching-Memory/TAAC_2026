from __future__ import annotations

import torch

from taac2026.domain.config import ModelConfig
from taac2026.infrastructure.io.default_data_pipeline import load_dataloaders
from tests.support import TinyExperimentModel


def test_e2e_train_step_baseline(benchmark, benchmark_device, benchmark_workspace, cuda_timer, performance_recorder) -> None:
    model_config = ModelConfig(name="tiny_benchmark", **benchmark_workspace.model_kwargs)
    model = TinyExperimentModel(benchmark_workspace.data_config, model_config, benchmark_workspace.data_config.dense_feature_dim)
    model = model.to(benchmark_device)
    model.train()

    train_loader, _, _ = load_dataloaders(
        config=benchmark_workspace.data_config,
        vocab_size=model_config.vocab_size,
        batch_size=2,
        eval_batch_size=2,
        num_workers=0,
        seed=7,
    )
    batch = next(iter(train_loader)).to(benchmark_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    def run() -> float:
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch)
        loss = loss_fn(logits, batch.labels)
        loss.backward()
        optimizer.step()
        return float(loss.detach().cpu().item())

    benchmark.extra_info.update({
        "component": "e2e_train_step",
        "phase": "baseline",
        "label": "phase-0",
        "metric": "latency",
    })
    benchmark(run)

    stats = cuda_timer(run)
    stats.update({
        "name": "e2e_train_step",
        "component": "e2e_train_step",
        "phase": "baseline",
        "label": "phase-0",
        "throughput": float(batch.batch_size) / max(stats["median_ms"] / 1e3, 1e-9),
    })
    performance_recorder("e2e_train_step", stats)
