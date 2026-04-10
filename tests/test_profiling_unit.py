from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from taac2026.application.training import profiling
from taac2026.application.training.profiling import (
    _count_latency_probe_batches,
    collect_compute_profile,
    collect_inference_profile,
    collect_loader_outputs,
    collect_model_profile,
    measure_latency,
)
from taac2026.domain.config import DataConfig, ModelConfig, TrainConfig
from taac2026.domain.experiment import ExperimentSpec
from taac2026.domain.types import BatchTensors, DataStats


pytestmark = pytest.mark.unit


def _batch(batch_size: int, fill: float = 1.0) -> BatchTensors:
    tokens = torch.ones((batch_size, 3), dtype=torch.long)
    mask = torch.ones((batch_size, 3), dtype=torch.bool)
    sequence_tokens = torch.ones((batch_size, 2, 3), dtype=torch.long)
    sequence_mask = torch.ones((batch_size, 2, 3), dtype=torch.bool)
    dense_features = torch.full((batch_size, 4), fill, dtype=torch.float32)
    labels = torch.tensor([0.0, 1.0, 0.0, 1.0][:batch_size], dtype=torch.float32)
    indices = torch.arange(batch_size, dtype=torch.long)
    return BatchTensors(
        candidate_tokens=tokens,
        candidate_mask=mask,
        context_tokens=tokens,
        context_mask=mask,
        history_tokens=tokens,
        history_mask=mask,
        sequence_tokens=sequence_tokens,
        sequence_mask=sequence_mask,
        dense_features=dense_features,
        labels=labels,
        user_indices=indices,
        item_indices=indices,
        item_logq=torch.zeros(batch_size, dtype=torch.float32),
    )


class _ConstantModel(nn.Module):
    def __init__(self, value: float = 0.0) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.tensor([value], dtype=torch.float32))

    def forward(self, batch: BatchTensors) -> torch.Tensor:
        return self.bias.expand(batch.batch_size)


def _make_experiment() -> ExperimentSpec:
    return ExperimentSpec(
        name="profiling",
        data=DataConfig(dataset_path="dummy"),
        model=ModelConfig(name="profiling", vocab_size=16, embedding_dim=4, hidden_dim=4),
        train=TrainConfig(epochs=2, batch_size=2, eval_batch_size=2, latency_warmup_steps=1, latency_measure_steps=2),
        build_data_pipeline=lambda *_: None,
        build_model_component=lambda *_: _ConstantModel(),
        build_loss_stack=lambda *_: (nn.BCEWithLogitsLoss(), SimpleNamespace(enabled=False, requires_aux=False)),
        build_optimizer_component=lambda model, _train: torch.optim.SGD(model.parameters(), lr=0.1),
    )


def test_collect_loader_outputs_returns_empty_arrays_for_empty_loader() -> None:
    logits, labels, groups, loss = collect_loader_outputs(_ConstantModel(), [], torch.device("cpu"))

    assert logits.shape == (0,)
    assert labels.shape == (0,)
    assert groups.shape == (0,)
    assert loss == 0.0


def test_measure_latency_respects_warmup_and_open_ended_measurement(monkeypatch) -> None:
    loader = [_batch(2), _batch(2), _batch(2)]
    clock = iter([0.0, 0.10, 1.0, 1.30])
    monkeypatch.setattr(profiling.time, "perf_counter", lambda: next(clock))

    latency = measure_latency(_ConstantModel(), loader, torch.device("cpu"), warmup_steps=1, measure_steps=0)

    assert latency["mean_latency_ms_per_sample"] == pytest.approx(100.0)
    assert latency["p95_latency_ms_per_sample"] >= latency["mean_latency_ms_per_sample"]


def test_collect_inference_profile_scales_latency_by_sample_count() -> None:
    experiment = _make_experiment()
    loader = [_batch(2), _batch(1)]

    profile = collect_inference_profile(
        experiment,
        loader,
        {"mean_latency_ms_per_sample": 10.0, "p95_latency_ms_per_sample": 15.0},
    )

    assert profile["val_sample_count"] == 3
    assert profile["estimated_end_to_end_inference_seconds"] == pytest.approx(0.03)
    assert profile["estimated_end_to_end_inference_seconds_p95"] == pytest.approx(0.045)


def test_collect_model_profile_returns_zero_flops_for_empty_loader() -> None:
    profile = collect_model_profile(_ConstantModel(), [], torch.device("cpu"))

    assert profile["profile_batch_size"] == 0
    assert profile["flops_per_batch"] == 0.0
    assert profile["flops_per_sample"] == 0.0


def test_collect_compute_profile_scales_profiled_flops(monkeypatch) -> None:
    experiment = _make_experiment()
    train_loader = [_batch(2), _batch(2)]
    val_loader = [_batch(1), _batch(1), _batch(1)]
    data_stats = DataStats(dense_dim=4, pos_weight=1.0, train_size=4, val_size=3)

    class _DummyProfiler:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def key_averages(self):
            return SimpleNamespace(total_average=lambda: SimpleNamespace(flops=120.0))

    monkeypatch.setattr(profiling, "profile", lambda *args, **kwargs: _DummyProfiler())

    result = collect_compute_profile(
        experiment=experiment,
        model=_ConstantModel(),
        loss_fn=nn.BCEWithLogitsLoss(),
        train_loader=train_loader,
        val_loader=val_loader,
        data_stats=data_stats,
        device=torch.device("cpu"),
        model_profile={"flops_per_sample": 30.0},
    )

    assert result["train_batches_per_epoch"] == 2
    assert result["val_batches_per_epoch"] == 3
    assert result["latency_probe_batches"] == 3
    assert result["latency_probe_samples"] == 3
    assert result["train_step_flops_per_sample"] == pytest.approx(60.0)
    assert result["estimated_train_flops_total"] == pytest.approx(480.0)
    assert result["estimated_eval_flops_total"] == pytest.approx(180.0)
    assert result["estimated_latency_probe_flops_total"] == pytest.approx(90.0)
    assert result["estimated_end_to_end_flops_total"] == pytest.approx(750.0)


def test_latency_probe_batch_counter_handles_zero_and_negative_limits() -> None:
    assert _count_latency_probe_batches(total_batches=0, warmup_steps=3, measure_steps=1) == 0
    assert _count_latency_probe_batches(total_batches=5, warmup_steps=-1, measure_steps=-1) == 5
