from __future__ import annotations

from pathlib import Path

import pytest

from taac2026.application.training.profiling import (
    collect_loader_outputs,
    collect_model_profile,
    measure_latency,
)
from taac2026.domain.config import ModelConfig, TrainConfig
from tests.support import (
    TestWorkspace,
    build_local_data_pipeline,
    build_local_model_component,
    create_test_workspace,
)


@pytest.fixture
def test_workspace(tmp_path: Path) -> TestWorkspace:
    return create_test_workspace(tmp_path)


def _build_profile_model_and_batch(test_workspace: TestWorkspace):
    train_loader, _, data_stats = build_local_data_pipeline(
        test_workspace.data_config,
        ModelConfig(name="profile_test", **test_workspace.model_kwargs),
        TrainConfig(batch_size=2, eval_batch_size=2, num_workers=0),
    )
    model = build_local_model_component(
        test_workspace.data_config,
        ModelConfig(name="profile_test", **test_workspace.model_kwargs),
        data_stats.dense_dim,
    )
    return model, next(iter(train_loader))


def test_collect_loader_outputs_returns_empty_arrays_for_empty_loader(test_workspace: TestWorkspace) -> None:
    model, _ = _build_profile_model_and_batch(test_workspace)

    logits, labels, groups, loss = collect_loader_outputs(model, [], "cpu")

    assert logits.size == 0
    assert labels.size == 0
    assert groups.size == 0
    assert loss == pytest.approx(0.0)


def test_measure_latency_tracks_warmup_only_runs(test_workspace: TestWorkspace) -> None:
    model, batch = _build_profile_model_and_batch(test_workspace)

    latency = measure_latency(model, [batch], "cpu", warmup_steps=3, measure_steps=1)

    assert latency["warmup_batches"] == 1
    assert latency["warmup_samples"] == batch.batch_size
    assert latency["measured_batches"] == 0
    assert latency["measured_samples"] == 0
    assert latency["profiled_batches"] == 1


def test_measure_latency_zero_measure_steps_profiles_all_remaining_batches(
    test_workspace: TestWorkspace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, batch = _build_profile_model_and_batch(test_workspace)
    ticks = iter([1.0, 1.004, 2.0, 2.008])
    monkeypatch.setattr("taac2026.application.training.profiling.time.perf_counter", lambda: next(ticks))

    latency = measure_latency(model, [batch, batch, batch], "cpu", warmup_steps=1, measure_steps=0)

    assert latency["warmup_batches"] == 1
    assert latency["measured_batches"] == 2
    assert latency["mean_latency_ms_per_sample"] == pytest.approx(3.0)
    assert latency["profiled_batches"] == 3


def test_collect_model_profile_handles_empty_loader(test_workspace: TestWorkspace) -> None:
    model, _ = _build_profile_model_and_batch(test_workspace)

    profile = collect_model_profile(model, [], "cpu")

    assert profile["profile_batch_size"] == 0
    assert profile["flops_per_batch"] == pytest.approx(0.0)
    assert profile["profiled_wall_time_ms"] == pytest.approx(0.0)
    assert profile["operator_summary"]["operator_count"] == 0
    assert profile["operator_summary"]["top_operations"] == []
