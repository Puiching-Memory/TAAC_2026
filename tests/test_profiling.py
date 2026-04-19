from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from taac2026.application.training.external_profilers import build_training_external_profiler_plan
from taac2026.application.training.profiling import collect_inference_profile, measure_latency
from taac2026.application.training.service import run_training
from taac2026.domain.config import ModelConfig, TrainConfig
from taac2026.domain.experiment import ExperimentSpec
from tests.support import (
    TestWorkspace,
    build_local_data_pipeline,
    build_local_loss_stack,
    build_local_model_component,
    build_local_optimizer_component,
    create_test_workspace,
)


@pytest.fixture
def test_workspace(tmp_path: Path) -> TestWorkspace:
    return create_test_workspace(tmp_path)


def test_measure_latency_reports_rich_statistics(test_workspace: TestWorkspace, monkeypatch: pytest.MonkeyPatch) -> None:
    train_config = TrainConfig(batch_size=2, eval_batch_size=2, num_workers=0)
    train_loader, val_loader, data_stats = build_local_data_pipeline(
        test_workspace.data_config,
        ModelConfig(name="profile_test", **test_workspace.model_kwargs),
        train_config,
    )
    del train_loader
    batch = next(iter(val_loader))
    loader = [batch, batch]
    model = build_local_model_component(test_workspace.data_config, ModelConfig(name="profile_test", **test_workspace.model_kwargs), data_stats.dense_dim)

    ticks = iter([1.0, 1.010, 2.0, 2.030])
    monkeypatch.setattr("taac2026.application.training.profiling.time.perf_counter", lambda: next(ticks))

    latency = measure_latency(model, loader, "cpu", warmup_steps=0, measure_steps=2)

    assert latency["profile_schema_version"] == 1
    assert latency["measured_batches"] == 2
    assert latency["measured_samples"] == 4
    assert latency["mean_latency_ms_per_sample"] == pytest.approx(10.0)
    assert latency["p50_latency_ms_per_sample"] == pytest.approx(10.0)
    assert latency["p95_latency_ms_per_sample"] == pytest.approx(14.5)
    assert latency["min_latency_ms_per_sample"] == pytest.approx(5.0)
    assert latency["max_latency_ms_per_sample"] == pytest.approx(15.0)
    assert latency["latency_std_ms_per_sample"] == pytest.approx(5.0)


def test_collect_inference_profile_accepts_sample_count(test_workspace: TestWorkspace) -> None:
    experiment = ExperimentSpec(
        name="profile_test",
        data=test_workspace.data_config,
        model=ModelConfig(name="profile_test", **test_workspace.model_kwargs),
        train=TrainConfig(
            seed=7,
            epochs=1,
            batch_size=2,
            eval_batch_size=2,
            output_dir=str(test_workspace.root / "profile_test"),
            latency_warmup_steps=1,
            latency_measure_steps=2,
        ),
        build_data_pipeline=build_local_data_pipeline,
        build_model_component=build_local_model_component,
        build_loss_stack=build_local_loss_stack,
        build_optimizer_component=build_local_optimizer_component,
    )

    inference_profile = collect_inference_profile(
        experiment,
        12,
        {
            "device": "cpu",
            "measured_batches": 2,
            "measured_samples": 4,
            "mean_latency_ms_per_sample": 2.5,
            "p50_latency_ms_per_sample": 2.0,
            "p95_latency_ms_per_sample": 3.0,
        },
    )

    assert inference_profile["profile_schema_version"] == 1
    assert inference_profile["val_sample_count"] == 12
    assert inference_profile["latency_observed_batches"] == 2
    assert inference_profile["estimated_end_to_end_inference_seconds"] == pytest.approx(0.03)
    assert inference_profile["estimated_end_to_end_inference_seconds_p50"] == pytest.approx(0.024)
    assert inference_profile["estimated_end_to_end_inference_seconds_p95"] == pytest.approx(0.036)


def test_build_training_external_profiler_plan_contains_uv_commands(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "taac2026.application.training.external_profilers._resolve_profiler_executable",
        lambda tool: f"C:/tools/{tool}.exe",
    )
    monkeypatch.setattr(
        "taac2026.application.training.external_profilers._read_profiler_version",
        lambda tool, executable: f"{tool} version",
    )

    plan = build_training_external_profiler_plan(
        device="cuda",
        output_dir=tmp_path,
        experiment_path="config/baseline",
    )

    assert plan["schema_version"] == 1
    assert plan["tools"]["ncu"]["available"] is True
    assert plan["tools"]["nsys"]["available"] is True
    assert plan["tools"]["ncu"]["suggested_command"][:6] == [
        "ncu",
        "--set",
        "full",
        "--target-processes",
        "all",
        "-o",
    ]
    assert plan["tools"]["nsys"]["suggested_command"][:7] == [
        "nsys",
        "profile",
        "--trace",
        "cuda,nvtx,osrt",
        "--sample",
        "none",
        "-o",
    ]
    assert "uv run taac-train --experiment config/baseline" in plan["tools"]["ncu"]["suggested_command_string"]


def test_build_training_external_profiler_plan_includes_runtime_flags(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "taac2026.application.training.external_profilers._resolve_profiler_executable",
        lambda tool: f"C:/tools/{tool}.exe",
    )
    monkeypatch.setattr(
        "taac2026.application.training.external_profilers._read_profiler_version",
        lambda tool, executable: f"{tool} version",
    )

    plan = build_training_external_profiler_plan(
        device="cuda",
        output_dir=tmp_path,
        experiment_path="config/baseline",
        train_config=TrainConfig(enable_torch_compile=True, torch_compile_backend="inductor", enable_amp=True, amp_dtype="bfloat16"),
    )

    command = plan["tools"]["ncu"]["suggested_command_string"]
    assert "--compile" in command
    assert "--compile-backend inductor" in command
    assert "--amp --amp-dtype bfloat16" in command


def test_build_evaluation_external_profiler_plan_includes_runtime_flags(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "taac2026.application.training.external_profilers._resolve_profiler_executable",
        lambda tool: f"C:/tools/{tool}.exe",
    )
    monkeypatch.setattr(
        "taac2026.application.training.external_profilers._read_profiler_version",
        lambda tool, executable: f"{tool} version",
    )

    from taac2026.application.training.external_profilers import build_evaluation_external_profiler_plan

    plan = build_evaluation_external_profiler_plan(
        device="cuda",
        output_dir=tmp_path,
        experiment_path="config/baseline",
        checkpoint_path=tmp_path / "best.pt",
        output_path=tmp_path / "evaluation.json",
        run_dir=tmp_path / "run",
        train_config=TrainConfig(enable_torch_compile=True, torch_compile_mode="max-autotune", enable_amp=True, amp_dtype="bfloat16"),
    )

    command = plan["tools"]["nsys"]["suggested_command_string"]
    assert "taac-evaluate single" in command
    assert "--compile" in command
    assert "--compile-mode max-autotune" in command
    assert "--amp --amp-dtype bfloat16" in command


def test_run_training_includes_unified_profiling_report(test_workspace: TestWorkspace) -> None:
    experiment = ExperimentSpec(
        name="profile_test",
        data=test_workspace.data_config,
        model=ModelConfig(name="profile_test", **test_workspace.model_kwargs),
        train=TrainConfig(
            seed=7,
            epochs=1,
            batch_size=2,
            eval_batch_size=2,
            num_workers=0,
            output_dir=str(test_workspace.root / "profile_training"),
            latency_warmup_steps=0,
            latency_measure_steps=1,
        ),
        build_data_pipeline=build_local_data_pipeline,
        build_model_component=build_local_model_component,
        build_loss_stack=build_local_loss_stack,
        build_optimizer_component=build_local_optimizer_component,
    )

    summary = run_training(experiment)

    assert summary["profiling"]["schema_version"] == 1
    assert summary["profiling"]["device"] == summary["profiling"]["latency"]["device"]
    assert summary["profiling"]["latency"]["mean_latency_ms_per_sample"] == summary["mean_latency_ms_per_sample"]
    assert summary["profiling"]["model_profile"]["profile_schema_version"] == 1
    assert summary["profiling"]["compute_profile"]["profile_schema_version"] == 1
    assert summary["profiling"]["inference_profile"]["profile_schema_version"] == 1
    assert summary["profiling"]["model_profile"]["operator_summary"]["top_operations"]
    assert summary["profiling"]["compute_profile"]["train_operator_summary"]["top_operations"]
    assert "external_profilers" in summary["profiling"]
    assert summary["runtime_optimization"]["torch_compile"]["active"] is False
    assert summary["runtime_optimization"]["amp"]["active"] is False
    assert "ncu" in summary["profiling"]["external_profilers"]["tools"]
    assert "nsys" in summary["profiling"]["external_profilers"]["tools"]
    assert (Path(experiment.train.output_dir) / "profiling" / "external_profilers.json").exists()
    script_extension = ".ps1" if os.name == "nt" else ".sh"
    assert (Path(experiment.train.output_dir) / "profiling" / f"profile_ncu{script_extension}").exists()
    assert (Path(experiment.train.output_dir) / "profiling" / f"profile_nsys{script_extension}").exists()


def test_run_training_enables_cpu_bfloat16_amp(test_workspace: TestWorkspace) -> None:
    experiment = ExperimentSpec(
        name="profile_amp_test",
        data=test_workspace.data_config,
        model=ModelConfig(name="profile_amp_test", **test_workspace.model_kwargs),
        train=TrainConfig(
            seed=7,
            epochs=1,
            batch_size=2,
            eval_batch_size=2,
            num_workers=0,
            output_dir=str(test_workspace.root / "profile_amp_training"),
            latency_warmup_steps=0,
            latency_measure_steps=1,
            enable_amp=True,
            amp_dtype="bfloat16",
        ),
        build_data_pipeline=build_local_data_pipeline,
        build_model_component=build_local_model_component,
        build_loss_stack=build_local_loss_stack,
        build_optimizer_component=build_local_optimizer_component,
    )

    summary = run_training(experiment)
    checkpoint = Path(experiment.train.output_dir) / "best.pt"
    payload = torch.load(checkpoint, map_location="cpu")

    assert summary["runtime_optimization"]["amp"]["requested"] is True
    assert summary["runtime_optimization"]["amp"]["active"] is True
    assert summary["runtime_optimization"]["amp"]["resolved_dtype"] == "bfloat16"
    assert payload["runtime_optimization"]["amp"]["active"] is True