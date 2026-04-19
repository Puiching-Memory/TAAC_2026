from __future__ import annotations

import pytest
import torch
from torch import nn

from taac2026.application.training.cli import parse_train_args
from taac2026.application.training.runtime_optimization import prepare_runtime_execution, runtime_optimization_cli_args
from taac2026.domain.config import TrainConfig


def test_prepare_runtime_execution_disables_cpu_float16_amp() -> None:
    model = nn.Linear(4, 2)

    runtime_execution = prepare_runtime_execution(
        model,
        TrainConfig(enable_amp=True, amp_dtype="float16"),
        "cpu",
    )

    assert runtime_execution.amp_requested is True
    assert runtime_execution.amp_active is False
    assert runtime_execution.amp_resolved_dtype is None
    assert runtime_execution.amp_reason is not None


def test_prepare_runtime_execution_allows_cpu_bfloat16_amp() -> None:
    model = nn.Linear(4, 2)

    runtime_execution = prepare_runtime_execution(
        model,
        TrainConfig(enable_amp=True, amp_dtype="bfloat16"),
        "cpu",
    )

    assert runtime_execution.amp_requested is True
    assert runtime_execution.amp_active is True
    assert runtime_execution.amp_resolved_dtype == "bfloat16"
    assert runtime_execution.uses_grad_scaler is False


def test_prepare_runtime_execution_rejects_unknown_amp_dtype() -> None:
    model = nn.Linear(4, 2)

    with pytest.raises(ValueError, match="Unsupported AMP dtype"):
        prepare_runtime_execution(model, TrainConfig(enable_amp=True, amp_dtype="fp8"), "cpu")


def test_prepare_runtime_execution_calls_torch_compile(monkeypatch: pytest.MonkeyPatch) -> None:
    model = nn.Linear(4, 2)
    captured: dict[str, object] = {}

    def fake_compile(input_model, **kwargs):
        captured["model"] = input_model
        captured["kwargs"] = kwargs
        return nn.Sequential(input_model)

    monkeypatch.setattr(torch, "compile", fake_compile)

    runtime_execution = prepare_runtime_execution(
        model,
        TrainConfig(enable_torch_compile=True, torch_compile_backend="inductor", torch_compile_mode="max-autotune"),
        "cpu",
    )

    assert captured["model"] is model
    assert captured["kwargs"] == {"backend": "inductor", "mode": "max-autotune"}
    assert runtime_execution.compile_requested is True
    assert runtime_execution.compile_active is True
    assert runtime_execution.execution_model is not model


def test_prepare_runtime_execution_compile_failure_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    model = nn.Linear(4, 2)

    def fake_compile(input_model, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(torch, "compile", fake_compile)

    with pytest.raises(RuntimeError, match="torch.compile failed"):
        prepare_runtime_execution(model, TrainConfig(enable_torch_compile=True), "cpu")


def test_runtime_optimization_cli_args_omits_disabled_flags() -> None:
    assert runtime_optimization_cli_args(TrainConfig()) == []


def test_runtime_optimization_cli_args_emits_enabled_flags() -> None:
    args = runtime_optimization_cli_args(
        TrainConfig(
            enable_torch_compile=True,
            torch_compile_backend="inductor",
            torch_compile_mode="max-autotune",
            enable_amp=True,
            amp_dtype="bfloat16",
        )
    )

    assert args == [
        "--compile",
        "--compile-backend",
        "inductor",
        "--compile-mode",
        "max-autotune",
        "--amp",
        "--amp-dtype",
        "bfloat16",
    ]


def test_parse_train_args_accepts_runtime_optimization_flags() -> None:
    args = parse_train_args(
        [
            "--experiment",
            "config/baseline",
            "--compile",
            "--compile-backend",
            "inductor",
            "--compile-mode",
            "max-autotune",
            "--amp",
            "--amp-dtype",
            "bfloat16",
        ]
    )

    assert args.compile is True
    assert args.compile_backend == "inductor"
    assert args.compile_mode == "max-autotune"
    assert args.amp is True
    assert args.amp_dtype == "bfloat16"