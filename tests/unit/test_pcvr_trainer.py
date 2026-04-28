from __future__ import annotations

from contextlib import contextmanager
import logging
import math

import pytest
import torch

import taac2026.infrastructure.pcvr.trainer as trainer_module
from taac2026.infrastructure.pcvr.trainer import PCVRPointwiseTrainer
from taac2026.infrastructure.training.runtime import (
    BinaryClassificationLossConfig,
    EarlyStopping,
    RuntimeExecutionConfig,
)


class _DummyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bias = torch.nn.Parameter(torch.zeros(1))

    def forward(self, model_input):
        del model_input
        return self.bias.view(1, 1)

    def predict(self, model_input):
        logits = self.forward(model_input)
        return logits, torch.empty(0)


def test_train_logs_progress_when_tqdm_is_disabled(monkeypatch, tmp_path, caplog) -> None:
    train_loader = [{"label": torch.tensor([0.0])} for _ in range(4)]
    valid_loader = [{"label": torch.tensor([0.0])} for _ in range(3)]
    trainer = PCVRPointwiseTrainer(
        model=_DummyModel(),
        model_input_type=object,
        train_loader=train_loader,
        valid_loader=valid_loader,
        lr=1e-3,
        num_epochs=1,
        device="cpu",
        save_dir=tmp_path / "checkpoints",
        early_stopping=EarlyStopping(tmp_path / "best" / "model.pt", patience=2),
    )

    losses = iter((0.5, 0.4, 0.3, 0.2))
    monotonic_values = iter((100.0, 101.0, 102.0, 103.0, 104.0))
    monkeypatch.setattr(trainer_module, "_use_interactive_progress", lambda: False)
    monkeypatch.setattr(trainer_module.time, "monotonic", lambda: next(monotonic_values))
    monkeypatch.setattr(trainer, "_train_step", lambda batch: next(losses))
    monkeypatch.setattr(trainer, "evaluate", lambda epoch=None: (0.75, 0.25))

    with caplog.at_level(logging.INFO):
        trainer.train()

    messages = [record.getMessage() for record in caplog.records]
    assert "Train epoch 1 progress 1/4 (25.0%) | eta=0:00:03 | loss=0.5000" in messages
    assert "Train epoch 1 progress 4/4 (100.0%) | eta=0:00:00 | loss=0.2000" in messages
    assert "Epoch 1, Average Loss: 0.35" in messages


def test_trainer_runtime_execution_wraps_train_and_predict(monkeypatch, tmp_path) -> None:
    compile_labels: list[tuple[bool, str]] = []
    autocast_devices: list[str] = []
    original_autocast_context = trainer_module.RuntimeExecutionConfig.autocast_context

    def fake_maybe_compile_callable(callable_obj, *, enabled: bool, label: str):
        compile_labels.append((enabled, label))
        return callable_obj

    @contextmanager
    def recording_autocast(self, device):
        autocast_devices.append(str(device))
        with original_autocast_context(self, device):
            yield

    monkeypatch.setattr(trainer_module, "maybe_compile_callable", fake_maybe_compile_callable)
    monkeypatch.setattr(trainer_module.RuntimeExecutionConfig, "autocast_context", recording_autocast)

    trainer = PCVRPointwiseTrainer(
        model=_DummyModel(),
        model_input_type=object,
        train_loader=[],
        valid_loader=[],
        lr=1e-3,
        num_epochs=1,
        device="cpu",
        save_dir=tmp_path / "checkpoints",
        early_stopping=EarlyStopping(tmp_path / "best" / "model.pt", patience=2),
        runtime_execution=RuntimeExecutionConfig(amp=True, amp_dtype="float16", compile=True),
    )
    monkeypatch.setattr(trainer, "_make_model_input", lambda batch: object())

    train_loss = trainer._train_step({"label": torch.tensor([1.0])})
    eval_logits, eval_labels = trainer._evaluate_step({"label": torch.tensor([0.0])})

    assert train_loss >= 0.0
    assert eval_logits.shape == (1,)
    assert torch.equal(eval_labels, torch.tensor([0.0]))
    assert compile_labels == [
        (True, "PCVR training forward"),
        (True, "PCVR trainer predict"),
    ]
    assert autocast_devices == ["cpu", "cpu"]


def test_trainer_delegates_loss_to_shared_runtime(monkeypatch, tmp_path) -> None:
    captured: dict[str, object] = {}

    def fake_compute_binary_classification_loss(logits, targets, loss_config, reduction="mean"):
        captured["logits_shape"] = tuple(logits.shape)
        captured["targets"] = targets.detach().clone()
        captured["loss_config"] = loss_config
        captured["reduction"] = reduction
        return logits.mean() * 0 + 0.25

    monkeypatch.setattr(trainer_module, "compute_binary_classification_loss", fake_compute_binary_classification_loss)

    trainer = PCVRPointwiseTrainer(
        model=_DummyModel(),
        model_input_type=object,
        train_loader=[],
        valid_loader=[],
        lr=1e-3,
        num_epochs=1,
        device="cpu",
        save_dir=tmp_path / "checkpoints",
        early_stopping=EarlyStopping(tmp_path / "best" / "model.pt", patience=2),
        loss_type="FOCAL",
        focal_alpha=0.25,
        focal_gamma=1.5,
    )
    monkeypatch.setattr(trainer, "_make_model_input", lambda batch: object())

    train_loss = trainer._train_step({"label": torch.tensor([1.0])})

    assert train_loss == pytest.approx(0.25)
    assert captured["logits_shape"] == (1,)
    assert torch.equal(captured["targets"], torch.tensor([1.0]))
    assert captured["reduction"] == "mean"
    assert captured["loss_config"] == BinaryClassificationLossConfig(
        loss_type="focal",
        focal_alpha=0.25,
        focal_gamma=1.5,
    )


def test_evaluate_accepts_bfloat16_logits(tmp_path) -> None:
    trainer = PCVRPointwiseTrainer(
        model=_DummyModel(),
        model_input_type=object,
        train_loader=[],
        valid_loader=[{"label": torch.tensor([0.0])}, {"label": torch.tensor([1.0])}],
        lr=1e-3,
        num_epochs=1,
        device="cpu",
        save_dir=tmp_path / "checkpoints",
        early_stopping=EarlyStopping(tmp_path / "best" / "model.pt", patience=2),
    )
    logits = iter(
        (
            torch.tensor([0.0], dtype=torch.bfloat16),
            torch.tensor([1.0], dtype=torch.bfloat16),
        )
    )
    trainer._evaluate_step = lambda batch: (next(logits), batch["label"])

    auc, logloss = trainer.evaluate(epoch=1)

    assert auc == 1.0
    assert math.isfinite(logloss)


def test_evaluate_records_score_diagnostics(tmp_path, caplog) -> None:
    trainer = PCVRPointwiseTrainer(
        model=_DummyModel(),
        model_input_type=object,
        train_loader=[],
        valid_loader=[
            {"label": torch.tensor([0.0, 1.0])},
            {"label": torch.tensor([1.0, 0.0])},
        ],
        lr=1e-3,
        num_epochs=1,
        device="cpu",
        save_dir=tmp_path / "checkpoints",
        early_stopping=EarlyStopping(tmp_path / "best" / "model.pt", patience=2),
    )
    logits = iter(
        (
            torch.tensor([-2.0, 2.0]),
            torch.tensor([1.0, -1.0]),
        )
    )
    trainer._evaluate_step = lambda batch: (next(logits), batch["label"])

    with caplog.at_level(logging.INFO):
        auc, logloss = trainer.evaluate(epoch=1)

    assert auc == 1.0
    assert math.isfinite(logloss)
    assert trainer.last_eval_diagnostics["positive_count"] == 2
    assert trainer.last_eval_diagnostics["negative_count"] == 2
    assert trainer.last_eval_diagnostics["positive_score_mean"] > trainer.last_eval_diagnostics["negative_score_mean"]
    assert "Validation score diagnostics" in caplog.text
