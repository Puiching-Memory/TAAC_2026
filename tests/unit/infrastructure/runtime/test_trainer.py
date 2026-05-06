from __future__ import annotations

import logging
import math
from typing import NamedTuple

import pytest
import torch

import taac2026.infrastructure.runtime.trainer as trainer_module
from taac2026.infrastructure.runtime.trainer import PCVRPointwiseTrainer
from taac2026.infrastructure.optimization.muon import Muon
from taac2026.infrastructure.runtime.execution import (
    BinaryClassificationLossConfig,
    EarlyStopping,
    RuntimeExecutionConfig,
    compute_binary_classification_loss,
    maybe_compile_callable,
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


class _MatrixDummyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([[0.5, -0.5], [0.25, -0.25]], dtype=torch.float32))
        self.bias = torch.nn.Parameter(torch.zeros(1))

    def forward(self, model_input):
        del model_input
        return self.weight.mean().view(1, 1) + self.bias.view(1, 1)

    def predict(self, model_input):
        logits = self.forward(model_input)
        return logits, torch.empty(0)


class _DummyModelInput(NamedTuple):
    user_int_feats: torch.Tensor
    item_int_feats: torch.Tensor
    user_dense_feats: torch.Tensor
    item_dense_feats: torch.Tensor
    seq_data: dict[str, torch.Tensor]
    seq_lens: dict[str, torch.Tensor]
    seq_time_buckets: dict[str, torch.Tensor]


def _dummy_batch(labels: list[float]) -> dict[str, object]:
    batch_size = len(labels)
    return {
        "label": torch.tensor(labels, dtype=torch.float32),
        "_seq_domains": [],
        "user_int_feats": torch.zeros((batch_size, 1), dtype=torch.long),
        "item_int_feats": torch.zeros((batch_size, 1), dtype=torch.long),
        "user_dense_feats": torch.zeros((batch_size, 0), dtype=torch.float32),
        "item_dense_feats": torch.zeros((batch_size, 0), dtype=torch.float32),
    }


def test_train_logs_progress_when_tqdm_is_disabled(monkeypatch, tmp_path, log_capture) -> None:
    train_loader = [{"label": torch.tensor([0.0])} for _ in range(4)]
    valid_loader = [{"label": torch.tensor([0.0])} for _ in range(3)]
    trainer = PCVRPointwiseTrainer(
        model=_DummyModel(),
        model_input_type=object,
        train_loader=train_loader,
        valid_loader=valid_loader,
        lr=1e-3,
        max_steps=4,
        device="cpu",
        save_dir=tmp_path / "checkpoints",
        early_stopping=EarlyStopping(tmp_path / "best" / "model.safetensors", patience=2),
    )

    losses = iter((0.5, 0.4, 0.3, 0.2))
    monotonic_values = iter((100.0, 101.0, 102.0, 103.0, 104.0))
    monkeypatch.setattr(trainer_module, "_use_interactive_progress", lambda: False)
    monkeypatch.setattr(trainer_module.time, "monotonic", lambda: next(monotonic_values))
    monkeypatch.setattr(trainer, "_train_step", lambda batch: next(losses))
    monkeypatch.setattr(trainer, "evaluate", lambda step=None: (0.75, 0.25))

    with log_capture.at_level(logging.INFO):
        trainer.train()

    messages = [record.getMessage() for record in log_capture.records]
    assert any(message.startswith("Train progress 1/4 (25.0%)") and "loss=0.5000" in message for message in messages)
    assert any(message.startswith("Train progress 4/4 (100.0%)") and "loss=0.2000" in message for message in messages)
    assert any(message.startswith("Train step 4, Average Loss:") and message.endswith("0.35") for message in messages)


def test_maybe_compile_callable_falls_back_to_eager_on_compile_error(
    monkeypatch: pytest.MonkeyPatch,
    log_capture,
) -> None:
    def sample_callable(value):
        return value

    def failing_compile(callable_obj):
        del callable_obj
        raise RuntimeError("boom")

    monkeypatch.setattr(torch, "compile", failing_compile)

    with log_capture.at_level(logging.WARNING):
        compiled = maybe_compile_callable(sample_callable, enabled=True, label="sample callable")

    assert compiled is sample_callable
    assert "Failed to compile sample callable" in log_capture.text
    assert "falling back to eager execution" in log_capture.text


def test_trainer_runtime_execution_runs_train_and_predict_on_cpu(tmp_path) -> None:
    trainer = PCVRPointwiseTrainer(
        model=_DummyModel(),
        model_input_type=_DummyModelInput,
        train_loader=[],
        valid_loader=[],
        lr=1e-3,
        max_steps=1,
        device="cpu",
        save_dir=tmp_path / "checkpoints",
        early_stopping=EarlyStopping(tmp_path / "best" / "model.safetensors", patience=2),
        runtime_execution=RuntimeExecutionConfig(amp=True, amp_dtype="float16", compile=False),
    )

    train_loss = trainer._train_step(_dummy_batch([1.0]))
    eval_logits, eval_labels = trainer._evaluate_step(_dummy_batch([0.0]))

    assert trainer.grad_scaler is None
    assert trainer.runtime_execution.amp_enabled_for("cpu") is False
    assert math.isfinite(train_loss)
    assert eval_logits.shape == (1,)
    assert torch.equal(eval_labels, torch.tensor([0.0]))


def test_trainer_train_step_matches_shared_focal_loss(tmp_path) -> None:
    expected_loss_config = BinaryClassificationLossConfig(
        loss_type="focal",
        focal_alpha=0.25,
        focal_gamma=1.5,
    )
    trainer = PCVRPointwiseTrainer(
        model=_DummyModel(),
        model_input_type=_DummyModelInput,
        train_loader=[],
        valid_loader=[],
        lr=1e-3,
        max_steps=1,
        device="cpu",
        save_dir=tmp_path / "checkpoints",
        early_stopping=EarlyStopping(tmp_path / "best" / "model.safetensors", patience=2),
        loss_type="FOCAL",
        focal_alpha=0.25,
        focal_gamma=1.5,
    )
    expected_loss = compute_binary_classification_loss(
        torch.zeros(1),
        torch.tensor([1.0]),
        expected_loss_config,
    ).item()

    train_loss = trainer._train_step(_dummy_batch([1.0]))

    assert trainer.loss_config == expected_loss_config
    assert train_loss == pytest.approx(expected_loss)
    assert trainer.model.bias.item() > 0.0


def test_pairwise_auc_loss_penalizes_misordered_pairs() -> None:
    targets = torch.tensor([1.0, 0.0])
    good_logits = torch.tensor([2.0, -2.0])
    bad_logits = torch.tensor([-2.0, 2.0])
    loss_config = BinaryClassificationLossConfig(pairwise_auc_weight=1.0)

    assert compute_binary_classification_loss(bad_logits, targets, loss_config) > compute_binary_classification_loss(good_logits, targets, loss_config)


@pytest.mark.parametrize("dense_optimizer_type", ["orthogonal_adamw", "fused_adamw", "muon"])
def test_trainer_accepts_supported_dense_optimizers(tmp_path, dense_optimizer_type: str) -> None:
    trainer = PCVRPointwiseTrainer(
        model=_DummyModel(),
        model_input_type=object,
        train_loader=[],
        valid_loader=[],
        lr=1e-3,
        max_steps=1,
        device="cpu",
        save_dir=tmp_path / "checkpoints",
        early_stopping=EarlyStopping(tmp_path / "best" / "model.safetensors", patience=2),
        dense_optimizer_type=dense_optimizer_type,
    )

    assert trainer.dense_optimizer_type == dense_optimizer_type


def test_trainer_builds_fused_adamw_optimizer(tmp_path) -> None:
    trainer = PCVRPointwiseTrainer(
        model=_DummyModel(),
        model_input_type=object,
        train_loader=[],
        valid_loader=[],
        lr=1e-3,
        max_steps=1,
        device="cpu",
        save_dir=tmp_path / "checkpoints",
        early_stopping=EarlyStopping(tmp_path / "best" / "model.safetensors", patience=2),
        dense_optimizer_type="fused_adamw",
    )

    assert isinstance(trainer.dense_optimizer, torch.optim.AdamW)
    assert trainer.dense_optimizer.defaults["fused"] is True


def test_trainer_muon_updates_matrix_parameters(tmp_path) -> None:
    model = _MatrixDummyModel()
    trainer = PCVRPointwiseTrainer(
        model=model,
        model_input_type=_DummyModelInput,
        train_loader=[],
        valid_loader=[],
        lr=1e-3,
        max_steps=1,
        device="cpu",
        save_dir=tmp_path / "checkpoints",
        early_stopping=EarlyStopping(tmp_path / "best" / "model.safetensors", patience=2),
        dense_optimizer_type="muon",
    )
    initial_weight = model.weight.detach().clone()
    initial_bias = model.bias.detach().clone()

    loss = trainer._train_step(_dummy_batch([1.0]))

    assert isinstance(trainer.dense_optimizer, Muon)
    assert math.isfinite(loss)
    assert not torch.equal(model.weight.detach(), initial_weight)
    assert not torch.equal(model.bias.detach(), initial_bias)


def test_trainer_applies_dense_warmup_and_cosine_decay(tmp_path) -> None:
    trainer = PCVRPointwiseTrainer(
        model=_DummyModel(),
        model_input_type=_DummyModelInput,
        train_loader=[],
        valid_loader=[],
        lr=1e-3,
        max_steps=4,
        device="cpu",
        save_dir=tmp_path / "checkpoints",
        early_stopping=EarlyStopping(tmp_path / "best" / "model.safetensors", patience=2),
        scheduler_type="cosine",
        warmup_steps=2,
        min_lr_ratio=0.2,
    )

    observed_lrs = []
    for _ in range(4):
        trainer._train_step(_dummy_batch([1.0]))
        observed_lrs.append(trainer.current_dense_lr)

    assert observed_lrs == pytest.approx([5e-4, 1e-3, 6e-4, 2e-4])
    assert trainer.dense_optimizer.param_groups[0]["lr"] == pytest.approx(2e-4)


def test_early_stopping_supports_step_based_patience(tmp_path) -> None:
    early_stopping = EarlyStopping(
        tmp_path / "best" / "model.safetensors",
        patience=2,
        patience_unit="steps",
        step_scale=3,
    )
    model = _DummyModel()

    early_stopping(0.8, model, step=3)
    early_stopping(0.7, model, step=6)

    assert early_stopping.counter == 3
    assert early_stopping.early_stop is False

    early_stopping(0.6, model, step=9)

    assert early_stopping.counter == 6
    assert early_stopping.early_stop is True


def test_trainer_scales_step_based_early_stopping_by_eval_interval(monkeypatch, tmp_path) -> None:
    early_stopping = EarlyStopping(
        tmp_path / "best" / "model.safetensors",
        patience=2,
        patience_unit="steps",
    )
    trainer = PCVRPointwiseTrainer(
        model=_DummyModel(),
        model_input_type=object,
        train_loader=[{"label": torch.tensor([0.0])}],
        valid_loader=[],
        lr=1e-3,
        max_steps=10,
        device="cpu",
        save_dir=tmp_path / "checkpoints",
        early_stopping=early_stopping,
        eval_every_n_steps=2,
    )

    train_step_count = 0

    def fake_train_step(batch):
        del batch
        nonlocal train_step_count
        train_step_count += 1
        return 0.1

    metrics = iter(((0.9, 0.1), (0.8, 0.2), (0.7, 0.3)))
    monkeypatch.setattr(trainer, "_train_step", fake_train_step)
    monkeypatch.setattr(trainer, "evaluate", lambda step=None: next(metrics))

    trainer.train()

    assert early_stopping.step_scale == 2
    assert early_stopping.resolved_patience == 4
    assert early_stopping.counter == 4
    assert early_stopping.early_stop is True
    assert train_step_count == 6


def test_evaluate_accepts_bfloat16_logits(tmp_path) -> None:
    trainer = PCVRPointwiseTrainer(
        model=_DummyModel(),
        model_input_type=object,
        train_loader=[],
        valid_loader=[{"label": torch.tensor([0.0])}, {"label": torch.tensor([1.0])}],
        lr=1e-3,
        max_steps=1,
        device="cpu",
        save_dir=tmp_path / "checkpoints",
        early_stopping=EarlyStopping(tmp_path / "best" / "model.safetensors", patience=2),
    )
    logits = iter(
        (
            torch.tensor([0.0], dtype=torch.bfloat16),
            torch.tensor([1.0], dtype=torch.bfloat16),
        )
    )
    trainer._evaluate_step = lambda batch: (next(logits), batch["label"])

    auc, logloss = trainer.evaluate(step=1)

    assert auc == 1.0
    assert math.isfinite(logloss)


def test_evaluate_records_score_diagnostics(tmp_path, log_capture) -> None:
    trainer = PCVRPointwiseTrainer(
        model=_DummyModel(),
        model_input_type=object,
        train_loader=[],
        valid_loader=[
            {"label": torch.tensor([0.0, 1.0])},
            {"label": torch.tensor([1.0, 0.0])},
        ],
        lr=1e-3,
        max_steps=1,
        device="cpu",
        save_dir=tmp_path / "checkpoints",
        early_stopping=EarlyStopping(tmp_path / "best" / "model.safetensors", patience=2),
    )
    logits = iter(
        (
            torch.tensor([-2.0, 2.0]),
            torch.tensor([1.0, -1.0]),
        )
    )
    trainer._evaluate_step = lambda batch: (next(logits), batch["label"])

    with log_capture.at_level(logging.INFO):
        auc, logloss = trainer.evaluate(step=1)

    assert auc == 1.0
    assert math.isfinite(logloss)
    assert trainer.last_eval_diagnostics["positive_count"] == 2
    assert trainer.last_eval_diagnostics["negative_count"] == 2
    assert trainer.last_eval_diagnostics["positive_score_mean"] > trainer.last_eval_diagnostics["negative_score_mean"]
    assert "Validation score diagnostics" in log_capture.text
