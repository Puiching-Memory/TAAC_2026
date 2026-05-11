from __future__ import annotations

import logging
import math
from typing import NamedTuple

import pytest
import torch

import taac2026.infrastructure.runtime.trainer as trainer_module
from taac2026.infrastructure.runtime.checkpoint_io import PCVRTrainerSupportMixin
from taac2026.infrastructure.runtime.trainer import PCVRPointwiseTrainer
from taac2026.infrastructure.optimization.muon import Muon
from taac2026.infrastructure.runtime.execution import (
    EarlyStopping,
    PCVRLossConfig,
    PCVRLossTermConfig,
    RuntimeExecutionConfig,
    compute_pcvr_loss,
    maybe_compile_callable,
    maybe_prepare_internal_compile,
    runtime_amp_enabled,
    set_seed,
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


class _TrainingScalarDummyModel(_DummyModel):
    def __init__(self) -> None:
        super().__init__()
        self.diagnostics_enabled = False
        self.pending_scalars = 0

    def set_training_diagnostics_enabled(self, enabled: bool) -> None:
        self.diagnostics_enabled = bool(enabled)

    def forward(self, model_input):
        if self.diagnostics_enabled:
            self.pending_scalars += 1
        return super().forward(model_input)

    def consume_training_scalars(self, *, phase: str) -> dict[str, float]:
        if self.pending_scalars <= 0:
            return {}
        value = float(self.pending_scalars)
        self.pending_scalars = 0
        return {f"Dummy/{phase}/value": value}


class _RecordingWriter:
    def __init__(self) -> None:
        self.scalars: list[tuple[str, float, int]] = []
        self.flush_count = 0

    def add_scalar(self, tag: str, scalar_value: float, global_step: int) -> None:
        self.scalars.append((tag, float(scalar_value), int(global_step)))

    def flush(self) -> None:
        self.flush_count += 1


class _RecordingReporter:
    def __init__(self) -> None:
        self.scalars: list[tuple[str, float, int]] = []

    def train_step(self, *, step: int, loss: float, loss_components, dense_lr: float) -> None:
        self.scalars.append(("Loss/train", float(loss), int(step)))
        for name, value in loss_components.items():
            self.scalars.append((f"Loss/train/{name}", float(value), int(step)))
        self.scalars.append(("LR/dense", float(dense_lr), int(step)))

    def validation_step(
        self,
        *,
        step: int,
        auc: float,
        logloss: float,
        metrics,
        score_diagnostics,
        probe_metrics,
        probe_score_diagnostics,
    ) -> None:
        del metrics, score_diagnostics, probe_metrics, probe_score_diagnostics
        self.scalars.append(("AUC/valid", float(auc), int(step)))
        self.scalars.append(("LogLoss/valid", float(logloss), int(step)))

    def should_collect_model_scalars(self, *, phase: str, step: int | None, trainer) -> bool:
        del phase
        if step is None:
            return False
        interval = int(trainer.runtime_execution.progress_log_interval_steps)
        return step == 1 or (interval > 0 and step % interval == 0) or (trainer.max_steps > 0 and step == trainer.max_steps)

    def set_model_diagnostics_enabled(self, model: torch.nn.Module, enabled: bool) -> None:
        model.set_training_diagnostics_enabled(enabled)

    def consume_model_scalars(self, model: torch.nn.Module, *, phase: str) -> dict[str, float]:
        return model.consume_training_scalars(phase=phase)

    def model_scalars(self, *, phase: str, step: int, scalars) -> None:
        del phase
        for tag, value in scalars.items():
            self.scalars.append((tag, float(value), int(step)))


class _InternalCompileDummyModel(_DummyModel):
    uses_internal_compile = True

    def __init__(self) -> None:
        super().__init__()
        self.prepare_calls = 0

    def prepare_for_runtime_compile(self) -> None:
        self.prepare_calls += 1


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


class _SparseProbeDummyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bias = torch.nn.Parameter(torch.zeros(1))
        self.forward_calls = 0

    def forward(self, model_input):
        self.forward_calls += 1
        sparse_score = model_input.user_int_feats[:, 0].float() + model_input.item_int_feats[:, 0].float()
        return (sparse_score - 0.5 + self.bias).view(-1, 1)

    def predict(self, model_input):
        logits = self.forward(model_input)
        return logits, torch.empty(0)


class _AuxLossDummyModel(_DummyModel):
    def pcvr_loss_terms(self):
        return {"aux": self.bias.square().sum() + self.bias.new_tensor(0.25)}


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


def _sparse_probe_batch() -> dict[str, object]:
    return {
        "label": torch.tensor([0.0, 1.0], dtype=torch.float32),
        "_seq_domains": [],
        "user_int_feats": torch.tensor([[0], [1]], dtype=torch.long),
        "item_int_feats": torch.zeros((2, 1), dtype=torch.long),
        "user_dense_feats": torch.zeros((2, 0), dtype=torch.float32),
        "item_dense_feats": torch.zeros((2, 0), dtype=torch.float32),
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
        early_stopping=EarlyStopping(tmp_path / "best" / "model.safetensors", patience_steps=2),
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


def test_train_uses_runtime_execution_progress_log_interval(monkeypatch, tmp_path, log_capture) -> None:
    train_loader = [{"label": torch.tensor([0.0])} for _ in range(4)]
    valid_loader = [{"label": torch.tensor([0.0])} for _ in range(1)]
    trainer = PCVRPointwiseTrainer(
        model=_DummyModel(),
        model_input_type=object,
        train_loader=train_loader,
        valid_loader=valid_loader,
        lr=1e-3,
        max_steps=4,
        device="cpu",
        save_dir=tmp_path / "checkpoints",
        early_stopping=EarlyStopping(tmp_path / "best" / "model.safetensors", patience_steps=2),
        runtime_execution=RuntimeExecutionConfig(progress_log_interval_steps=2),
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
    assert any(message.startswith("Train progress 2/4 (50.0%)") and "loss=0.4000" in message for message in messages)
    assert not any(message.startswith("Train progress 3/4 (75.0%)") for message in messages)
    assert any(message.startswith("Train progress 4/4 (100.0%)") and "loss=0.2000" in message for message in messages)


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


def test_maybe_prepare_internal_compile_uses_model_boundary() -> None:
    model = _InternalCompileDummyModel()

    handled = maybe_prepare_internal_compile(model, enabled=True, label="sample model")

    assert handled is True
    assert model.prepare_calls == 1


def test_trainer_skips_whole_model_compile_when_model_handles_internal_compile(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    compile_calls = 0

    def recording_compile(callable_obj):
        nonlocal compile_calls
        compile_calls += 1
        return callable_obj

    model = _InternalCompileDummyModel()
    monkeypatch.setattr(torch, "compile", recording_compile)

    trainer = PCVRPointwiseTrainer(
        model=model,
        model_input_type=_DummyModelInput,
        train_loader=[],
        valid_loader=[],
        lr=1e-3,
        max_steps=1,
        device="cpu",
        save_dir=tmp_path / "checkpoints",
        early_stopping=EarlyStopping(tmp_path / "best" / "model.safetensors", patience_steps=2),
        runtime_execution=RuntimeExecutionConfig(compile=True),
    )

    assert model.prepare_calls == 1
    assert compile_calls == 0
    assert trainer.forward_model is model


def test_logical_train_sweep_steps_uses_dataset_without_calling_dataloader_len() -> None:
    class DatasetWithLogicalSweep:
        def logical_sweep_steps(self) -> int:
            return 7

    class LoaderWithFailingLen:
        dataset = DatasetWithLogicalSweep()

        def __len__(self) -> int:
            raise AssertionError("len(train_loader) should not be called")

    class TrainerSupport(PCVRTrainerSupportMixin):
        train_loader = LoaderWithFailingLen()

    assert TrainerSupport()._logical_train_sweep_steps() == 7


def test_infinite_train_batches_advances_step_window_sampler() -> None:
    class StepWindowSampler:
        def __init__(self) -> None:
            self.start_steps: list[int] = []

        def set_start_step(self, start_step: int) -> None:
            self.start_steps.append(start_step)

    class OneBatchLoader:
        sampler = StepWindowSampler()

        def __iter__(self):
            return iter([{"label": torch.tensor([float(len(self.sampler.start_steps))])}])

    class TrainerSupport(PCVRTrainerSupportMixin):
        train_loader = OneBatchLoader()

    iterator = TrainerSupport()._infinite_train_batches()

    first = next(iterator)
    second = next(iterator)

    assert TrainerSupport.train_loader.sampler.start_steps == [0, 1]
    assert first["label"].tolist() == [1.0]
    assert second["label"].tolist() == [2.0]


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
        early_stopping=EarlyStopping(tmp_path / "best" / "model.safetensors", patience_steps=2),
        runtime_execution=RuntimeExecutionConfig(amp=True, amp_dtype="float16", compile=False),
    )

    train_loss = trainer._train_step(_dummy_batch([1.0]))
    eval_logits, eval_labels = trainer._evaluate_step(_dummy_batch([0.0]))

    assert trainer.grad_scaler is None
    assert runtime_amp_enabled(trainer.runtime_execution, "cpu") is False
    assert math.isfinite(train_loss)
    assert eval_logits.shape == (1,)
    assert torch.equal(eval_labels, torch.tensor([0.0]))


def test_trainer_writes_model_training_scalars(tmp_path) -> None:
    reporter = _RecordingReporter()
    trainer = PCVRPointwiseTrainer(
        model=_TrainingScalarDummyModel(),
        model_input_type=_DummyModelInput,
        train_loader=[_dummy_batch([1.0])],
        valid_loader=[_dummy_batch([0.0]), _dummy_batch([1.0])],
        lr=1e-3,
        max_steps=2,
        device="cpu",
        save_dir=tmp_path / "checkpoints",
        early_stopping=EarlyStopping(tmp_path / "best" / "model.safetensors", patience_steps=2),
        reporter=reporter,
        runtime_execution=RuntimeExecutionConfig(compile=False, progress_log_interval_steps=2),
    )

    trainer.train()

    assert ("Dummy/train/value", 1.0, 1) in reporter.scalars
    assert ("Dummy/train/value", 1.0, 2) in reporter.scalars
    assert ("Dummy/valid/value", 1.0, 2) in reporter.scalars


def test_set_seed_can_disable_cudnn_determinism() -> None:
    set_seed(7, deterministic=False)

    assert torch.backends.cudnn.deterministic is False

    set_seed(7, deterministic=True)

    assert torch.backends.cudnn.deterministic is True


def test_trainer_train_step_matches_shared_focal_loss(tmp_path) -> None:
    expected_loss_config = PCVRLossConfig(
        terms=(
            PCVRLossTermConfig(
                name="focal",
                kind="focal",
                weight=1.0,
                focal_alpha=0.25,
                focal_gamma=1.5,
            ),
        )
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
        early_stopping=EarlyStopping(tmp_path / "best" / "model.safetensors", patience_steps=2),
        loss_terms=expected_loss_config.to_list(),
    )
    expected_loss, _components = compute_pcvr_loss(
        torch.zeros(1),
        torch.tensor([1.0]),
        expected_loss_config,
    )

    train_loss = trainer._train_step(_dummy_batch([1.0]))

    assert trainer.loss_config == expected_loss_config
    assert train_loss == pytest.approx(expected_loss.item())
    assert trainer.model.bias.item() > 0.0


def test_pairwise_auc_loss_penalizes_misordered_pairs() -> None:
    targets = torch.tensor([1.0, 0.0])
    good_logits = torch.tensor([2.0, -2.0])
    bad_logits = torch.tensor([-2.0, 2.0])
    loss_config = PCVRLossConfig(
        terms=(PCVRLossTermConfig(name="pairwise_auc", kind="pairwise_auc", weight=1.0),)
    )

    bad_loss, _bad_components = compute_pcvr_loss(bad_logits, targets, loss_config)
    good_loss, _good_components = compute_pcvr_loss(good_logits, targets, loss_config)

    assert bad_loss > good_loss


def test_trainer_combines_multiple_weighted_loss_terms(tmp_path) -> None:
    loss_config = PCVRLossConfig(
        terms=(
            PCVRLossTermConfig(name="bce", kind="bce", weight=1.0),
            PCVRLossTermConfig(name="aux", kind="model", weight=0.5),
        )
    )
    trainer = PCVRPointwiseTrainer(
        model=_AuxLossDummyModel(),
        model_input_type=_DummyModelInput,
        train_loader=[],
        valid_loader=[],
        lr=1e-3,
        max_steps=1,
        device="cpu",
        save_dir=tmp_path / "checkpoints",
        early_stopping=EarlyStopping(tmp_path / "best" / "model.safetensors", patience_steps=2),
        loss_terms=loss_config.to_list(),
    )
    expected_loss, expected_components = compute_pcvr_loss(
        torch.zeros(1),
        torch.tensor([1.0]),
        loss_config,
        model=trainer.model,
    )

    train_loss = trainer._train_step(_dummy_batch([1.0]))

    assert train_loss == pytest.approx(expected_loss.item())
    assert trainer.last_train_loss_components == pytest.approx(
        {name: float(value) for name, value in expected_components.items()}
    )


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
        early_stopping=EarlyStopping(tmp_path / "best" / "model.safetensors", patience_steps=2),
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
        early_stopping=EarlyStopping(tmp_path / "best" / "model.safetensors", patience_steps=2),
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
        early_stopping=EarlyStopping(tmp_path / "best" / "model.safetensors", patience_steps=2),
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
        early_stopping=EarlyStopping(tmp_path / "best" / "model.safetensors", patience_steps=2),
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
        patience_steps=6,
    )
    model = _DummyModel()

    early_stopping(0.8, model, step=3)
    early_stopping(0.7, model, step=6)

    assert early_stopping.counter == 3
    assert early_stopping.early_stop is False

    early_stopping(0.6, model, step=9)

    assert early_stopping.counter == 6
    assert early_stopping.early_stop is True


def test_trainer_uses_step_based_early_stopping_without_interval_scaling(monkeypatch, tmp_path) -> None:
    early_stopping = EarlyStopping(
        tmp_path / "best" / "model.safetensors",
        patience_steps=4,
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
        early_stopping=EarlyStopping(tmp_path / "best" / "model.safetensors", patience_steps=2),
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


def test_evaluate_uses_domain_auc_for_single_class(tmp_path) -> None:
    trainer = PCVRPointwiseTrainer(
        model=_DummyModel(),
        model_input_type=object,
        train_loader=[],
        valid_loader=[{"label": torch.tensor([1.0])}, {"label": torch.tensor([1.0])}],
        lr=1e-3,
        max_steps=1,
        device="cpu",
        save_dir=tmp_path / "checkpoints",
        early_stopping=EarlyStopping(tmp_path / "best" / "model.safetensors", patience_steps=2),
    )
    logits = iter((torch.tensor([0.0]), torch.tensor([1.0])))
    trainer._evaluate_step = lambda batch: (next(logits), batch["label"])

    auc, logloss = trainer.evaluate(step=1)

    assert auc == pytest.approx(0.5)
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
        early_stopping=EarlyStopping(tmp_path / "best" / "model.safetensors", patience_steps=2),
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


def test_evaluate_does_not_run_legacy_sparse_drop_probe(tmp_path) -> None:
    model = _SparseProbeDummyModel()
    trainer = PCVRPointwiseTrainer(
        model=model,
        model_input_type=_DummyModelInput,
        train_loader=[],
        valid_loader=[_sparse_probe_batch()],
        lr=1e-3,
        max_steps=1,
        device="cpu",
        save_dir=tmp_path / "checkpoints",
        early_stopping=EarlyStopping(tmp_path / "best" / "model.safetensors", patience_steps=2),
        validation_probe_mode="drop_nonseq_sparse",
        early_stopping_metric="probe_auc",
    )

    auc, logloss = trainer.evaluate(step=1)

    assert auc == 1.0
    assert math.isfinite(logloss)
    assert model.forward_calls == 1
    assert trainer.early_stopping_metric == "auc"
    assert trainer.last_eval_probe_metrics == {}
    assert "probe_auc" not in trainer.last_eval_metrics
    assert trainer.validation_early_stopping_score(auc, logloss) == pytest.approx(1.0)


def test_validation_result_uses_auc_for_legacy_probe_early_stopping_metric(tmp_path) -> None:
    early_stopping = EarlyStopping(tmp_path / "best" / "model.safetensors", patience_steps=2)
    trainer = PCVRPointwiseTrainer(
        model=_SparseProbeDummyModel(),
        model_input_type=_DummyModelInput,
        train_loader=[],
        valid_loader=[],
        lr=1e-3,
        max_steps=1,
        device="cpu",
        save_dir=tmp_path / "checkpoints",
        early_stopping=early_stopping,
        validation_probe_mode="drop_nonseq_sparse",
        early_stopping_metric="probe_auc",
    )
    trainer.last_eval_diagnostics = {"sample_count": 2}

    trainer._handle_validation_result(total_step=1, val_auc=0.95, val_logloss=0.2)

    assert trainer.early_stopping_metric == "auc"
    assert early_stopping.best_score == pytest.approx(0.95)
    assert early_stopping.best_extra_metrics is not None
    assert early_stopping.best_extra_metrics["early_stopping_metric"] == "auc"
    assert early_stopping.best_extra_metrics["best_val_AUC"] == pytest.approx(0.95)


def test_validation_result_saves_current_step_checkpoint(tmp_path) -> None:
    early_stopping = EarlyStopping(tmp_path / "best" / "model.safetensors", patience_steps=4)
    trainer = PCVRPointwiseTrainer(
        model=_SparseProbeDummyModel(),
        model_input_type=_DummyModelInput,
        train_loader=[],
        valid_loader=[],
        lr=1e-3,
        max_steps=1,
        device="cpu",
        save_dir=tmp_path / "checkpoints",
        early_stopping=early_stopping,
    )
    trainer.last_eval_diagnostics = {"sample_count": 2}

    trainer._handle_validation_result(total_step=1, val_auc=0.90, val_logloss=0.2)
    trainer._handle_validation_result(total_step=2, val_auc=0.95, val_logloss=0.25)

    checkpoint_root = tmp_path / "checkpoints"
    assert (checkpoint_root / "global_step1" / "model.safetensors").exists()
    assert (checkpoint_root / "global_step2" / "model.safetensors").exists()
    assert early_stopping.best_score == pytest.approx(0.95)


class _NaNProducingModel(torch.nn.Module):
    """Model that produces NaN logits to test training NaN detection."""

    def __init__(self) -> None:
        super().__init__()
        self.bias = torch.nn.Parameter(torch.zeros(1))

    def forward(self, model_input):
        del model_input
        return torch.tensor([[float("nan")]], dtype=torch.float32)

    def predict(self, model_input):
        return self.forward(model_input), torch.empty(0)


def test_train_step_skips_backward_when_loss_is_nan(tmp_path, log_capture) -> None:
    trainer = PCVRPointwiseTrainer(
        model=_NaNProducingModel(),
        model_input_type=_DummyModelInput,
        train_loader=[],
        valid_loader=[],
        lr=1e-3,
        max_steps=1,
        device="cpu",
        save_dir=tmp_path / "checkpoints",
        early_stopping=EarlyStopping(tmp_path / "best" / "model.safetensors", patience_steps=2),
    )
    initial_bias = trainer.model.bias.detach().clone()

    with log_capture.at_level(logging.WARNING):
        loss = trainer._train_step(_dummy_batch([1.0]))

    assert math.isnan(loss)
    # optim_step should NOT increment for a NaN step
    assert trainer.optim_step == 0
    # Model parameters should remain unchanged (no backward was performed)
    assert torch.equal(trainer.model.bias.detach(), initial_bias)
    assert "non-finite loss" in log_capture.text
    assert "Skipping backward" in log_capture.text
    # Loss components should all be nan
    assert all(math.isnan(v) for v in trainer.last_train_loss_components.values())


class _InfProducingModel(torch.nn.Module):
    """Model that produces inf logits to test training inf detection."""

    def __init__(self) -> None:
        super().__init__()
        self.bias = torch.nn.Parameter(torch.zeros(1))

    def forward(self, model_input):
        del model_input
        return torch.tensor([[float("inf")]], dtype=torch.float32)

    def predict(self, model_input):
        return self.forward(model_input), torch.empty(0)


def test_train_step_skips_backward_when_loss_is_inf(tmp_path, log_capture) -> None:
    """BCEWithLogitsLoss(inf) produces nan loss, which triggers the non-finite guard."""
    trainer = PCVRPointwiseTrainer(
        model=_InfProducingModel(),
        model_input_type=_DummyModelInput,
        train_loader=[],
        valid_loader=[],
        lr=1e-3,
        max_steps=1,
        device="cpu",
        save_dir=tmp_path / "checkpoints",
        early_stopping=EarlyStopping(tmp_path / "best" / "model.safetensors", patience_steps=2),
    )
    initial_bias = trainer.model.bias.detach().clone()

    with log_capture.at_level(logging.WARNING):
        loss = trainer._train_step(_dummy_batch([1.0]))

    # BCEWithLogitsLoss maps inf logits to nan loss
    assert not math.isfinite(loss)
    assert trainer.optim_step == 0
    assert torch.equal(trainer.model.bias.detach(), initial_bias)
    assert "non-finite loss" in log_capture.text


def test_train_step_proceeds_normally_with_finite_loss(tmp_path) -> None:
    trainer = PCVRPointwiseTrainer(
        model=_DummyModel(),
        model_input_type=_DummyModelInput,
        train_loader=[],
        valid_loader=[],
        lr=1e-3,
        max_steps=1,
        device="cpu",
        save_dir=tmp_path / "checkpoints",
        early_stopping=EarlyStopping(tmp_path / "best" / "model.safetensors", patience_steps=2),
    )
    initial_bias = trainer.model.bias.detach().clone()

    loss = trainer._train_step(_dummy_batch([1.0]))

    assert math.isfinite(loss)
    assert trainer.optim_step == 1
    # Parameters should have changed
    assert not torch.equal(trainer.model.bias.detach(), initial_bias)
