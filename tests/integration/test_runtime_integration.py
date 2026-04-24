from __future__ import annotations

import json
from pathlib import Path

import pytest

from taac2026.application.evaluation.service import evaluate_checkpoint
from taac2026.application.training.service import run_training
from taac2026.domain.config import ModelConfig, TrainConfig
from taac2026.domain.experiment import ExperimentSpec
from taac2026.domain.runtime import Arbiter, Blackboard, Layer, LayerStack, Packet
from taac2026.infrastructure.experiments.loader import load_experiment_package
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


def test_blackboard_and_arbiter_gate_optional_layers() -> None:
    blackboard = Blackboard({"metrics.count": 0, "switches.logging": False})

    def core_handler(packet: Packet, board: Blackboard) -> Packet:
        board.increment("metrics.count", 1)
        return packet

    def logging_handler(packet: Packet, board: Blackboard) -> Packet:
        board.put("logging.called", True)
        return packet

    stack = LayerStack(
        [
            Layer("core", ("demo",), core_handler),
            Layer("logging", ("demo",), logging_handler, subsystem="logging"),
        ],
        Arbiter({"logging": True}),
    )

    stack.dispatch(Packet("demo"), blackboard)

    assert blackboard.require("metrics.count") == 1
    assert not blackboard.has("logging.called")


def test_folder_experiment_clone_keeps_settings_isolated(test_workspace: TestWorkspace) -> None:
    experiment = ExperimentSpec(
        name="clone_test",
        data=test_workspace.data_config,
        model=ModelConfig(name="clone_test", **test_workspace.model_kwargs),
        train=TrainConfig(
            seed=7,
            epochs=1,
            batch_size=2,
            eval_batch_size=2,
            output_dir=str(test_workspace.root / "clone"),
        ),
        build_data_pipeline=build_local_data_pipeline,
        build_model_component=build_local_model_component,
        build_loss_stack=build_local_loss_stack,
        build_optimizer_component=build_local_optimizer_component,
    )

    clone = experiment.clone()
    clone.train.seed = 99
    clone.data.max_seq_len = 16

    assert experiment.train.seed == 7
    assert experiment.data.max_seq_len == 4
    assert clone.train.seed == 99
    assert clone.data.max_seq_len == 16


def test_experiment_package_runs_end_to_end_with_visualization_switch(test_workspace: TestWorkspace) -> None:
    experiment_path = test_workspace.write_experiment_package(switches={"logging": False, "visualization": True})
    experiment = load_experiment_package(experiment_path)

    summary = run_training(experiment)
    evaluation_path = test_workspace.root / "evaluation.json"
    payload = evaluate_checkpoint(experiment_path=experiment_path, output_path=evaluation_path)
    training_predictions_path = Path(experiment.train.output_dir) / "validation_predictions.jsonl"
    evaluation_predictions_path = test_workspace.root / "evaluation.validation_predictions.jsonl"
    training_predictions = [json.loads(line) for line in training_predictions_path.read_text(encoding="utf-8").splitlines()]
    evaluation_predictions = [json.loads(line) for line in evaluation_predictions_path.read_text(encoding="utf-8").splitlines()]

    assert summary is not None
    assert "best_val_auc" in summary
    assert "profiling" in summary
    assert "model_profile" in summary
    assert "inference_profile" in summary
    assert "compute_profile" in summary
    assert summary["profiling"]["schema_version"] == 2
    assert summary["profiling"]["latency"]["mean_latency_ms_per_sample"] == summary["mean_latency_ms_per_sample"]
    assert summary["profiling"]["model_profile"]["parameter_size_mb"] == summary["model_profile"]["parameter_size_mb"]
    assert summary["model_profile"]["profile_scope"] == "synthetic_fixed_forward"
    assert summary["model_profile"]["profile_input_kind"] == "synthetic_fixed_batch"
    assert "external_profilers" in summary["profiling"]
    assert summary["runtime_optimization"]["torch_compile"]["requested"] is False
    assert payload["runtime_optimization"]["amp"]["requested"] is False
    assert summary["model_profile"]["parameter_size_mb"] > 0
    assert summary["inference_profile"]["val_sample_count"] > 0
    assert "estimated_end_to_end_inference_seconds" not in summary["inference_profile"]
    assert "estimated_end_to_end_tflops_total" not in summary["compute_profile"]
    assert summary["compute_profile"]["train_step_tflops"] > 0
    assert (Path(experiment.train.output_dir) / "summary.json").exists()
    assert (Path(experiment.train.output_dir) / "training_curves.json").exists()
    assert (Path(experiment.train.output_dir) / "training_curves.png").exists()
    assert (Path(experiment.train.output_dir) / "best.pt").exists()
    assert training_predictions_path.exists()
    assert evaluation_path.exists()
    assert evaluation_predictions_path.exists()
    assert payload["model_name"] == "temp_experiment"
    assert summary["validation_predictions_path"] == str(training_predictions_path)
    assert payload["validation_predictions_path"] == str(evaluation_predictions_path)
    assert len(training_predictions) == summary["inference_profile"]["val_sample_count"]
    assert len(evaluation_predictions) == summary["inference_profile"]["val_sample_count"]
    assert set(training_predictions[0]) == {"item_id", "raw_label", "sample_index", "score", "target", "timestamp", "user_id"}
    assert set(evaluation_predictions[0]) == {"item_id", "raw_label", "sample_index", "score", "target", "timestamp", "user_id"}


def test_run_training_is_reproducible_for_same_seed(test_workspace: TestWorkspace) -> None:
    experiment_path = test_workspace.write_experiment_package()
    experiment_a = load_experiment_package(experiment_path)
    experiment_b = load_experiment_package(experiment_path)
    experiment_a.train.output_dir = str(test_workspace.root / "repro_a")
    experiment_b.train.output_dir = str(test_workspace.root / "repro_b")

    summary_a = run_training(experiment_a)
    summary_b = run_training(experiment_b)

    assert summary_a["best_val_auc"] == summary_b["best_val_auc"]
    assert summary_a["metrics"]["auc"] == summary_b["metrics"]["auc"]
