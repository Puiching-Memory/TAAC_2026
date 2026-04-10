from __future__ import annotations

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

pytestmark = [pytest.mark.integration, pytest.mark.smoke]


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

    assert summary is not None
    assert "best_val_auc" in summary
    assert "model_profile" in summary
    assert "inference_profile" in summary
    assert "compute_profile" in summary
    assert summary["model_profile"]["parameter_size_mb"] > 0
    assert summary["inference_profile"]["estimated_end_to_end_inference_seconds"] >= 0
    assert summary["compute_profile"]["estimated_end_to_end_tflops_total"] > 0
    assert (Path(experiment.train.output_dir) / "summary.json").exists()
    assert (Path(experiment.train.output_dir) / "training_curves.json").exists()
    assert (Path(experiment.train.output_dir) / "training_curves.png").exists()
    assert (Path(experiment.train.output_dir) / "best.pt").exists()
    assert evaluation_path.exists()
    assert payload["model_name"] == "temp_experiment"


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
