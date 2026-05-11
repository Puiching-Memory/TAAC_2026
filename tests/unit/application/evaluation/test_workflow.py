from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch

from taac2026.application.evaluation.workflow import (
    PCVRPredictionContext,
    PCVRPredictionDataBundle,
    PCVRPredictionRunner,
    default_run_prediction_loop,
)
from taac2026.domain.runtime_config import RuntimeExecutionConfig
from taac2026.infrastructure.modeling.model_contract import ModelInput


def test_default_prediction_loop_runs_under_inference_mode(tmp_path: Path) -> None:
    observed_inference_mode: list[bool] = []
    batch = {
        "label": torch.tensor([0.0, 1.0]),
        "user_id": ["u0", "u1"],
        "timestamp": torch.tensor([100, 200]),
        "user_int_feats": torch.ones(2, 1, dtype=torch.long),
        "item_int_feats": torch.ones(2, 1, dtype=torch.long),
        "user_dense_feats": torch.zeros(2, 1),
        "item_dense_feats": torch.zeros(2, 1),
        "_seq_domains": [],
    }

    def predict_fn(model_input: ModelInput) -> tuple[torch.Tensor, torch.Tensor]:
        assert isinstance(model_input, ModelInput)
        observed_inference_mode.append(torch.is_inference_mode_enabled())
        return torch.tensor([[-2.0], [2.0]]), torch.empty(2, 0)

    context = PCVRPredictionContext(
        model_module=SimpleNamespace(ModelInput=ModelInput),
        model_class_name="DummyModel",
        package_dir=tmp_path,
        dataset_path=tmp_path / "eval.parquet",
        schema_path=tmp_path / "schema.json",
        checkpoint_path=tmp_path / "checkpoint" / "model.safetensors",
        batch_size=2,
        num_workers=0,
        device="cpu",
        is_training_data=False,
        dataset_role="inference",
        config={},
        runtime_execution=RuntimeExecutionConfig(compile=False),
    )
    data_bundle = PCVRPredictionDataBundle(dataset=SimpleNamespace(num_rows=2), loader=[batch])
    runner = PCVRPredictionRunner(model=object(), predict_fn=predict_fn)

    payload = default_run_prediction_loop(context, data_bundle, runner)

    assert observed_inference_mode == [True]
    assert payload["processed_rows"] == 2
    assert payload["batch_count"] == 1
    assert [record["user_id"] for record in payload["records"]] == ["u0", "u1"]