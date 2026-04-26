from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch

from taac2026.infrastructure.experiments.loader import load_experiment_package


EXPERIMENTS = [
    "config/symbiosis",
    "config/ctr_baseline",
    "config/deepcontextnet",
    "config/interformer",
    "config/onetrans",
    "config/hyformer",
    "config/unirec",
    "config/uniscaleformer",
]
MODEL_CLASSES = {
    "config/symbiosis": "PCVRSymbiosis",
    "config/ctr_baseline": "PCVRCTRBaseline",
    "config/deepcontextnet": "PCVRDeepContextNet",
    "config/interformer": "PCVRInterFormer",
    "config/onetrans": "PCVROneTrans",
    "config/hyformer": "PCVRHyFormer",
    "config/unirec": "PCVRUniRec",
    "config/uniscaleformer": "PCVRUniScaleFormer",
}
EXPERIMENT_NAMES = {
    "config/symbiosis": "pcvr_symbiosis",
    "config/ctr_baseline": "pcvr_ctr_baseline",
    "config/deepcontextnet": "pcvr_deepcontextnet",
    "config/interformer": "pcvr_interformer",
    "config/onetrans": "pcvr_onetrans",
    "config/hyformer": "pcvr_hyformer_paper",
    "config/unirec": "pcvr_unirec",
    "config/uniscaleformer": "pcvr_uniscaleformer",
}


def _load_model_module(experiment_path: str):
    model_path = Path(__file__).resolve().parents[2] / experiment_path / "model.py"
    spec = importlib.util.spec_from_file_location(experiment_path.replace("/", "_") + "_model", model_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sample_model_input(model_module):
    return model_module.ModelInput(
        user_int_feats=torch.tensor([[1, 2, 3], [4, 0, 1]], dtype=torch.long),
        item_int_feats=torch.tensor([[1], [2]], dtype=torch.long),
        user_dense_feats=torch.randn(2, 2),
        item_dense_feats=torch.randn(2, 1),
        seq_data={
            "seq_a": torch.tensor(
                [
                    [[1, 2, 0, 0], [2, 3, 0, 0]],
                    [[4, 1, 2, 3], [1, 2, 3, 4]],
                ],
                dtype=torch.long,
            ),
            "seq_b": torch.tensor([[[1, 0, 0]], [[2, 3, 0]]], dtype=torch.long),
        },
        seq_lens={
            "seq_a": torch.tensor([2, 4], dtype=torch.long),
            "seq_b": torch.tensor([1, 2], dtype=torch.long),
        },
        seq_time_buckets={
            "seq_a": torch.zeros(2, 4, dtype=torch.long),
            "seq_b": torch.zeros(2, 3, dtype=torch.long),
        },
    )


def _make_model(experiment_path: str, model_module):
    model_class = getattr(model_module, MODEL_CLASSES[experiment_path])
    return model_class(
        user_int_feature_specs=[(8, 0, 1), (7, 1, 2)],
        item_int_feature_specs=[(5, 0, 1)],
        user_dense_dim=2,
        item_dense_dim=1,
        seq_vocab_sizes={"seq_a": [6, 5], "seq_b": [4]},
        user_ns_groups=[[0], [1]],
        item_ns_groups=[[0]],
        d_model=16,
        emb_dim=8,
        num_blocks=1,
        num_heads=2,
        hidden_mult=2,
        dropout_rate=0.0,
        action_num=1,
        num_time_buckets=0,
        ns_tokenizer_type="rankmixer",
        user_ns_tokens=2,
        item_ns_tokens=1,
    )


@pytest.mark.parametrize("experiment_path", EXPERIMENTS)
def test_new_experiment_packages_load(experiment_path: str) -> None:
    experiment = load_experiment_package(experiment_path)

    assert experiment.name == EXPERIMENT_NAMES[experiment_path]
    assert experiment.package_dir is not None
    assert experiment.default_train_args
    assert experiment.metadata["kind"] == "pcvr"
    assert experiment.metadata["model_class"] == MODEL_CLASSES[experiment_path]
    assert (experiment.package_dir / "ns_groups.json").exists()
    assert "--ns_groups_json" in experiment.default_train_args
    assert "ns_groups.json" in experiment.default_train_args
    assert "--num_hyformer_blocks" not in experiment.default_train_args


@pytest.mark.parametrize("experiment_path", EXPERIMENTS)
def test_new_experiment_models_forward_and_predict(experiment_path: str) -> None:
    model_module = _load_model_module(experiment_path)
    if experiment_path != "config/hyformer":
        assert not hasattr(model_module, "PCVRHyFormer")
    assert hasattr(model_module, MODEL_CLASSES[experiment_path])
    model = _make_model(experiment_path, model_module)
    model_input = _sample_model_input(model_module)

    logits = model(model_input)
    loss = logits.sum()
    loss.backward()

    model.eval()
    with torch.no_grad():
        predicted_logits, embeddings = model.predict(model_input)

    assert logits.shape == (2, 1)
    assert predicted_logits.shape == (2, 1)
    assert embeddings.shape[0] == 2
    assert model.num_ns > 0
    assert torch.isfinite(logits).all()
    assert torch.isfinite(predicted_logits).all()


def test_symbiosis_trims_sequence_padding_before_attention() -> None:
    model_module = _load_model_module("config/symbiosis")
    model = _make_model("config/symbiosis", model_module)
    model_input = model_module.ModelInput(
        user_int_feats=torch.tensor([[1, 2, 3], [4, 0, 1]], dtype=torch.long),
        item_int_feats=torch.tensor([[1], [2]], dtype=torch.long),
        user_dense_feats=torch.randn(2, 2),
        item_dense_feats=torch.randn(2, 1),
        seq_data={
            "seq_a": torch.tensor(
                [
                    [[1, 2, 0, 0, 0, 0], [2, 3, 0, 0, 0, 0]],
                    [[4, 1, 2, 0, 0, 0], [1, 2, 3, 0, 0, 0]],
                ],
                dtype=torch.long,
            ),
            "seq_b": torch.tensor([[[1, 0, 0, 0, 0]], [[2, 3, 0, 0, 0]]], dtype=torch.long),
        },
        seq_lens={
            "seq_a": torch.tensor([2, 3], dtype=torch.long),
            "seq_b": torch.tensor([1, 2], dtype=torch.long),
        },
        seq_time_buckets={
            "seq_a": torch.zeros(2, 6, dtype=torch.long),
            "seq_b": torch.zeros(2, 5, dtype=torch.long),
        },
    )

    sequences, masks, lengths = model._encode_sequences(model_input)

    assert [tensor.shape[1] for tensor in sequences] == [3, 2]
    assert [mask.shape[1] for mask in masks] == [3, 2]
    assert [length.tolist() for length in lengths] == [[2, 3], [1, 2]]


def test_symbiosis_unified_attention_uses_context_bottleneck() -> None:
    model_module = _load_model_module("config/symbiosis")
    model = _make_model("config/symbiosis", model_module)
    model_input = _sample_model_input(model_module)
    observed_self_attention_lengths: list[int] = []

    handle = model.unified_blocks[0].attention.register_forward_pre_hook(
        lambda _module, inputs: observed_self_attention_lengths.append(int(inputs[0].shape[1]))
    )
    try:
        logits = model(model_input)
    finally:
        handle.remove()

    sequence_token_count = sum(int(sequence.shape[2]) for sequence in model_input.seq_data.values())
    context_token_count = model.num_prompt_tokens + model.num_ns

    assert logits.shape == (2, 1)
    assert observed_self_attention_lengths == [context_token_count]
    assert context_token_count < context_token_count + sequence_token_count