from __future__ import annotations

import pytest
import torch
from torch import nn

from taac2026.domain.features import FeatureSchema, FeatureTableSpec
from taac2026.infrastructure.nn.embedding import TorchRecEmbeddingBagAdapter
from taac2026.infrastructure.nn.quantization import normalize_quantization_mode, quantize_model_for_inference
from taac2026.infrastructure.nn.triton_attention import reference_attention, triton_attention
from taac2026.infrastructure.nn.triton_ffn import reference_ffn_activation, triton_ffn_activation


def test_normalize_quantization_mode_rejects_legacy_aliases() -> None:
    assert normalize_quantization_mode(None) == "none"
    with pytest.raises(ValueError, match="Unsupported quantization mode"):
        normalize_quantization_mode("off")
    with pytest.raises(ValueError, match="Unsupported quantization mode"):
        normalize_quantization_mode("linear-int8")


def test_quantize_model_for_inference_quantizes_linear_layers() -> None:
    model = nn.Sequential(
        nn.Linear(8, 16),
        nn.SiLU(),
        nn.Linear(16, 4),
    )

    quantized_model, summary = quantize_model_for_inference(model, "int8")

    assert summary["active"] is True
    assert summary["mode"] == "int8"
    assert summary["device"] == "cpu"
    assert summary["quantized_linear_layers"] > 0
    sample = torch.randn(2, 8)
    output = quantized_model(sample)
    assert output.shape == (2, 4)


def test_quantize_model_for_inference_keeps_original_model_unmodified() -> None:
    model = nn.Sequential(
        nn.Linear(8, 16),
        nn.SiLU(),
        nn.Linear(16, 4),
    )

    quantized_model, summary = quantize_model_for_inference(model, "int8")

    assert summary["active"] is True
    assert quantized_model is not model
    assert isinstance(model[0].weight, nn.Parameter)
    assert type(quantized_model[0].weight).__name__ == "Int8Tensor"


def test_quantize_model_for_inference_rejects_torchrec_embedding_bag_collections() -> None:
    feature_schema = FeatureSchema(
        tables=(
            FeatureTableSpec(name="user_tokens", family="user", num_embeddings=64, embedding_dim=4),
        ),
        dense_dim=0,
    )

    class TinyTorchRecModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.adapter = TorchRecEmbeddingBagAdapter(feature_schema, table_names=("user_tokens",))
            self.output = nn.Linear(4, 2)

        def forward(self, sparse_features) -> torch.Tensor:
            pooled = self.adapter.forward_dict(sparse_features)["user_tokens"]
            return self.output(pooled)

    model = TinyTorchRecModel()

    with pytest.raises(ValueError, match="does not support TorchRec EmbeddingBagCollection modules"):
        quantize_model_for_inference(model, "int8")


def test_reference_attention_supports_fp8_precision_mode() -> None:
    query = torch.randn(2, 2, 4, 8)
    key = torch.randn(2, 2, 4, 8)
    value = torch.randn(2, 2, 4, 8)

    output = reference_attention(query, key, value, precision="fp8-e4m3fn")

    assert output.shape == query.shape
    assert torch.isfinite(output).all().item()


def test_triton_attention_torch_backend_matches_fp8_reference() -> None:
    query = torch.randn(1, 2, 5, 8)
    key = torch.randn(1, 2, 5, 8)
    value = torch.randn(1, 2, 5, 8)
    key_mask = torch.tensor([[True, True, True, False, False]])

    expected = reference_attention(query, key, value, key_mask=key_mask, precision="fp8-e5m2")
    actual = triton_attention(query, key, value, key_mask=key_mask, backend="torch", precision="fp8-e5m2")

    assert torch.allclose(actual, expected, atol=1.0e-6, rtol=0.0)


def test_triton_ffn_activation_torch_backend_matches_fp8_reference() -> None:
    projected = torch.randn(3, 4, 16)

    expected = reference_ffn_activation(projected, "silu", precision="fp8-e4m3fn")
    actual = triton_ffn_activation(projected, "silu", backend="torch", precision="fp8-e4m3fn")

    assert torch.allclose(actual, expected, atol=1.0e-6, rtol=0.0)