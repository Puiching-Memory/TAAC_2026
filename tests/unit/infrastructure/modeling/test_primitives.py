from __future__ import annotations

import pytest
import torch

from taac2026.infrastructure.modeling import (
    DenseTokenProjector,
    FeatureEmbeddingBank,
    NonSequentialTokenizer,
    RMSNorm,
    SequenceTokenizer,
    configure_flash_attention_runtime,
    configure_rms_norm_runtime,
    flash_attention_runtime_state,
    rms_norm_runtime_state,
    scaled_dot_product_attention,
)
from taac2026.infrastructure.modeling.embeddings import FeatureEmbeddingBank as FeatureEmbeddingBankOwner
from taac2026.infrastructure.modeling.normalization import RMSNorm as RMSNormOwner
from taac2026.infrastructure.modeling import sequence as sequence_ops
from taac2026.infrastructure.modeling.sequence import make_padding_mask
from taac2026.infrastructure.modeling.tokenizers import SequenceTokenizer as SequenceTokenizerOwner


def test_modeling_submodules_own_shared_primitives() -> None:
    assert FeatureEmbeddingBankOwner is FeatureEmbeddingBank
    assert RMSNormOwner is RMSNorm
    assert SequenceTokenizerOwner is SequenceTokenizer
    assert make_padding_mask(torch.tensor([1, 3]), 4).tolist() == [
        [False, True, True, True],
        [False, False, False, True],
    ]


def test_configure_rms_norm_runtime_validates_backend_and_block_rows() -> None:
    with pytest.raises(ValueError, match="unsupported rms_norm backend: nope"):
        configure_rms_norm_runtime(backend="nope", block_rows=1)
    with pytest.raises(ValueError, match="rms_norm block_rows must be positive"):
        configure_rms_norm_runtime(backend="torch", block_rows=0)
    configure_rms_norm_runtime(backend="triton", block_rows=2)
    assert rms_norm_runtime_state() == ("triton", 2)
    configure_rms_norm_runtime(backend="torch", block_rows=1)


def test_rms_norm_captures_runtime_state_at_construction() -> None:
    configure_rms_norm_runtime(backend="triton", block_rows=4)
    norm = RMSNorm(6)
    configure_rms_norm_runtime(backend="torch", block_rows=1)

    assert norm.backend == "triton"
    assert norm.block_rows == 4


def test_configure_flash_attention_runtime_validates_backend() -> None:
    with pytest.raises(ValueError, match="unsupported flash attention backend: nope"):
        configure_flash_attention_runtime(backend="nope")

    configure_flash_attention_runtime(backend="tilelang")
    assert flash_attention_runtime_state() == "tilelang"
    configure_flash_attention_runtime(backend="torch")


def test_scaled_dot_product_attention_uses_configured_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_flash_attention(q, k, v, **kwargs):
        del k, v
        captured.update(kwargs)
        return torch.zeros_like(q)

    monkeypatch.setattr(sequence_ops, "flash_attention", fake_flash_attention)
    configure_flash_attention_runtime(backend="tilelang")
    try:
        output = scaled_dot_product_attention(
            torch.ones(2, 3, 4),
            torch.ones(2, 3, 4),
            torch.ones(2, 3, 4),
            num_heads=2,
            attn_mask=None,
            dropout_p=0.0,
            training=False,
        )
    finally:
        configure_flash_attention_runtime(backend="torch")

    assert captured["backend"] == "tilelang"
    assert output.shape == (2, 3, 4)


def test_shared_tokenizers_return_stable_shapes() -> None:
    configure_rms_norm_runtime(backend="torch", block_rows=1)
    configure_flash_attention_runtime(backend="torch")
    int_feats = torch.tensor([[1, 2, 3], [0, 4, 5]])
    bank = FeatureEmbeddingBank([(10, 0, 1), (8, 1, 2)], emb_dim=4)
    non_seq = NonSequentialTokenizer([(10, 0, 1), (8, 1, 2)], [[0], [1]], emb_dim=4, d_model=6)
    dense = DenseTokenProjector(input_dim=3, d_model=6)
    sequence = SequenceTokenizer([10, 8], emb_dim=4, d_model=6, num_time_buckets=4)
    norm = RMSNorm(6)

    assert bank(int_feats).shape == (2, 2, 4)
    assert non_seq(int_feats).shape == (2, 2, 6)
    assert dense(torch.ones(2, 3)).shape == (2, 1, 6)
    assert sequence(torch.ones(2, 2, 5), torch.ones(2, 5)).shape == (2, 5, 6)
    assert norm(torch.ones(2, 5, 6)).shape == (2, 5, 6)
