from __future__ import annotations

import pytest
import torch

from experiments.onetrans.layers import RMSNorm as ExportedRMSNorm
from taac2026.infrastructure.modeling import (
    DenseTokenProjector,
    FeatureEmbeddingBank,
    NonSequentialTokenizer,
    RMSNorm,
    SequenceTokenizer,
    configure_rms_norm_runtime,
)
from taac2026.infrastructure.modeling.embeddings import FeatureEmbeddingBank as FeatureEmbeddingBankOwner
from taac2026.infrastructure.modeling.normalization import RMSNorm as RMSNormOwner
from taac2026.infrastructure.modeling.sequence import make_padding_mask
from taac2026.infrastructure.modeling.tokenizers import SequenceTokenizer as SequenceTokenizerOwner


def test_layers_modules_reexport_shared_primitives() -> None:
    assert ExportedRMSNorm is RMSNorm


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


def test_shared_tokenizers_return_stable_shapes() -> None:
    configure_rms_norm_runtime(backend="torch", block_rows=1)
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