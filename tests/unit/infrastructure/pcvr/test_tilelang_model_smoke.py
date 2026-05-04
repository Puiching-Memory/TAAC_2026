from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

ctr_layers = importlib.import_module("experiments.pcvr.ctr_baseline.layers")
ctr_model = importlib.import_module("experiments.pcvr.ctr_baseline.model")
tilelang_ops = importlib.import_module("taac2026.infrastructure.pcvr.tilelang_ops")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for TileLang model smoke validation")
def test_ctr_baseline_model_trains_with_explicit_tilelang_rms_norm() -> None:
    if not tilelang_ops.tilelang_available():
        pytest.skip("tilelang is not installed")

    tilelang_ops.clear_tilelang_kernel_cache()
    old_backend = ctr_layers.RMS_NORM_BACKEND
    old_block_rows = ctr_layers.RMS_NORM_BLOCK_ROWS
    ctr_model.configure_rms_norm_runtime(
        rms_norm_backend="tilelang",
        rms_norm_block_rows=1,
    )
    try:
        model = ctr_model.PCVRCTRBaseline(
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
        ).to("cuda")
        model_input = ctr_model.ModelInput(
            user_int_feats=torch.tensor([[1, 2, 3], [4, 0, 1]], dtype=torch.long, device="cuda"),
            item_int_feats=torch.tensor([[1], [2]], dtype=torch.long, device="cuda"),
            user_dense_feats=torch.randn(2, 2, device="cuda"),
            item_dense_feats=torch.randn(2, 1, device="cuda"),
            seq_data={
                "seq_a": torch.tensor(
                    [
                        [[1, 2, 0, 0], [2, 3, 0, 0]],
                        [[4, 1, 2, 3], [1, 2, 3, 4]],
                    ],
                    dtype=torch.long,
                    device="cuda",
                ),
                "seq_b": torch.tensor([[[1, 0, 0]], [[2, 3, 0]]], dtype=torch.long, device="cuda"),
            },
            seq_lens={
                "seq_a": torch.tensor([2, 4], dtype=torch.long, device="cuda"),
                "seq_b": torch.tensor([1, 2], dtype=torch.long, device="cuda"),
            },
            seq_time_buckets={
                "seq_a": torch.zeros(2, 4, dtype=torch.long, device="cuda"),
                "seq_b": torch.zeros(2, 3, dtype=torch.long, device="cuda"),
            },
        )

        logits = model(model_input)
        loss = logits.sum()
        loss.backward()

        assert logits.shape == (2, 1)
        assert torch.isfinite(logits).all()
        assert any(parameter.grad is not None for parameter in model.parameters())
    finally:
        ctr_model.configure_rms_norm_runtime(
            rms_norm_backend=old_backend,
            rms_norm_block_rows=old_block_rows,
        )