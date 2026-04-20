from __future__ import annotations

import pytest
import torch
from torch import nn

from taac2026.infrastructure.nn import te_backend as te_backend_module
from taac2026.infrastructure.nn import transformer as transformer_module
from taac2026.infrastructure.nn.gpu_capability import detect_precision_support
from taac2026.infrastructure.nn.hstu import BranchTransducer, HSTUBlock, MixtureOfTransducers, ULTRAHSTUBlock, build_causal_mask
from taac2026.infrastructure.nn.te_backend import adapt_mask_for_te
from taac2026.infrastructure.nn.transformer import TaacCrossAttentionBlock, TaacMixedCausalBlock, TaacTransformerBlock


def test_taac_transformer_block_zeroes_masked_positions() -> None:
    block = TaacTransformerBlock(
        hidden_dim=8,
        num_heads=2,
        ffn_dim=16,
        dropout=0.0,
        norm_type="layernorm",
        ffn_type="gelu",
    )
    hidden_states = torch.randn(2, 4, 8)
    token_mask = torch.tensor(
        [
            [True, True, False, False],
            [True, True, True, False],
        ]
    )

    output = block(hidden_states, token_mask)

    assert output.shape == hidden_states.shape
    assert torch.allclose(output[0, 2:], torch.zeros_like(output[0, 2:]), atol=1.0e-6, rtol=0.0)
    assert torch.allclose(output[1, 3:], torch.zeros_like(output[1, 3:]), atol=1.0e-6, rtol=0.0)


def test_taac_cross_attention_block_zeroes_masked_queries() -> None:
    block = TaacCrossAttentionBlock(
        hidden_dim=8,
        num_heads=2,
        ffn_dim=16,
        dropout=0.0,
        norm_type="layernorm",
        ffn_type="silu",
    )
    query_states = torch.randn(2, 3, 8)
    context_states = torch.randn(2, 5, 8)
    query_mask = torch.tensor([[True, True, False], [True, False, False]])
    context_mask = torch.tensor([[True, True, True, False, False], [True, True, False, False, False]])

    output = block(query_states, context_states, query_mask=query_mask, context_mask=context_mask)

    assert output.shape == query_states.shape
    assert torch.allclose(output[0, 2], torch.zeros_like(output[0, 2]), atol=1.0e-6, rtol=0.0)
    assert torch.allclose(output[1, 1:], torch.zeros_like(output[1, 1:]), atol=1.0e-6, rtol=0.0)


def test_hstu_block_accepts_causal_mask() -> None:
    block = HSTUBlock(hidden_dim=8, num_heads=2, ffn_dim=16, dropout=0.0)
    hidden_states = torch.randn(2, 4, 8)
    attn_mask = build_causal_mask(4, torch.device("cpu"))

    output = block(hidden_states, attn_mask)

    assert output.shape == hidden_states.shape
    assert torch.isfinite(output).all().item()


def test_taac_mixed_causal_block_truncates_sequence_and_zeroes_masked_tokens() -> None:
    block = TaacMixedCausalBlock(
        hidden_dim=8,
        num_heads=2,
        ffn_dim=16,
        ns_token_count=2,
        dropout=0.0,
        attention_dropout=0.0,
    )
    sequence_tokens = torch.randn(2, 6, 8)
    sequence_mask = torch.tensor(
        [
            [True, True, True, True, False, False],
            [True, True, True, True, True, True],
        ]
    )
    ns_tokens = torch.randn(2, 2, 8)
    ns_mask = torch.tensor([[True, False], [True, True]])

    next_sequence, next_mask, next_ns, returned_ns_mask = block(
        sequence_tokens,
        sequence_mask,
        ns_tokens,
        ns_mask,
        next_sequence_length=3,
    )

    assert next_sequence.shape == (2, 3, 8)
    assert next_mask.shape == (2, 3)
    assert next_ns.shape == (2, 2, 8)
    assert returned_ns_mask.shape == (2, 2)
    assert torch.allclose(next_ns[0, 1], torch.zeros_like(next_ns[0, 1]), atol=1.0e-6, rtol=0.0)


def test_ultra_hstu_block_accepts_causal_mask() -> None:
    block = ULTRAHSTUBlock(hidden_dim=8, num_heads=2, ffn_dim=16, dropout=0.0, local_window=2)
    hidden_states = torch.randn(2, 5, 8)
    attn_mask = build_causal_mask(5, torch.device("cpu"))

    output = block(hidden_states, attn_mask)

    assert output.shape == hidden_states.shape
    assert torch.isfinite(output).all().item()


def test_branch_transducer_returns_last_valid_token() -> None:
    encoder = BranchTransducer(dim=8, num_heads=2, num_layers=2, dropout=0.0)
    hidden_states = torch.randn(2, 4, 8)
    token_mask = torch.tensor(
        [
            [True, True, True, False],
            [True, True, False, False],
        ]
    )

    output = encoder(hidden_states, token_mask)

    assert output.shape == (2, 8)
    assert torch.isfinite(output).all().item()


def test_mixture_of_transducers_fuses_branch_outputs() -> None:
    encoder = MixtureOfTransducers(dim=8, num_heads=2, num_layers=1, dropout=0.0, num_branches=3)
    branch_tokens = [torch.randn(2, 3, 8) for _ in range(3)]
    branch_masks = [torch.tensor([[True, True, False], [True, False, False]]) for _ in range(3)]

    output = encoder(branch_tokens, branch_masks)

    assert output.shape == (2, 8)
    assert torch.isfinite(output).all().item()


def test_detect_precision_support_reports_no_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    support = detect_precision_support()

    assert support["compute_capability"] is None
    assert support["recommended_precision"] is None
    assert support["supported_precisions"] == []


def test_detect_precision_support_prefers_blackwell_nvfp4(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (10, 0))

    support = detect_precision_support(torch.device("cuda:0"))

    assert support["architecture"] == "blackwell"
    assert support["supported_precisions"] == ["nvfp4", "mxfp8", "fp8", "bf16", "fp16"]
    assert support["recommended_precision"] == "nvfp4"
    assert support["recommended_recipe"] == "nvfp4_block_scaling"


def test_adapt_mask_for_te_inverts_keep_mask() -> None:
    mask = torch.tensor([[True, False, True], [False, False, True]])

    adapted = adapt_mask_for_te(mask)

    assert torch.equal(adapted, torch.tensor([[False, True, False], [True, True, False]]))


def test_resolve_te_attention_mask_uses_no_mask_fast_path() -> None:
    te_mask, attn_mask_type = te_backend_module._resolve_te_attention_mask(
        query_length=4,
        key_length=4,
        batch_size=2,
        device=torch.device("cpu"),
        attention_mask=None,
        query_mask=None,
        key_mask=None,
        is_causal=False,
    )

    assert te_mask is None
    assert attn_mask_type == "no_mask"


def test_resolve_te_attention_mask_uses_causal_fast_path() -> None:
    te_mask, attn_mask_type = te_backend_module._resolve_te_attention_mask(
        query_length=4,
        key_length=4,
        batch_size=2,
        device=torch.device("cpu"),
        attention_mask=None,
        query_mask=None,
        key_mask=None,
        is_causal=True,
    )

    assert te_mask is None
    assert attn_mask_type == "causal"


def test_resolve_te_attention_mask_keeps_arbitrary_path_for_padding_masks() -> None:
    te_mask, attn_mask_type = te_backend_module._resolve_te_attention_mask(
        query_length=3,
        key_length=3,
        batch_size=1,
        device=torch.device("cpu"),
        attention_mask=None,
        query_mask=torch.tensor([[True, True, False]]),
        key_mask=torch.tensor([[True, True, False]]),
        is_causal=False,
    )

    assert attn_mask_type == "arbitrary"
    assert torch.equal(
        te_mask,
        torch.tensor(
            [
                [
                    [
                        [False, False, True],
                        [False, False, True],
                        [True, True, True],
                    ]
                ]
            ]
        ),
    )


def test_taac_transformer_block_routes_te_backends(monkeypatch: pytest.MonkeyPatch) -> None:
    entered_context_count = 0

    class FakeContext:
        def __enter__(self):
            nonlocal entered_context_count
            entered_context_count += 1
            return self

        def __exit__(self, exc_type, exc, tb) -> bool:
            return False

    class FakeAttention(nn.Module):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__()
            self.init_kwargs = kwargs
            self.last_query_mask = None
            self.last_key_mask = None
            self.last_is_causal = None
            self.last_use_external_context = False

        def forward(
            self,
            query_states: torch.Tensor,
            key_states: torch.Tensor,
            value_states: torch.Tensor,
            *,
            attention_mask: torch.Tensor | None = None,
            query_mask: torch.Tensor | None = None,
            key_mask: torch.Tensor | None = None,
            additive_bias: torch.Tensor | None = None,
            is_causal: bool = False,
            use_external_context: bool = False,
        ) -> torch.Tensor:
            del attention_mask, additive_bias
            self.last_query_mask = query_mask
            self.last_key_mask = key_mask
            self.last_is_causal = is_causal
            self.last_use_external_context = use_external_context
            return query_states + key_states + value_states

    class FakeFeedForward(nn.Module):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__()
            self.last_input = None
            self.last_use_external_context = False

        def forward(self, hidden_states: torch.Tensor, *, use_external_context: bool = False) -> torch.Tensor:
            self.last_input = hidden_states
            self.last_use_external_context = use_external_context
            return hidden_states

    monkeypatch.setattr(transformer_module, "TransformerEngineAttention", FakeAttention)
    monkeypatch.setattr(transformer_module, "TransformerEngineFeedForward", FakeFeedForward)
    monkeypatch.setattr(transformer_module, "build_transformer_engine_context", lambda *args, **kwargs: FakeContext())

    block = TaacTransformerBlock(
        hidden_dim=8,
        num_heads=2,
        ffn_dim=16,
        dropout=0.0,
        norm_type="layernorm",
        ffn_type="silu",
        attention_backend="te",
        ffn_backend="te",
    )
    hidden_states = torch.randn(2, 4, 8)
    token_mask = torch.tensor(
        [
            [True, True, False, False],
            [True, True, True, False],
        ]
    )

    output = block(hidden_states, token_mask)

    assert isinstance(block.self_attention, FakeAttention)
    assert isinstance(block.feed_forward, FakeFeedForward)
    assert block.self_attention.last_query_mask is token_mask
    assert block.self_attention.last_key_mask is token_mask
    assert block.self_attention.last_is_causal is False
    assert block.self_attention.last_use_external_context is True
    assert block.feed_forward.last_input is not None
    assert block.feed_forward.last_use_external_context is True
    assert entered_context_count == 1
    assert output.shape == hidden_states.shape
    assert torch.allclose(output[0, 2:], torch.zeros_like(output[0, 2:]), atol=1.0e-6, rtol=0.0)
    assert torch.allclose(output[1, 3:], torch.zeros_like(output[1, 3:]), atol=1.0e-6, rtol=0.0)


def test_taac_cross_attention_block_routes_te_backends(monkeypatch: pytest.MonkeyPatch) -> None:
    entered_context_count = 0

    class FakeContext:
        def __enter__(self):
            nonlocal entered_context_count
            entered_context_count += 1
            return self

        def __exit__(self, exc_type, exc, tb) -> bool:
            return False

    class FakeAttention(nn.Module):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__()
            self.init_kwargs = kwargs
            self.last_query_mask = None
            self.last_key_mask = None
            self.last_use_external_context = False

        def forward(
            self,
            query_states: torch.Tensor,
            key_states: torch.Tensor,
            value_states: torch.Tensor,
            *,
            attention_mask: torch.Tensor | None = None,
            query_mask: torch.Tensor | None = None,
            key_mask: torch.Tensor | None = None,
            additive_bias: torch.Tensor | None = None,
            is_causal: bool = False,
            use_external_context: bool = False,
        ) -> torch.Tensor:
            del attention_mask, additive_bias, is_causal
            self.last_query_mask = query_mask
            self.last_key_mask = key_mask
            self.last_use_external_context = use_external_context
            del key_states, value_states
            return query_states

    class FakeFeedForward(nn.Module):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__()
            self.last_use_external_context = False

        def forward(self, hidden_states: torch.Tensor, *, use_external_context: bool = False) -> torch.Tensor:
            self.last_use_external_context = use_external_context
            return hidden_states

    monkeypatch.setattr(transformer_module, "TransformerEngineAttention", FakeAttention)
    monkeypatch.setattr(transformer_module, "TransformerEngineFeedForward", FakeFeedForward)
    monkeypatch.setattr(transformer_module, "build_transformer_engine_context", lambda *args, **kwargs: FakeContext())

    block = TaacCrossAttentionBlock(
        hidden_dim=8,
        num_heads=2,
        ffn_dim=16,
        dropout=0.0,
        norm_type="layernorm",
        ffn_type="gelu",
        attention_backend="te",
        ffn_backend="te",
    )
    query_states = torch.randn(2, 3, 8)
    context_states = torch.randn(2, 5, 8)
    query_mask = torch.tensor([[True, True, False], [True, False, False]])
    context_mask = torch.tensor([[True, True, True, False, False], [True, True, False, False, False]])

    output = block(query_states, context_states, query_mask=query_mask, context_mask=context_mask)

    assert isinstance(block.cross_attention, FakeAttention)
    assert block.cross_attention.last_query_mask is query_mask
    assert block.cross_attention.last_key_mask is context_mask
    assert block.cross_attention.last_use_external_context is True
    assert block.feed_forward.last_use_external_context is True
    assert entered_context_count == 1
    assert output.shape == query_states.shape
    assert torch.allclose(output[0, 2], torch.zeros_like(output[0, 2]), atol=1.0e-6, rtol=0.0)
    assert torch.allclose(output[1, 1:], torch.zeros_like(output[1, 1:]), atol=1.0e-6, rtol=0.0)


def test_taac_mixed_causal_block_rejects_te_backend() -> None:
    with pytest.raises(ValueError, match="does not support"):
        TaacMixedCausalBlock(
            hidden_dim=8,
            num_heads=2,
            ffn_dim=16,
            ns_token_count=2,
            dropout=0.0,
            attention_dropout=0.0,
            attention_backend="te",
        )