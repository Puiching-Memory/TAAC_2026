from __future__ import annotations

import torch

from taac2026.infrastructure.nn.hstu import BranchTransducer, HSTUBlock, MixtureOfTransducers, ULTRAHSTUBlock, build_causal_mask
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