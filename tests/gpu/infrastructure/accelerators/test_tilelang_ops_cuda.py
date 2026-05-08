from __future__ import annotations

import importlib

import pytest
import torch
import taac2026.infrastructure.accelerators as tilelang_ops

from taac2026.infrastructure.accelerators import (
    embedding_bag_mean,
    flash_attention,
    resolved_embedding_bag_mean_backend,
    rms_norm,
)


flash_attention_ops = importlib.import_module("taac2026.infrastructure.accelerators.attention.flash_attention")

pytestmark = pytest.mark.gpu


def _causal_valid_mask(lengths: torch.Tensor, num_heads: int) -> torch.Tensor:
    token_count = int(lengths.shape[1]) if lengths.ndim == 2 else int(lengths.numel())
    padding_mask = lengths if lengths.ndim == 2 else torch.arange(token_count).unsqueeze(0) >= lengths.unsqueeze(1)
    causal = torch.ones(token_count, token_count, dtype=torch.bool, device=padding_mask.device).tril()
    key_valid = ~padding_mask
    mask = causal.unsqueeze(0) & key_valid.unsqueeze(1)
    query_invalid = padding_mask.unsqueeze(-1)
    fallback = torch.eye(token_count, dtype=torch.bool, device=padding_mask.device).unsqueeze(0)
    mask = torch.where(query_invalid, fallback, mask)
    return mask.unsqueeze(1).expand(padding_mask.shape[0], num_heads, token_count, token_count)


def _manual_flash_attention_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    attn_mask: torch.Tensor | None = None,
    dropout_mask: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
) -> torch.Tensor:
    scale = q.shape[-1] ** -0.5
    scores = torch.matmul(q.float(), k.float().transpose(-1, -2)) * scale
    if attn_mask is not None:
        scores = scores.masked_fill(~attn_mask, float("-inf"))
    elif is_causal:
        causal_mask = torch.ones(q.shape[2], k.shape[2], dtype=torch.bool, device=q.device).tril()
        scores = scores.masked_fill(~causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    if dropout_mask is not None and dropout_p > 0.0:
        probs = probs * dropout_mask.to(probs.dtype) / (1.0 - dropout_p)
    return torch.matmul(probs, v.float()).to(q.dtype)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for TileLang kernel validation")
def test_flash_attention_tilelang_matches_torch_forward_on_cuda() -> None:
    if not tilelang_ops.tilelang_available():
        pytest.skip("tilelang is not installed")

    tilelang_ops.clear_tilelang_kernel_cache()
    q = torch.randn(2, 4, 32, 16, dtype=torch.float16, device="cuda")
    k = torch.randn(2, 4, 32, 16, dtype=torch.float16, device="cuda")
    v = torch.randn(2, 4, 32, 16, dtype=torch.float16, device="cuda")

    output = flash_attention(q, k, v, backend="tilelang", is_causal=False)
    reference = flash_attention(q, k, v, backend="torch", is_causal=False)

    torch.testing.assert_close(output.float(), reference.float(), atol=1e-2, rtol=1e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for TileLang kernel validation")
def test_flash_attention_tilelang_matches_torch_forward_and_backward_on_cuda() -> None:
    if not tilelang_ops.tilelang_available():
        pytest.skip("tilelang is not installed")

    tilelang_ops.clear_tilelang_kernel_cache()
    q = torch.randn(2, 4, 32, 16, dtype=torch.float16, device="cuda", requires_grad=True)
    k = torch.randn(2, 4, 32, 16, dtype=torch.float16, device="cuda", requires_grad=True)
    v = torch.randn(2, 4, 32, 16, dtype=torch.float16, device="cuda", requires_grad=True)
    q_ref = q.detach().clone().requires_grad_(True)
    k_ref = k.detach().clone().requires_grad_(True)
    v_ref = v.detach().clone().requires_grad_(True)

    output = flash_attention(q, k, v, backend="tilelang", training=True, is_causal=False)
    reference = flash_attention(q_ref, k_ref, v_ref, backend="torch", training=True, is_causal=False)

    loss = output.float().square().mean()
    reference_loss = reference.float().square().mean()
    loss.backward()
    reference_loss.backward()

    torch.testing.assert_close(output.float(), reference.float(), atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(q.grad.float(), q_ref.grad.float(), atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(k.grad.float(), k_ref.grad.float(), atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(v.grad.float(), v_ref.grad.float(), atol=2e-2, rtol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for TileLang kernel validation")
def test_flash_attention_tilelang_matches_reference_with_training_dropout_on_cuda(monkeypatch) -> None:
    if not tilelang_ops.tilelang_available():
        pytest.skip("tilelang is not installed")

    tilelang_ops.clear_tilelang_kernel_cache()
    dropout_p = 0.125
    q = torch.randn(2, 4, 32, 16, dtype=torch.float16, device="cuda", requires_grad=True)
    k = torch.randn(2, 4, 32, 16, dtype=torch.float16, device="cuda", requires_grad=True)
    v = torch.randn(2, 4, 32, 16, dtype=torch.float16, device="cuda", requires_grad=True)
    q_ref = q.detach().clone().requires_grad_(True)
    k_ref = k.detach().clone().requires_grad_(True)
    v_ref = v.detach().clone().requires_grad_(True)
    fixed_dropout_mask = (torch.rand((2, 4, 32, 32), device="cuda") >= dropout_p).to(torch.uint8)

    monkeypatch.setattr(
        flash_attention_ops,
        "_build_tilelang_dropout_mask",
        lambda _q, _k, _dropout_p: fixed_dropout_mask,
    )

    output = flash_attention(q, k, v, backend="tilelang", dropout_p=dropout_p, training=True, is_causal=False)
    reference = _manual_flash_attention_reference(
        q_ref,
        k_ref,
        v_ref,
        dropout_mask=fixed_dropout_mask,
        dropout_p=dropout_p,
        is_causal=False,
    )

    grad_out = torch.randn_like(output)
    output.backward(grad_out)
    reference.backward(grad_out)

    torch.testing.assert_close(output.float(), reference.float(), atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(q.grad.float(), q_ref.grad.float(), atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(k.grad.float(), k_ref.grad.float(), atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(v.grad.float(), v_ref.grad.float(), atol=3e-2, rtol=3e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for TileLang kernel validation")
def test_flash_attention_tilelang_supports_float32_callers_on_cuda() -> None:
    if not tilelang_ops.tilelang_available():
        pytest.skip("tilelang is not installed")

    tilelang_ops.clear_tilelang_kernel_cache()
    q = torch.randn(2, 4, 32, 16, dtype=torch.float32, device="cuda")
    k = torch.randn(2, 4, 32, 16, dtype=torch.float32, device="cuda")
    v = torch.randn(2, 4, 32, 16, dtype=torch.float32, device="cuda")

    output = flash_attention(q, k, v, backend="tilelang", is_causal=False)
    reference = flash_attention(q, k, v, backend="torch", is_causal=False)

    assert output.dtype == torch.float32
    torch.testing.assert_close(output, reference, atol=1.5e-2, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for TileLang kernel validation")
def test_flash_attention_tilelang_matches_torch_forward_with_causal_valid_mask_on_cuda() -> None:
    if not tilelang_ops.tilelang_available():
        pytest.skip("tilelang is not installed")

    tilelang_ops.clear_tilelang_kernel_cache()
    q = torch.randn(2, 4, 32, 16, dtype=torch.float16, device="cuda")
    k = torch.randn(2, 4, 32, 16, dtype=torch.float16, device="cuda")
    v = torch.randn(2, 4, 32, 16, dtype=torch.float16, device="cuda")
    padding_mask = torch.tensor(
        [
            [False] * 20 + [True] * 12,
            [False] * 11 + [True] * 21,
        ],
        dtype=torch.bool,
        device="cuda",
    )
    attn_mask = _causal_valid_mask(padding_mask, q.shape[1])

    output = flash_attention(q, k, v, backend="tilelang", attn_mask=attn_mask)
    reference = flash_attention(q, k, v, backend="torch", attn_mask=attn_mask)

    torch.testing.assert_close(output.float(), reference.float(), atol=2e-2, rtol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for TileLang kernel validation")
def test_rms_norm_tilelang_matches_torch_forward_and_backward_on_cuda() -> None:
    if not tilelang_ops.tilelang_available():
        pytest.skip("tilelang is not installed")

    tilelang_ops.clear_tilelang_kernel_cache()
    x = torch.randn(16, 64, dtype=torch.float16, device="cuda", requires_grad=True)
    weight = torch.randn(64, dtype=torch.float16, device="cuda", requires_grad=True)
    x_ref = x.detach().clone().requires_grad_(True)
    weight_ref = weight.detach().clone().requires_grad_(True)

    output = rms_norm(x, weight, backend="tilelang")
    reference = rms_norm(x_ref, weight_ref, backend="torch")

    loss = output.float().square().mean()
    reference_loss = reference.float().square().mean()
    loss.backward()
    reference_loss.backward()

    torch.testing.assert_close(output.float(), reference.float(), atol=5e-3, rtol=5e-3)
    torch.testing.assert_close(x.grad.float(), x_ref.grad.float(), atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(weight.grad.float(), weight_ref.grad.float(), atol=1e-2, rtol=1e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for TileLang kernel validation")
def test_embedding_bag_mean_tilelang_matches_torch_forward_and_backward_on_cuda() -> None:
    if not tilelang_ops.tilelang_available():
        pytest.skip("tilelang is not installed")

    tilelang_ops.clear_tilelang_kernel_cache()
    weight = torch.randn(17, 16, dtype=torch.float16, device="cuda", requires_grad=True)
    values = torch.tensor(
        [
            [1, 2, 0, 4],
            [0, 0, 0, 0],
            [3, 3, 7, 0],
            [16, 5, 1, 0],
        ],
        dtype=torch.long,
        device="cuda",
    )
    weight_ref = weight.detach().clone().requires_grad_(True)

    output = embedding_bag_mean(weight, values, backend="tilelang", block_rows=1, block_cols=16)
    reference = embedding_bag_mean(weight_ref, values, backend="torch")

    loss = output.float().square().mean()
    reference_loss = reference.float().square().mean()
    loss.backward()
    reference_loss.backward()

    assert resolved_embedding_bag_mean_backend(weight.detach(), values, "tilelang") == "tilelang"
    torch.testing.assert_close(output.float(), reference.float(), atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(weight.grad.float(), weight_ref.grad.float(), atol=1e-3, rtol=1e-3)
