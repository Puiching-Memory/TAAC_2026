from __future__ import annotations

import importlib

import pytest
import torch
import taac2026.infrastructure.accelerators as tilelang_ops
import torch.nn.functional as F

from taac2026.infrastructure.accelerators import (
    embedding_bag_mean,
    flash_attention,
    multi_latent_attention,
    resolved_embedding_bag_mean_backend,
    resolved_flash_attention_backend,
    resolved_rms_norm_backend,
    rms_norm,
)

flash_attention_ops = importlib.import_module("taac2026.infrastructure.accelerators.attention.flash_attention")
attention_tilelang_kernels = importlib.import_module("taac2026.infrastructure.accelerators.attention.kernels.tilelang")
embedding_bag_ops = importlib.import_module("taac2026.infrastructure.accelerators.embedding.embedding_bag")
embedding_tilelang_kernels = importlib.import_module("taac2026.infrastructure.accelerators.embedding.kernels.tilelang")
rms_norm_ops = importlib.import_module("taac2026.infrastructure.accelerators.normalization.rms_norm")
normalization_tilelang_kernels = importlib.import_module("taac2026.infrastructure.accelerators.normalization.kernels.tilelang")
tilelang_runtime = importlib.import_module("taac2026.infrastructure.accelerators.tilelang_runtime")


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


def _assert_error_mentions(exc_info: pytest.ExceptionInfo[BaseException], *parts: str) -> None:
    message = str(exc_info.value)
    for part in parts:
        assert part in message


def test_tilelang_runtime_does_not_export_domain_kernel_builders() -> None:
    assert "build_flash_attention_forward_kernel" not in tilelang_runtime.__all__
    assert "build_embedding_bag_mean_forward_kernel" not in tilelang_runtime.__all__
    assert "build_rms_norm_forward_kernel" not in tilelang_runtime.__all__
    assert not hasattr(tilelang_runtime, "build_flash_attention_forward_kernel")
    assert not hasattr(tilelang_runtime, "build_embedding_bag_mean_forward_kernel")
    assert not hasattr(tilelang_runtime, "build_rms_norm_forward_kernel")


def test_tilelang_runtime_exports_shared_capability_helpers() -> None:
    assert tilelang_runtime.cuda_multiprocessor_count() is None or isinstance(
        tilelang_runtime.cuda_multiprocessor_count(),
        int,
    )


def test_shared_tensor_validation_helpers_reject_cpu_for_cuda_ops() -> None:
    with pytest.raises(RuntimeError, match="sample_op requires CUDA tensors"):
        tilelang_ops.require_cuda_tensors("sample_op", torch.empty(1))


def test_shared_tensor_validation_helpers_reject_dtype_mismatch() -> None:
    with pytest.raises(ValueError, match="sample_op requires tensors to share the same dtype"):
        tilelang_ops.require_same_dtype("sample_op", torch.empty(1, dtype=torch.float32), torch.empty(1, dtype=torch.float16))


def test_tilelang_kernel_builders_are_domain_scoped() -> None:
    assert hasattr(attention_tilelang_kernels, "build_flash_attention_forward_kernel")
    assert hasattr(attention_tilelang_kernels, "build_flash_attention_training_forward_kernel")
    assert hasattr(attention_tilelang_kernels, "build_flash_attention_backward_kernel")
    assert hasattr(embedding_tilelang_kernels, "build_embedding_bag_mean_forward_kernel")
    assert hasattr(embedding_tilelang_kernels, "build_embedding_bag_mean_backward_kernel")
    assert hasattr(normalization_tilelang_kernels, "build_rms_norm_forward_kernel")
    assert hasattr(normalization_tilelang_kernels, "build_rms_norm_backward_kernel")
    assert attention_tilelang_kernels.build_flash_attention_forward_kernel.__module__ == attention_tilelang_kernels.__name__
    assert (
        embedding_tilelang_kernels.build_embedding_bag_mean_forward_kernel.__module__
        == embedding_tilelang_kernels.__name__
    )
    assert normalization_tilelang_kernels.build_rms_norm_forward_kernel.__module__ == normalization_tilelang_kernels.__name__


def test_embedding_bag_mean_matches_reference_on_cpu_default_torch_backend() -> None:
    weight = torch.randn(8, 5, dtype=torch.float32)
    values = torch.tensor(
        [
            [1, 2, 0, 4],
            [0, 0, 0, 0],
            [3, 3, 7, 0],
        ],
        dtype=torch.long,
    )

    output = embedding_bag_mean(weight, values)
    embedded = F.embedding(values, weight, padding_idx=0)
    valid = values.ne(0).to(embedded.dtype).unsqueeze(-1)
    reference = (embedded * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)

    assert resolved_embedding_bag_mean_backend(weight, values) == "torch"
    torch.testing.assert_close(output, reference)


def test_embedding_bag_mean_torch_backend_is_compile_friendly() -> None:
    weight = torch.randn(8, 5, dtype=torch.float32, requires_grad=True)
    values = torch.tensor(
        [
            [1, 2, 0, 4],
            [0, 0, 0, 0],
            [3, 3, 7, 0],
        ],
        dtype=torch.long,
    )

    def run(weight: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        return embedding_bag_mean(weight, values)

    compiled = torch.compile(run, backend="eager")
    output = compiled(weight, values)
    reference = embedding_bag_mean(weight, values)
    output.sum().backward()

    torch.testing.assert_close(output, reference)
    assert weight.grad is not None
    torch.testing.assert_close(weight.grad[0], torch.zeros_like(weight.grad[0]))


def test_embedding_bag_mean_torch_backend_matches_reference_backward() -> None:
    weight = torch.randn(8, 5, dtype=torch.float32, requires_grad=True)
    reference_weight = weight.detach().clone().requires_grad_(True)
    values = torch.tensor(
        [
            [1, 2, 0, 4],
            [0, 0, 0, 0],
            [3, 3, 7, 0],
        ],
        dtype=torch.long,
    )

    output = embedding_bag_mean(weight, values)
    embedded = F.embedding(values, reference_weight, padding_idx=0)
    valid = values.ne(0).to(embedded.dtype).unsqueeze(-1)
    reference = (embedded * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
    output.square().sum().backward()
    reference.square().sum().backward()

    torch.testing.assert_close(output, reference)
    torch.testing.assert_close(weight.grad, reference_weight.grad)


def test_embedding_bag_mean_default_preserves_registered_torch_kernel(monkeypatch) -> None:
    weight = torch.randn(8, 5, dtype=torch.float32)
    values = torch.ones(3, 4, dtype=torch.long)
    expected = torch.full((3, 5), 7.0)

    monkeypatch.setattr(embedding_bag_ops, "_embedding_bag_mean_kernel", lambda _weight, _values: expected)

    output = embedding_bag_mean(weight, values)

    torch.testing.assert_close(output, expected)


def test_embedding_bag_mean_tilelang_defaults_use_forward_row_block_only_for_inference() -> None:
    weight = torch.randn(8, 5, dtype=torch.float32)
    train_weight = weight.detach().clone().requires_grad_(True)
    values = torch.ones(3, 4, dtype=torch.long)

    inference_key = embedding_bag_ops._embedding_bag_mean_cache_key(weight, values, None, None)
    training_key = embedding_bag_ops._embedding_bag_mean_cache_key(train_weight, values, 1, None)

    assert inference_key.block_rows == 8
    assert training_key.block_rows == 1


def test_resolved_embedding_bag_mean_backend_rejects_unknown_backend() -> None:
    weight = torch.randn(8, 5, dtype=torch.float32)
    values = torch.ones(3, 4, dtype=torch.long)

    with pytest.raises(ValueError, match="unsupported embedding_bag_mean backend"):
        resolved_embedding_bag_mean_backend(weight, values, "invalid")  # type: ignore[arg-type]


def test_resolved_embedding_bag_mean_backend_rejects_tilelang_when_unavailable(monkeypatch) -> None:
    weight = torch.randn(8, 5, dtype=torch.float32)
    values = torch.ones(3, 4, dtype=torch.long)

    monkeypatch.setattr(embedding_bag_ops, "tilelang_available", lambda: False)

    with pytest.raises(RuntimeError) as exc_info:
        resolved_embedding_bag_mean_backend(weight, values, "tilelang")

    _assert_error_mentions(exc_info, "tilelang", "not installed")


def test_resolved_embedding_bag_mean_backend_rejects_tilelang_on_cpu(monkeypatch) -> None:
    weight = torch.randn(8, 5, dtype=torch.float32)
    values = torch.ones(3, 4, dtype=torch.long)

    monkeypatch.setattr(embedding_bag_ops, "tilelang_available", lambda: True)

    with pytest.raises(RuntimeError) as exc_info:
        resolved_embedding_bag_mean_backend(weight, values, "tilelang")

    _assert_error_mentions(exc_info, "requires CUDA tensors")


def test_resolved_embedding_bag_mean_backend_rejects_cuembed_when_unavailable(monkeypatch) -> None:
    weight = torch.randn(8, 5, dtype=torch.float32)
    values = torch.ones(3, 4, dtype=torch.long)

    monkeypatch.setattr(embedding_bag_ops, "cuembed_available", lambda: False)

    with pytest.raises(RuntimeError) as exc_info:
        resolved_embedding_bag_mean_backend(weight, values, "cuembed")

    _assert_error_mentions(exc_info, "cuembed", "CUDA", "not available")


def test_resolved_embedding_bag_mean_backend_rejects_cuembed_on_cpu(monkeypatch) -> None:
    weight = torch.randn(8, 5, dtype=torch.float32)
    values = torch.ones(3, 4, dtype=torch.long)

    monkeypatch.setattr(embedding_bag_ops, "cuembed_available", lambda: True)

    with pytest.raises(RuntimeError) as exc_info:
        resolved_embedding_bag_mean_backend(weight, values, "cuembed")

    _assert_error_mentions(exc_info, "requires CUDA tensors")


def test_resolved_embedding_bag_mean_backend_rejects_cuembed_training_path(monkeypatch) -> None:
	weight = torch.randn(8, 5, dtype=torch.float32, requires_grad=True)
	values = torch.ones(3, 4, dtype=torch.long)

	monkeypatch.setattr(embedding_bag_ops, "cuembed_available", lambda: True)
	monkeypatch.setattr(embedding_bag_ops, "require_cuda_tensors", lambda *_args, **_kwargs: None)

	with pytest.raises(RuntimeError, match="forward-only"):
		resolved_embedding_bag_mean_backend(weight, values, "cuembed")


def test_flash_attention_matches_reference_on_cpu_torch_backend() -> None:
    q = torch.randn(2, 3, 5, 8, dtype=torch.float32)
    k = torch.randn(2, 3, 7, 8, dtype=torch.float32)
    v = torch.randn(2, 3, 7, 8, dtype=torch.float32)

    output = flash_attention(q, k, v, backend="torch")
    reference = F.scaled_dot_product_attention(q, k, v)

    torch.testing.assert_close(output, reference)


def test_multi_latent_attention_matches_torch_attention_boundary() -> None:
    q = torch.randn(2, 3, 4, 8, dtype=torch.float32)
    latent_k = torch.randn(2, 3, 6, 8, dtype=torch.float32)
    latent_v = torch.randn(2, 3, 6, 8, dtype=torch.float32)

    output = multi_latent_attention(q, latent_k, latent_v, backend="torch")
    reference = flash_attention(q, latent_k, latent_v, backend="torch")

    torch.testing.assert_close(output, reference)


def test_resolved_flash_attention_backend_falls_back_to_torch_on_cpu() -> None:
    q = torch.randn(2, 3, 5, 8, dtype=torch.float32)
    k = torch.randn(2, 3, 5, 8, dtype=torch.float32)
    v = torch.randn(2, 3, 5, 8, dtype=torch.float32)

    assert resolved_flash_attention_backend(q, k, v, "torch") == "torch"


def test_resolved_flash_attention_backend_rejects_tilelang_when_unavailable(monkeypatch) -> None:
    q = torch.randn(2, 3, 5, 8, dtype=torch.float32)
    k = torch.randn(2, 3, 5, 8, dtype=torch.float32)
    v = torch.randn(2, 3, 5, 8, dtype=torch.float32)

    monkeypatch.setattr(flash_attention_ops, "tilelang_available", lambda: False)

    with pytest.raises(RuntimeError) as exc_info:
        resolved_flash_attention_backend(q, k, v, "tilelang")

    _assert_error_mentions(exc_info, "tilelang", "not installed")


def test_resolved_flash_attention_backend_rejects_tilelang_on_cpu(monkeypatch) -> None:
    q = torch.randn(2, 3, 5, 8, dtype=torch.float32)
    k = torch.randn(2, 3, 5, 8, dtype=torch.float32)
    v = torch.randn(2, 3, 5, 8, dtype=torch.float32)

    monkeypatch.setattr(flash_attention_ops, "tilelang_available", lambda: True)

    with pytest.raises(RuntimeError) as exc_info:
        resolved_flash_attention_backend(q, k, v, "tilelang")

    _assert_error_mentions(exc_info, "requires CUDA tensors")


def test_resolved_flash_attention_backend_rejects_unstructured_attention_mask() -> None:
    q = torch.randn(2, 3, 5, 8, dtype=torch.float32)
    k = torch.randn(2, 3, 5, 8, dtype=torch.float32)
    v = torch.randn(2, 3, 5, 8, dtype=torch.float32)
    attn_mask = torch.tensor(
        [
            [
                [
                    [True, False, True, False, False],
                    [True, False, True, False, False],
                    [True, False, True, False, False],
                    [True, False, True, False, False],
                    [True, False, True, False, False],
                ]
            ]
        ],
        dtype=torch.bool,
    ).expand(2, 3, 5, 5)

    with pytest.raises(RuntimeError, match="requires bool key masks to be prefix-valid"):
        resolved_flash_attention_backend(q, k, v, "tilelang", attn_mask=attn_mask)


def test_plan_tilelang_flash_attention_mask_extracts_prefix_key_lengths() -> None:
    q = torch.randn(2, 3, 4, 8, dtype=torch.float32)
    k = torch.randn(2, 3, 6, 8, dtype=torch.float32)
    key_valid = torch.tensor(
        [
            [True, True, True, True, False, False],
            [True, True, False, False, False, False],
        ],
        dtype=torch.bool,
    )
    attn_mask = key_valid[:, None, None, :].expand(2, 3, 4, 6)

    plan = tilelang_ops._plan_tilelang_flash_attention_mask(q, k, attn_mask, is_causal=False)

    assert torch.equal(plan.key_lengths.cpu(), torch.tensor([4, 2], dtype=torch.int32))
    assert plan.query_self_mask is None
    assert plan.is_causal is False


def test_plan_tilelang_flash_attention_mask_extracts_causal_valid_mask() -> None:
    q = torch.randn(2, 3, 5, 8, dtype=torch.float32)
    k = torch.randn(2, 3, 5, 8, dtype=torch.float32)
    padding_mask = torch.tensor(
        [
            [False, False, False, True, True],
            [False, True, True, True, True],
        ],
        dtype=torch.bool,
    )
    attn_mask = _causal_valid_mask(padding_mask, q.shape[1])

    plan = tilelang_ops._plan_tilelang_flash_attention_mask(q, k, attn_mask, is_causal=False)

    assert torch.equal(plan.key_lengths.cpu(), torch.tensor([3, 1], dtype=torch.int32))
    assert torch.equal(
        plan.query_self_mask.cpu(),
        torch.tensor(
            [
                [False, False, False, True, True],
                [False, True, True, True, True],
            ],
            dtype=torch.bool,
        ),
    )
    assert plan.is_causal is True


def test_resolved_flash_attention_backend_rejects_dropout_outside_training() -> None:
    q = torch.randn(2, 3, 5, 8, dtype=torch.float32)
    k = torch.randn(2, 3, 5, 8, dtype=torch.float32)
    v = torch.randn(2, 3, 5, 8, dtype=torch.float32)

    with pytest.raises(RuntimeError, match="dropout during training"):
        resolved_flash_attention_backend(q, k, v, "tilelang", dropout_p=0.1, training=False)


def test_tilelang_flash_attention_launch_config_specializes_h_series_training_defaults() -> None:
    q = torch.randn(2, 4, 256, 16, dtype=torch.float16)
    k = torch.randn(2, 4, 256, 16, dtype=torch.float16)

    assert flash_attention_ops._resolve_tilelang_flash_attention_launch_config(
        q,
        k,
        training=True,
        use_dropout=False,
        block_m=64,
        block_n=64,
        num_stages=1,
        threads=128,
    ) == (64, 128, 2, 128)


def test_tilelang_flash_attention_launch_config_specializes_h_series_training_dropout_defaults() -> None:
    q = torch.randn(2, 4, 256, 16, dtype=torch.float16)
    k = torch.randn(2, 4, 256, 16, dtype=torch.float16)

    assert flash_attention_ops._resolve_tilelang_flash_attention_launch_config(
        q,
        k,
        training=True,
        use_dropout=True,
        block_m=64,
        block_n=64,
        num_stages=1,
        threads=128,
    ) == (64, 128, 2, 128)


def test_tilelang_flash_attention_launch_config_respects_explicit_override() -> None:
    q = torch.randn(2, 4, 256, 16, dtype=torch.float16)
    k = torch.randn(2, 4, 256, 16, dtype=torch.float16)

    assert flash_attention_ops._resolve_tilelang_flash_attention_launch_config(
        q,
        k,
        training=True,
        use_dropout=False,
        block_m=128,
        block_n=64,
        num_stages=2,
        threads=128,
    ) == (128, 64, 2, 128)


def test_flash_attention_tilelang_raises_when_compile_fails(monkeypatch) -> None:
    q = torch.randn(2, 3, 5, 8, dtype=torch.float32)
    k = torch.randn(2, 3, 5, 8, dtype=torch.float32)
    v = torch.randn(2, 3, 5, 8, dtype=torch.float32)

    monkeypatch.setattr(flash_attention_ops, "_resolve_flash_attention_backend", lambda *_args, **_kwargs: "tilelang")
    monkeypatch.setattr(
        flash_attention_ops,
        "compile_flash_attention_kernel",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("compile failed")),
    )

    with pytest.raises(RuntimeError, match="compile failed"):
        flash_attention(q, k, v, backend="tilelang")


def test_flash_attention_tilelang_casts_float32_inputs_to_bfloat16(monkeypatch) -> None:
    q = torch.randn(2, 3, 5, 8, dtype=torch.float32)
    k = torch.randn(2, 3, 5, 8, dtype=torch.float32)
    v = torch.randn(2, 3, 5, 8, dtype=torch.float32)

    monkeypatch.setattr(flash_attention_ops, "_resolve_flash_attention_backend", lambda *_args, **_kwargs: "tilelang")

    def fake_compile(q_kernel: torch.Tensor, k_kernel: torch.Tensor, v_kernel: torch.Tensor, **_kwargs):
        assert q_kernel.dtype == torch.bfloat16
        assert k_kernel.dtype == torch.bfloat16
        assert v_kernel.dtype == torch.bfloat16

        def runner(q_runtime: torch.Tensor, k_runtime: torch.Tensor, v_runtime: torch.Tensor, _key_lengths: torch.Tensor) -> torch.Tensor:
            assert q_runtime.dtype == torch.bfloat16
            assert k_runtime.dtype == torch.bfloat16
            assert v_runtime.dtype == torch.bfloat16
            return q_runtime + k_runtime + v_runtime

        return runner

    monkeypatch.setattr(flash_attention_ops, "compile_flash_attention_kernel", fake_compile)

    output = flash_attention(q, k, v, backend="tilelang")

    assert output.dtype == torch.float32
    expected = (q.to(torch.bfloat16) + k.to(torch.bfloat16) + v.to(torch.bfloat16)).to(torch.float32)
    torch.testing.assert_close(output, expected)


def test_rms_norm_matches_reference_on_cpu_torch_backend() -> None:
    x = torch.randn(8, 4, 16, dtype=torch.float32)
    weight = torch.randn(16, dtype=torch.float32)

    output = rms_norm(x, weight, backend="torch")
    reference = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + 1e-6) * weight

    torch.testing.assert_close(output, reference)


def test_resolved_rms_norm_backend_falls_back_to_torch_on_cpu() -> None:
    x = torch.randn(16, 64, dtype=torch.float32)

    assert resolved_rms_norm_backend(x, "torch") == "torch"


def test_resolved_rms_norm_backend_rejects_tilelang_when_unavailable(monkeypatch) -> None:
    x = torch.randn(16, 64, dtype=torch.float32)

    monkeypatch.setattr(rms_norm_ops, "tilelang_available", lambda: False)

    with pytest.raises(RuntimeError) as exc_info:
        resolved_rms_norm_backend(x, "tilelang")

    _assert_error_mentions(exc_info, "tilelang", "not installed")


def test_resolved_rms_norm_backend_rejects_tilelang_on_cpu(monkeypatch) -> None:
    x = torch.randn(16, 64, dtype=torch.float32)

    monkeypatch.setattr(rms_norm_ops, "tilelang_available", lambda: True)

    with pytest.raises(RuntimeError) as exc_info:
        resolved_rms_norm_backend(x, "tilelang")

    _assert_error_mentions(exc_info, "requires CUDA tensors")


def test_ensure_tilelang_cuda_fp8_compatibility_patches_guard_when_cuda_lacks_e8m0(tmp_path) -> None:
    tilelang_header = tmp_path / "cuda_fp8.h"
    tilelang_header.write_text(
        "prefix\n" + tilelang_ops._TILELANG_E8M0_ORIGINAL_GUARD + "suffix\n",
        encoding="utf-8",
    )
    cuda_header = tmp_path / "system_cuda_fp8.h"
    cuda_header.write_text("// no e8m0 symbols here\n", encoding="utf-8")

    patched = tilelang_ops._ensure_tilelang_cuda_fp8_compatibility(
        tilelang_header=tilelang_header,
        cuda_header_paths=(cuda_header,),
    )

    assert patched is True
    content = tilelang_header.read_text(encoding="utf-8")
    assert tilelang_ops._TILELANG_E8M0_COMPAT_GUARD in content
    assert tilelang_ops._TILELANG_E8M0_ORIGINAL_GUARD not in content


def test_ensure_tilelang_cuda_fp8_compatibility_patches_installed_header_format(tmp_path) -> None:
    tilelang_header = tmp_path / "cuda_fp8.h"
    tilelang_header.write_text(
        """#pragma once

#include \"common.h\"

// __nv_fp8_e8m0 is only available in CUDA 12.6+
#if __CUDACC_VER_MAJOR__ > 12 ||                                               \\
    (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ >= 6)
using fp8_e8_t = __nv_fp8_e8m0;
#define TL_HAS_FP8_E8M0 1
#else
// Placeholder for CUDA < 12.6
struct fp8_e8_t {
  unsigned char data;
};
#define TL_HAS_FP8_E8M0 0
#endif

struct trailing_type {};
""",
        encoding="utf-8",
    )
    cuda_header = tmp_path / "system_cuda_fp8.h"
    cuda_header.write_text("// no e8m0 symbols here\n", encoding="utf-8")

    patched = tilelang_ops._ensure_tilelang_cuda_fp8_compatibility(
        tilelang_header=tilelang_header,
        cuda_header_paths=(cuda_header,),
    )

    assert patched is True
    content = tilelang_header.read_text(encoding="utf-8")
    assert tilelang_ops._TILELANG_E8M0_COMPAT_GUARD in content
    assert "struct trailing_type {};" in content


def test_ensure_tilelang_cuda_fp8_compatibility_skips_patch_when_cuda_has_e8m0(tmp_path) -> None:
    tilelang_header = tmp_path / "cuda_fp8.h"
    tilelang_header.write_text(tilelang_ops._TILELANG_E8M0_ORIGINAL_GUARD, encoding="utf-8")
    cuda_header = tmp_path / "system_cuda_fp8.h"
    cuda_header.write_text("using fp8_e8_t = __nv_fp8_e8m0;\n", encoding="utf-8")

    patched = tilelang_ops._ensure_tilelang_cuda_fp8_compatibility(
        tilelang_header=tilelang_header,
        cuda_header_paths=(cuda_header,),
    )

    assert patched is False
    assert tilelang_header.read_text(encoding="utf-8") == tilelang_ops._TILELANG_E8M0_ORIGINAL_GUARD


def test_embedding_bag_mean_tilelang_raises_when_tilelang_compile_fails(monkeypatch) -> None:
    weight = torch.randn(8, 5, dtype=torch.float32)
    values = torch.ones(3, 4, dtype=torch.long)

    monkeypatch.setattr(embedding_bag_ops, "_resolve_embedding_bag_mean_backend", lambda *_args, **_kwargs: "tilelang")
    monkeypatch.setattr(
        embedding_bag_ops,
        "compile_embedding_bag_mean_kernel",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("compile failed")),
    )

    with pytest.raises(RuntimeError, match="compile failed"):
        embedding_bag_mean(weight, values, backend="tilelang")


def test_rms_norm_tilelang_raises_when_tilelang_compile_fails(monkeypatch) -> None:
    x = torch.randn(4, 8, dtype=torch.float32)
    weight = torch.randn(8, dtype=torch.float32)

    monkeypatch.setattr(rms_norm_ops, "resolved_rms_norm_backend", lambda *_args, **_kwargs: "tilelang")
    monkeypatch.setattr(
        rms_norm_ops,
        "compile_rms_norm_kernel",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("compile failed")),
    )

    with pytest.raises(RuntimeError, match="compile failed"):
        rms_norm(x, weight, backend="tilelang")
