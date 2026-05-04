from __future__ import annotations

import pytest
import torch
import taac2026.infrastructure.pcvr.tilelang_ops as tilelang_ops

from taac2026.infrastructure.pcvr.tilelang_ops import resolved_rms_norm_backend, rms_norm


def test_rms_norm_matches_reference_on_cpu_torch_backend() -> None:
    x = torch.randn(8, 4, 16, dtype=torch.float32)
    weight = torch.randn(16, dtype=torch.float32)

    output = rms_norm(x, weight, backend="torch")
    reference = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + 1e-6) * weight

    torch.testing.assert_close(output, reference)


def test_resolved_rms_norm_backend_falls_back_to_torch_on_cpu() -> None:
    x = torch.randn(16, 64, dtype=torch.float32)

    assert resolved_rms_norm_backend(x, "torch") == "torch"


def test_resolved_rms_norm_backend_rejects_tilelang_on_cpu() -> None:
    x = torch.randn(16, 64, dtype=torch.float32)

    with pytest.raises(RuntimeError, match="requires CUDA tensors"):
        resolved_rms_norm_backend(x, "tilelang")


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


def test_rms_norm_tilelang_raises_when_tilelang_compile_fails(monkeypatch) -> None:
    x = torch.randn(4, 8, dtype=torch.float32)
    weight = torch.randn(8, dtype=torch.float32)

    monkeypatch.setattr(tilelang_ops, "resolved_rms_norm_backend", lambda *_args, **_kwargs: "tilelang")
    monkeypatch.setattr(tilelang_ops, "compile_rms_norm_kernel", lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("compile failed")))

    with pytest.raises(RuntimeError, match="compile failed"):
        rms_norm(x, weight, backend="tilelang")