"""Shared TileLang runtime discovery and compatibility helpers."""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path

import torch

try:
    import tilelang as tl  # type: ignore[import-not-found]
    import tilelang.language as T  # type: ignore[import-not-found]
except ImportError:
    tl = None
    T = None

_TILELANG_E8M0_ORIGINAL_GUARD = """// __nv_fp8_e8m0 is only available in CUDA 12.6+
#if __CUDACC_VER_MAJOR__ > 12 ||                                               \
        (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ >= 6)
using fp8_e8_t = __nv_fp8_e8m0;
#define TL_HAS_FP8_E8M0 1
#else
// Fallback struct for CUDA < 12.6
struct fp8_e8_t {
    unsigned char data;
};
#define TL_HAS_FP8_E8M0 0
#endif
"""

_TILELANG_E8M0_COMPAT_GUARD = """// Some CUDA 12.6 toolchains still ship without e8m0 symbols in cuda_fp8.h.
// Keep e8m0 disabled unless the toolchain is known to provide those APIs.
#if 0
using fp8_e8_t = __nv_fp8_e8m0;
#define TL_HAS_FP8_E8M0 1
#else
// Fallback struct for CUDA < 12.6
struct fp8_e8_t {
    unsigned char data;
};
#define TL_HAS_FP8_E8M0 0
#endif
"""

_TILELANG_E8M0_GUARD_START = "// __nv_fp8_e8m0 is only available in CUDA 12.6+"


def tilelang_available() -> bool:
    return tl is not None and T is not None


def _tilelang_e8m0_guard_bounds(content: str) -> tuple[int, int] | None:
    start = content.find(_TILELANG_E8M0_GUARD_START)
    if start < 0:
        return None
    end = content.find("#endif", start)
    if end < 0:
        return None
    end += len("#endif")
    block = content[start:end]
    if "using fp8_e8_t = __nv_fp8_e8m0;" not in block:
        return None
    if "#define TL_HAS_FP8_E8M0 0" not in block:
        return None
    if end < len(content) and content[end] == "\n":
        end += 1
    return start, end


def _tilelang_cuda_fp8_header_path() -> Path | None:
    if tl is None:
        return None
    package_root = Path(tl.__file__).resolve().parent
    header_path = package_root / "src" / "tl_templates" / "cuda" / "cuda_fp8.h"
    return header_path if header_path.is_file() else None


def _cuda_fp8_header_candidates() -> tuple[Path, ...]:
    include_dirs: list[Path] = []
    for env_name in ("CUDA_HOME", "CUDA_PATH"):
        env_value = os.environ.get(env_name)
        if env_value:
            include_dirs.append(Path(env_value) / "include")
    include_dirs.append(Path("/usr/local/cuda/include"))
    include_dirs.extend(sorted(Path("/usr/local").glob("cuda-*/include")))

    headers: list[Path] = []
    for include_dir in include_dirs:
        headers.append(include_dir / "cuda_fp8.h")
        headers.append(include_dir / "cuda_fp8.hpp")
    return tuple(dict.fromkeys(headers))


def _cuda_fp8_supports_e8m0(header_paths: Sequence[Path] | None = None) -> bool:
    candidates = tuple(header_paths) if header_paths is not None else _cuda_fp8_header_candidates()
    for header_path in candidates:
        if not header_path.is_file():
            continue
        try:
            content = header_path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if "__nv_fp8_e8m0" in content:
            return True
    return False


def _ensure_tilelang_cuda_fp8_compatibility(
    *,
    tilelang_header: Path | None = None,
    cuda_header_paths: Sequence[Path] | None = None,
) -> bool:
    header_path = tilelang_header or _tilelang_cuda_fp8_header_path()
    if header_path is None or _cuda_fp8_supports_e8m0(cuda_header_paths):
        return False
    try:
        content = header_path.read_text(encoding="utf-8")
    except OSError as error:
        raise RuntimeError(f"failed to read tilelang fp8 header: {header_path}") from error
    if _TILELANG_E8M0_COMPAT_GUARD in content:
        return False
    guard_bounds = _tilelang_e8m0_guard_bounds(content)
    if guard_bounds is None:
        return False
    start, end = guard_bounds
    try:
        header_path.write_text(content[:start] + _TILELANG_E8M0_COMPAT_GUARD + content[end:], encoding="utf-8")
    except OSError as error:
        raise RuntimeError(f"failed to patch tilelang fp8 header: {header_path}") from error
    return True


def tilelang_dtype(dtype: torch.dtype):
    if T is None:
        raise RuntimeError("tilelang language module is unavailable")
    if dtype == torch.float16:
        return T.float16
    if dtype == torch.bfloat16:
        return T.bfloat16
    if dtype == torch.float32:
        return T.float32
    raise RuntimeError(f"tilelang kernels do not support dtype {dtype}")


__all__ = [
    "T",
    "tilelang_available",
    "tilelang_dtype",
    "tl",
]
