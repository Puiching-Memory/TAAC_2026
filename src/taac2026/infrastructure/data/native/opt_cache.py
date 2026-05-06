"""JIT loader for the native PCVR OPT cache index."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from types import ModuleType

from torch.utils.cpp_extension import load


@lru_cache(maxsize=1)
def load_native_opt_cache() -> ModuleType:
    source_path = Path(__file__).with_name("opt_cache.cpp")
    build_root_env = os.environ.get("TAAC_TORCH_EXTENSIONS_DIR", "")
    build_directory = None
    if build_root_env:
        build_root = Path(build_root_env)
        extension_build_dir = (build_root / "taac_opt_cache_index").resolve()
        extension_build_dir.mkdir(parents=True, exist_ok=True)
        build_directory = str(extension_build_dir)
    return load(
        name="taac_opt_cache_index",
        sources=[str(source_path)],
        extra_cflags=["-O3", "-std=c++17"],
        build_directory=build_directory,
        verbose=os.environ.get("TAAC_VERBOSE_EXTENSIONS") == "1",
        with_cuda=False,
    )


__all__ = ["load_native_opt_cache"]
