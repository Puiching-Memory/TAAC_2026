"""Low-level experiment module loading primitives."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

from taac2026.infrastructure.io.files import stable_hash64


def load_module_from_path(path: Path) -> ModuleType:
    resolved_path = path.expanduser().resolve()
    if resolved_path.is_dir():
        init_path = resolved_path / "__init__.py"
        if not init_path.exists():
            raise FileNotFoundError(f"experiment directory lacks __init__.py: {resolved_path}")
        module_name = f"taac2026_dynamic_experiment_{stable_hash64(str(resolved_path))}"
        spec = importlib.util.spec_from_file_location(
            module_name,
            init_path,
            submodule_search_locations=[str(resolved_path)],
        )
    else:
        module_name = f"taac2026_dynamic_experiment_{stable_hash64(str(resolved_path))}"
        spec = importlib.util.spec_from_file_location(module_name, resolved_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load experiment package from {resolved_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


__all__ = ["load_module_from_path"]