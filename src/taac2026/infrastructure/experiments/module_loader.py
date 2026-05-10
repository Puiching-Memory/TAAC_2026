"""Low-level experiment module loading primitives."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

from taac2026.infrastructure.io.files import stable_hash64


def _dynamic_experiment_module_name(resolved_path: Path) -> str:
    return f"taac2026_dynamic_experiment_{stable_hash64(str(resolved_path))}"


def load_module_from_path(path: Path) -> ModuleType:
    resolved_path = path.expanduser().resolve()
    if resolved_path.is_dir():
        init_path = resolved_path / "__init__.py"
        if not init_path.exists():
            raise FileNotFoundError(f"experiment directory lacks __init__.py: {resolved_path}")
        module_name = _dynamic_experiment_module_name(resolved_path)
        spec = importlib.util.spec_from_file_location(
            module_name,
            init_path,
            submodule_search_locations=[str(resolved_path)],
        )
    else:
        module_name = _dynamic_experiment_module_name(resolved_path)
        spec = importlib.util.spec_from_file_location(module_name, resolved_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load experiment package from {resolved_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_experiment_submodule(package_dir: Path, submodule: str) -> ModuleType:
    resolved_dir = package_dir.expanduser().resolve()
    init_path = resolved_dir / "__init__.py"
    if not init_path.exists():
        raise FileNotFoundError(f"experiment directory lacks __init__.py: {resolved_dir}")
    if "." in submodule or "/" in submodule or "\\" in submodule:
        raise ValueError(f"experiment submodule must be a direct module name, got {submodule!r}")
    submodule_path = resolved_dir / f"{submodule}.py"
    if not submodule_path.exists():
        raise FileNotFoundError(f"experiment submodule not found: {submodule_path}")

    package_name = _dynamic_experiment_module_name(resolved_dir)
    if package_name not in sys.modules:
        load_module_from_path(resolved_dir)
    module_name = f"{package_name}.{submodule}"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(module_name, submodule_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load experiment submodule {submodule!r} from {resolved_dir}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


__all__ = ["load_experiment_submodule", "load_module_from_path"]