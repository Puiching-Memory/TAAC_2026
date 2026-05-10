from __future__ import annotations

import sys
from pathlib import Path
from textwrap import dedent

from taac2026.infrastructure.experiments.module_loader import load_experiment_submodule, load_module_from_path


def _write_package(package_dir: Path) -> None:
    package_dir.mkdir()
    (package_dir / "__init__.py").write_text("EXPERIMENT = object()\n", encoding="utf-8")
    (package_dir / "utils.py").write_text("VALUE = 41\n", encoding="utf-8")
    (package_dir / "model.py").write_text(
        dedent(
            """
            from .utils import VALUE

            MODEL_VALUE = VALUE + 1
            """
        ).lstrip(),
        encoding="utf-8",
    )


def test_load_experiment_submodule_uses_package_qualified_name(tmp_path: Path) -> None:
    package_dir = tmp_path / "experiment_pkg"
    _write_package(package_dir)
    sys.modules.pop("model", None)
    sys.modules.pop("utils", None)

    package_module = load_module_from_path(package_dir)
    model_module = load_experiment_submodule(package_dir, "model")

    assert model_module.MODEL_VALUE == 42
    assert model_module.__name__ == f"{package_module.__name__}.model"
    assert "model" not in sys.modules
    assert "utils" not in sys.modules


def test_load_experiment_submodule_rejects_nested_names(tmp_path: Path) -> None:
    package_dir = tmp_path / "experiment_pkg"
    _write_package(package_dir)

    try:
        load_experiment_submodule(package_dir, "nested.model")
    except ValueError as error:
        assert "direct module name" in str(error)
    else:
        raise AssertionError("expected nested experiment submodule names to be rejected")