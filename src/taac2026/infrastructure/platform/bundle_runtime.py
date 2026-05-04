"""Shared runtime helpers for online training and inference bundles."""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path

from taac2026.infrastructure.platform.adapters import RuntimePlatform


TENCENT_PYPI_INDEX_URL = "https://mirrors.cloud.tencent.com/pypi/simple/"


def read_manifest(manifest_path: Path) -> dict[str, object]:
    if not manifest_path.exists():
        return {}
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def split_env_words(name: str) -> list[str]:
    value = os.environ.get(name, "")
    if not value:
        return []
    return shlex.split(value)


def resolve_python() -> str:
    return os.environ.get("TAAC_PYTHON") or sys.executable


def should_install_project_pip_dependencies(platform: RuntimePlatform) -> bool:
    if os.environ.get("TAAC_SKIP_PIP_INSTALL") == "1":
        return False
    install_project_deps = os.environ.get("TAAC_INSTALL_PROJECT_DEPS")
    if install_project_deps is not None:
        return install_project_deps != "0"
    return platform.install_project_deps_by_default


def install_project_pip_dependencies(project_dir: Path, platform: RuntimePlatform) -> None:
    if not should_install_project_pip_dependencies(platform):
        return

    extras = split_env_words(platform.pip_extras_env)
    target = "."
    if extras:
        target = f".[{','.join(extras)}]"

    command = [resolve_python(), "-m", "pip", "install", "--disable-pip-version-check"]
    index_url = os.environ.get("TAAC_PIP_INDEX_URL", TENCENT_PYPI_INDEX_URL)
    if index_url:
        command.extend(["-i", index_url])
    command.extend(split_env_words("TAAC_PIP_EXTRA_ARGS"))
    command.append(target)

    print("Installing TAAC project dependencies from pyproject.toml", file=sys.stderr)
    subprocess.check_call(command, cwd=project_dir)


def set_default_experiment_from_manifest(manifest: Mapping[str, object]) -> None:
    default_experiment = manifest.get("bundled_experiment_path")
    if isinstance(default_experiment, str) and default_experiment and "TAAC_EXPERIMENT" not in os.environ:
        os.environ["TAAC_EXPERIMENT"] = default_experiment


def prepare_project_imports(project_dir: Path) -> None:
    src_dir = project_dir / "src"
    for candidate in (src_dir, project_dir):
        text = str(candidate)
        if text not in sys.path:
            sys.path.insert(0, text)


__all__ = [
    "TENCENT_PYPI_INDEX_URL",
    "install_project_pip_dependencies",
    "prepare_project_imports",
    "read_manifest",
    "resolve_python",
    "set_default_experiment_from_manifest",
    "should_install_project_pip_dependencies",
    "split_env_words",
]