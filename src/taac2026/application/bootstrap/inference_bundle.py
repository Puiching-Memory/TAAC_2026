"""Online inference bundle runtime entrypoint."""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path

from taac2026.infrastructure.platform.env import ONLINE_INFERENCE_BUNDLE_PLATFORM
from taac2026.infrastructure.platform.deps import (
    install_project_pip_dependencies,
    prepare_project_imports,
    read_manifest,
    set_default_experiment_from_manifest,
)


@contextmanager
def _temporary_default_env(name: str, value: str):
    had_original = name in os.environ
    original = os.environ.get(name)
    if not had_original:
        os.environ[name] = value
    try:
        yield
    finally:
        if not had_original:
            os.environ.pop(name, None)
        elif original is not None:
            os.environ[name] = original


def run_inference_bundle(project_dir: Path) -> None:
    with _temporary_default_env("TAAC_BUNDLE_MODE", "1"):
        manifest = read_manifest(project_dir / ".taac_inference_manifest.json")
        install_project_pip_dependencies(project_dir, ONLINE_INFERENCE_BUNDLE_PLATFORM)
        set_default_experiment_from_manifest(manifest)
        prepare_project_imports(project_dir)
        os.chdir(project_dir)

        from taac2026.application.evaluation.infer import main as packaged_main

        packaged_main()


__all__ = ["run_inference_bundle"]
