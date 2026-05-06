"""Online inference bundle runtime entrypoint."""

from __future__ import annotations

import os
from pathlib import Path

from taac2026.infrastructure.platform.env import ONLINE_INFERENCE_BUNDLE_PLATFORM
from taac2026.infrastructure.platform.deps import (
    install_project_pip_dependencies,
    prepare_project_imports,
    read_manifest,
    set_default_experiment_from_manifest,
)


def run_inference_bundle(project_dir: Path) -> None:
    manifest = read_manifest(project_dir / ".taac_inference_manifest.json")
    install_project_pip_dependencies(project_dir, ONLINE_INFERENCE_BUNDLE_PLATFORM)
    set_default_experiment_from_manifest(manifest)
    prepare_project_imports(project_dir)
    os.chdir(project_dir)

    from taac2026.application.evaluation.infer import main as packaged_main

    packaged_main()


__all__ = ["run_inference_bundle"]
