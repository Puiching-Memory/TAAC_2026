#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import os
import shutil
import sys
import zipfile
from pathlib import Path

# Delegated to taac2026.application.bootstrap.inference_bundle:
# TAAC_INSTALL_PROJECT_DEPS, TAAC_BUNDLE_PIP_EXTRAS, TAAC_PIP_EXTRAS, and
# taac2026.application.evaluation.infer.


def _default_bundle_workdir(script_dir: Path, code_package: Path) -> Path:
    package_stat = code_package.stat() if code_package.exists() else None
    cache_source = "::".join(
        (
            str(script_dir.resolve()),
            str(code_package.resolve(strict=False)),
            str(package_stat.st_mtime_ns) if package_stat is not None else "missing",
            str(package_stat.st_size) if package_stat is not None else "missing",
        )
    )
    cache_key = hashlib.sha256(cache_source.encode("utf-8")).hexdigest()[:16]
    user_cache_path = os.environ.get("USER_CACHE_PATH")
    if user_cache_path:
        return Path(user_cache_path).expanduser() / "taac2026_infer_bundle" / cache_key
    raise RuntimeError("USER_CACHE_PATH is required unless TAAC_BUNDLE_WORKDIR is set")


def _extract_code_package(package_path: Path, workdir: Path) -> Path:
    project_dir = workdir / "project"
    if project_dir.exists():
        shutil.rmtree(project_dir)
    workdir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(package_path) as archive:
        archive.extractall(workdir)
    return project_dir


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    code_package = Path(os.environ.get("TAAC_CODE_PACKAGE", str(script_dir / "code_package.zip"))).expanduser()
    workdir = Path(
        os.environ.get(
            "TAAC_BUNDLE_WORKDIR",
            str(_default_bundle_workdir(script_dir, code_package)),
        )
    ).expanduser()
    if not code_package.is_file():
        raise FileNotFoundError(f"code_package.zip not found: {code_package}")

    project_dir = _extract_code_package(code_package, workdir)
    sys.path.insert(0, str(project_dir))
    sys.path.insert(0, str(project_dir / "src"))

    from taac2026.application.bootstrap.inference_bundle import run_inference_bundle

    run_inference_bundle(project_dir)


if __name__ == "__main__":
    main()
