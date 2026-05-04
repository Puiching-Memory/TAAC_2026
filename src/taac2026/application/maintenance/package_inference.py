"""Build uploadable online inference files."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from taac2026.infrastructure.bundles.common import (
    BundleResult,
    bundle_payload,
    resolve_experiment_path,
    write_workspace_code_package,
)
from taac2026.infrastructure.io.files import repo_root
from taac2026.infrastructure.io.json_utils import dumps


_INFER_ENTRYPOINT = """#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import os
import shlex
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import orjson


_TENCENT_PYPI_INDEX_URL = "https://mirrors.cloud.tencent.com/pypi/simple/"


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
    if os.environ.get("TAAC_FORCE_EXTRACT") == "1" or not (project_dir / "pyproject.toml").exists():
        if project_dir.exists():
            shutil.rmtree(project_dir)
        workdir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(package_path) as archive:
            archive.extractall(workdir)
    return project_dir


def _read_manifest(manifest_path: Path) -> dict[str, object]:
    if not manifest_path.exists():
        return {}
    with manifest_path.open("rb") as handle:
        return orjson.loads(handle.read())


def _split_env_words(name: str) -> list[str]:
    value = os.environ.get(name, "")
    if not value:
        return []
    return shlex.split(value)


def _should_install_project_pip_dependencies() -> bool:
    if os.environ.get("TAAC_SKIP_PIP_INSTALL") == "1":
        return False
    install_project_deps = os.environ.get("TAAC_INSTALL_PROJECT_DEPS")
    if install_project_deps is not None:
        return install_project_deps != "0"
    return True


def _install_project_pip_dependencies(project_dir: Path) -> None:
    if not _should_install_project_pip_dependencies():
        return
    extras = _split_env_words("TAAC_BUNDLE_PIP_EXTRAS")
    target = "."
    if extras:
        target = f".[{','.join(extras)}]"

    command = [sys.executable, "-m", "pip", "install", "--disable-pip-version-check"]
    index_url = os.environ.get("TAAC_PIP_INDEX_URL", _TENCENT_PYPI_INDEX_URL)
    if index_url:
        command.extend(["-i", index_url])
    command.extend(_split_env_words("TAAC_PIP_EXTRA_ARGS"))
    command.append(target)

    print("Installing TAAC project dependencies from pyproject.toml", file=sys.stderr)
    subprocess.check_call(command, cwd=project_dir)


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
    manifest = _read_manifest(project_dir / ".taac_inference_manifest.json")
    _install_project_pip_dependencies(project_dir)
    default_experiment = manifest.get("bundled_experiment_path")
    if isinstance(default_experiment, str) and default_experiment and "TAAC_EXPERIMENT" not in os.environ:
        os.environ["TAAC_EXPERIMENT"] = default_experiment

    sys.path.insert(0, str(project_dir))
    sys.path.insert(0, str(project_dir / "src"))
    os.chdir(project_dir)

    from taac2026.application.evaluation.infer import main as packaged_main

    packaged_main()


if __name__ == "__main__":
    main()
"""


def _format_bundle_summary(result: BundleResult) -> str:
    manifest = result.manifest
    runtime_env = manifest.get("runtime_env", {})
    lines = [
        "Built TAAC online inference bundle",
        f"Experiment: {manifest.get('bundled_experiment_path', '<unknown>')}",
        f"Output dir: {result.output_dir}",
        f"infer.py: {result.run_script_path}",
        f"code_package.zip: {result.code_package_path}",
        f"Bundle format: {manifest.get('bundle_format', '<unknown>')}",
    ]
    if isinstance(runtime_env, dict):
        lines.extend(
            [
                "Runtime env:",
                f"  model: {runtime_env.get('model_path', '<unknown>')}",
                f"  dataset: {runtime_env.get('dataset_path', '<unknown>')}",
                f"  result: {runtime_env.get('result_path', '<unknown>')}",
                f"  schema: {runtime_env.get('schema_path', '<unknown>')}",
                f"  pip extras: {runtime_env.get('pip_extras', '<unknown>')}",
            ]
        )
    lines.append("Upload the two files above: infer.py and code_package.zip")
    return "\n".join(lines)
def build_inference_bundle(
    experiment: str,
    *,
    output_dir: Path | None = None,
    output_path: Path | None = None,
    force: bool = False,
    root: Path | None = None,
) -> BundleResult:
    workspace_root = (root or repo_root()).resolve()
    experiment_path = resolve_experiment_path(experiment, workspace_root)
    if output_path is not None and output_dir is not None:
        raise ValueError("output_path and output_dir cannot both be set")
    if output_dir is None:
        output_dir = output_path
    if output_dir is None:
        output_dir = workspace_root / "outputs" / "inference_bundles" / f"{experiment_path.name}_inference_bundle"
    resolved_output_dir = output_dir.expanduser().resolve()
    if resolved_output_dir.exists() and not resolved_output_dir.is_dir():
        raise NotADirectoryError(f"output path is not a directory: {resolved_output_dir}")
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    infer_script_path = resolved_output_dir / "infer.py"
    code_package_path = resolved_output_dir / "code_package.zip"
    existing_targets = [path for path in (infer_script_path, code_package_path) if path.exists()]
    if existing_targets and not force:
        names = ", ".join(path.name for path in existing_targets)
        raise FileExistsError(f"inference bundle file(s) already exist: {names}")

    manifest: dict[str, object] = {
        "bundle_format": "taac2026-inference-v1",
        "bundled_experiment_path": str(experiment_path.relative_to(workspace_root)),
        "entrypoint": "infer.py",
        "code_package": "code_package.zip",
        "runtime_env": {
            "model_path": "MODEL_OUTPUT_PATH",
            "dataset_path": "EVAL_DATA_PATH",
            "result_path": "EVAL_RESULT_PATH",
            "schema_path": "TAAC_SCHEMA_PATH",
            "pip_extras": "TAAC_BUNDLE_PIP_EXTRAS (optional; defaults to runtime-only install with no dev extra)",
        },
    }
    if force:
        for target in (infer_script_path, code_package_path):
            if target.exists():
                target.unlink()
    infer_script_path.write_text(_INFER_ENTRYPOINT, encoding="utf-8")
    infer_script_path.chmod(0o755)
    write_workspace_code_package(
        code_package_path=code_package_path,
        experiment_path=experiment_path,
        root=workspace_root,
        manifest=manifest,
        manifest_name=".taac_inference_manifest.json",
    )
    return BundleResult(
        output_dir=resolved_output_dir,
        run_script_path=infer_script_path,
        code_package_path=code_package_path,
        manifest=manifest,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a TAAC online inference bundle")
    parser.add_argument("--experiment", default="experiments/pcvr/baseline")
    parser.add_argument("--output-dir", "--output", dest="output_dir", default=None)
    parser.add_argument("--force", dest="force", action="store_true", default=True)
    parser.add_argument("--no-force", dest="force", action="store_false")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    result = build_inference_bundle(
        args.experiment,
        output_dir=Path(args.output_dir) if args.output_dir else None,
        force=args.force,
    )
    if args.json:
        print(dumps(bundle_payload(result), indent=2))
    else:
        print(_format_bundle_summary(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())