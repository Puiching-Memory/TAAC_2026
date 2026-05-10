"""Shared bundle helpers used by training and inference packaging commands."""

from __future__ import annotations

import zipfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from taac2026.infrastructure.bundles.manifest_store import BundleKind, get_bundle_definition
from taac2026.infrastructure.io.json import dump_bytes


@dataclass(slots=True)
class BundleResult:
    output_dir: Path
    run_script_path: Path
    code_package_path: Path
    manifest: dict[str, object]


def bundle_payload(result: BundleResult) -> dict[str, object]:
    return {
        "output_dir": str(result.output_dir),
        "run_script_path": str(result.run_script_path),
        "code_package_path": str(result.code_package_path),
        "manifest": result.manifest,
    }


def format_bundle_summary(result: BundleResult, *, kind: BundleKind) -> str:
    definition = get_bundle_definition(kind)
    manifest = result.manifest
    runtime_env = manifest.get("runtime_env", {})
    lines = [
        f"Built TAAC online {kind} bundle",
        f"Experiment: {manifest.get('bundled_experiment_path', '<unknown>')}",
        f"Output dir: {result.output_dir}",
        f"{definition.entrypoint}: {result.run_script_path}",
        f"code_package.zip: {result.code_package_path}",
        f"Bundle format: {manifest.get('bundle_format', '<unknown>')}",
    ]
    if isinstance(runtime_env, dict):
        lines.append("Runtime env:")
        for label, key in definition.summary_runtime_fields:
            lines.append(f"  {label}: {runtime_env.get(key, '<unknown>')}")
    lines.append(f"Upload the two files above: {definition.entrypoint} and code_package.zip")
    return "\n".join(lines)


def _iter_clean_tree(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        if path.is_dir():
            continue
        if "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        yield path


def iter_python_tree(root: Path) -> Iterable[Path]:
    return _iter_clean_tree(root)


def iter_file_tree(root: Path) -> Iterable[Path]:
    return _iter_clean_tree(root)


def add_file_to_zip(archive: zipfile.ZipFile, source: Path, arcname: str) -> None:
    archive.write(source, arcname=arcname)


def _add_experiment_parent_inits(archive: zipfile.ZipFile, *, root: Path, bundled_experiment_path: str) -> None:
    parent_parts = PurePosixPath(bundled_experiment_path).parts[:-1]
    current = root
    for index, part in enumerate(parent_parts, start=1):
        current = current / part
        init_path = current / "__init__.py"
        if init_path.exists():
            archive_parent = PurePosixPath(*parent_parts[:index])
            add_file_to_zip(archive, init_path, f"project/{archive_parent}/__init__.py")


def write_workspace_code_package(
    *,
    code_package_path: Path,
    experiment_path: Path,
    root: Path,
    manifest: dict[str, object],
    manifest_name: str,
) -> None:
    bundled_experiment_path = str(manifest["bundled_experiment_path"])
    with zipfile.ZipFile(code_package_path, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            f"project/{manifest_name}",
            dump_bytes(manifest, indent=2, trailing_newline=True),
        )
        pyproject_path = root / "pyproject.toml"
        if pyproject_path.exists():
            add_file_to_zip(archive, pyproject_path, "project/pyproject.toml")

        _add_experiment_parent_inits(archive, root=root, bundled_experiment_path=bundled_experiment_path)

        for source in iter_python_tree(root / "src" / "taac2026"):
            add_file_to_zip(archive, source, f"project/{source.relative_to(root)}")
        for source in iter_file_tree(experiment_path):
            archive_relative_path = PurePosixPath(bundled_experiment_path) / source.relative_to(experiment_path).as_posix()
            add_file_to_zip(archive, source, f"project/{archive_relative_path}")


__all__ = [
    "BundleResult",
    "add_file_to_zip",
    "bundle_payload",
    "format_bundle_summary",
    "iter_file_tree",
    "iter_python_tree",
    "write_workspace_code_package",
]