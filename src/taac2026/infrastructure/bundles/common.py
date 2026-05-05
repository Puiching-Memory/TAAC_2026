"""Shared bundle helpers used by training and inference packaging commands."""

from __future__ import annotations

import zipfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from taac2026.infrastructure.bundles.manifest import BundleKind, get_bundle_definition
from taac2026.infrastructure.experiments.loader import load_experiment_package
from taac2026.infrastructure.io.files import repo_root
from taac2026.infrastructure.io.json_utils import dump_bytes


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


def resolve_experiment_path(experiment: str, root: Path | None = None) -> Path:
    workspace_root = (root or repo_root()).resolve()
    direct = Path(experiment)
    candidates = [direct, workspace_root / experiment, workspace_root / experiment.replace(".", "/")]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    loaded = load_experiment_package(experiment)
    if loaded.package_dir is None:
        raise FileNotFoundError(f"cannot resolve filesystem package for {experiment}")
    return loaded.package_dir.resolve()


def write_workspace_code_package(
    *,
    code_package_path: Path,
    experiment_path: Path,
    root: Path,
    manifest: dict[str, object],
    manifest_name: str,
) -> None:
    relative_experiment_path = experiment_path.relative_to(root)
    with zipfile.ZipFile(code_package_path, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            f"project/{manifest_name}",
            dump_bytes(manifest, indent=2, trailing_newline=True),
        )
        pyproject_path = root / "pyproject.toml"
        if pyproject_path.exists():
            add_file_to_zip(archive, pyproject_path, "project/pyproject.toml")

        current = root
        for part in relative_experiment_path.parts[:-1]:
            current = current / part
            init_path = current / "__init__.py"
            if init_path.exists():
                add_file_to_zip(archive, init_path, f"project/{init_path.relative_to(root)}")

        for source in iter_python_tree(root / "src" / "taac2026"):
            add_file_to_zip(archive, source, f"project/{source.relative_to(root)}")
        for source in iter_file_tree(experiment_path):
            add_file_to_zip(archive, source, f"project/{source.relative_to(root)}")


__all__ = [
    "BundleResult",
    "add_file_to_zip",
    "bundle_payload",
    "format_bundle_summary",
    "iter_file_tree",
    "iter_python_tree",
    "resolve_experiment_path",
    "write_workspace_code_package",
]