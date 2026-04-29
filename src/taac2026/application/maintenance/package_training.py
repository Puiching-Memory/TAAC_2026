"""Build uploadable online training files."""

from __future__ import annotations

import argparse
import shutil
import zipfile
from dataclasses import dataclass
from collections.abc import Iterable, Sequence
from pathlib import Path

from taac2026.infrastructure.experiments.loader import load_experiment_package
from taac2026.infrastructure.io.files import repo_root
from taac2026.infrastructure.io.json_utils import dump_bytes, dumps


@dataclass(slots=True)
class BundleResult:
    output_dir: Path
    run_script_path: Path
    code_package_path: Path
    manifest: dict[str, object]


def _bundle_payload(result: BundleResult) -> dict[str, object]:
    return {
        "output_dir": str(result.output_dir),
        "run_script_path": str(result.run_script_path),
        "code_package_path": str(result.code_package_path),
        "manifest": result.manifest,
    }


def _format_bundle_summary(result: BundleResult) -> str:
    manifest = result.manifest
    runtime_env = manifest.get("runtime_env", {})
    lines = [
        "Built TAAC online training bundle",
        f"Experiment: {manifest.get('bundled_experiment_path', '<unknown>')}",
        f"Output dir: {result.output_dir}",
        f"run.sh: {result.run_script_path}",
        f"code_package.zip: {result.code_package_path}",
        f"Bundle format: {manifest.get('bundle_format', '<unknown>')}",
    ]
    if isinstance(runtime_env, dict):
        lines.extend(
            [
                "Runtime env:",
                f"  dataset: {runtime_env.get('dataset_path', '<unknown>')}",
                f"  schema: {runtime_env.get('schema_path', '<unknown>')}",
                f"  output: {runtime_env.get('checkpoint_path', '<unknown>')}",
                f"  cuda profile: {runtime_env.get('cuda_profile', '<unknown>')}",
            ]
        )
    lines.append("Upload the two files above: run.sh and code_package.zip")
    return "\n".join(lines)


def _iter_python_tree(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        if path.is_dir():
            continue
        if "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        yield path


def _iter_file_tree(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        if path.is_dir():
            continue
        if "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        yield path


def _add_file_to_zip(archive: zipfile.ZipFile, source: Path, arcname: str) -> None:
    archive.write(source, arcname=arcname)


def _write_code_package(
    *,
    code_package_path: Path,
    experiment_path: Path,
    root: Path,
    manifest: dict[str, object],
) -> None:
    with zipfile.ZipFile(code_package_path, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "project/.taac_training_manifest.json",
            dump_bytes(manifest, indent=2, trailing_newline=True),
        )
        for filename in ("pyproject.toml", "uv.lock", "README.md"):
            source = root / filename
            if source.exists():
                _add_file_to_zip(archive, source, f"project/{filename}")
        config_init = root / "config" / "__init__.py"
        if config_init.exists():
            _add_file_to_zip(archive, config_init, "project/config/__init__.py")
        for source in _iter_python_tree(root / "src" / "taac2026"):
            _add_file_to_zip(archive, source, f"project/{source.relative_to(root)}")
        for source in _iter_file_tree(experiment_path):
            _add_file_to_zip(archive, source, f"project/{source.relative_to(root)}")


def _resolve_experiment_path(experiment: str, root: Path) -> Path:
    direct = Path(experiment)
    candidates = [direct, root / experiment, root / experiment.replace(".", "/")]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    loaded = load_experiment_package(experiment)
    if loaded.package_dir is None:
        raise FileNotFoundError(f"cannot resolve filesystem package for {experiment}")
    return loaded.package_dir.resolve()


def build_training_bundle(
    experiment: str,
    *,
    output_dir: Path | None = None,
    output_path: Path | None = None,
    force: bool = False,
    root: Path | None = None,
) -> BundleResult:
    workspace_root = (root or repo_root()).resolve()
    experiment_path = _resolve_experiment_path(experiment, workspace_root)
    if output_path is not None and output_dir is not None:
        raise ValueError("output_path and output_dir cannot both be set")
    if output_dir is None:
        output_dir = output_path
    if output_dir is None:
        output_dir = workspace_root / "outputs" / "training_bundles" / f"{experiment_path.name}_training_bundle"
    resolved_output_dir = output_dir.expanduser().resolve()
    if resolved_output_dir.exists() and not resolved_output_dir.is_dir():
        raise NotADirectoryError(f"output path is not a directory: {resolved_output_dir}")
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    run_script_path = resolved_output_dir / "run.sh"
    code_package_path = resolved_output_dir / "code_package.zip"
    existing_targets = [path for path in (run_script_path, code_package_path) if path.exists()]
    if existing_targets and not force:
        names = ", ".join(path.name for path in existing_targets)
        raise FileExistsError(f"training bundle file(s) already exist: {names}")

    manifest: dict[str, object] = {
        "bundle_format": "taac2026-training-v2",
        "bundled_experiment_path": str(experiment_path.relative_to(workspace_root)),
        "lockfile": "uv.lock",
        "entrypoint": "run.sh",
        "code_package": "code_package.zip",
        "runtime_env": {
            "dataset_path": "TAAC_DATASET_PATH or TRAIN_DATA_PATH",
            "schema_path": "TAAC_SCHEMA_PATH",
            "checkpoint_path": "TAAC_OUTPUT_DIR or TRAIN_CKPT_PATH",
            "cuda_profile": "TAAC_CUDA_PROFILE",
        },
    }
    if force:
        for target in (run_script_path, code_package_path):
            if target.exists():
                target.unlink()
    shutil.copy2(workspace_root / "run.sh", run_script_path)
    run_script_path.chmod(0o755)
    _write_code_package(
        code_package_path=code_package_path,
        experiment_path=experiment_path,
        root=workspace_root,
        manifest=manifest,
    )
    return BundleResult(
        output_dir=resolved_output_dir,
        run_script_path=run_script_path,
        code_package_path=code_package_path,
        manifest=manifest,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a TAAC online training bundle")
    parser.add_argument("--experiment", default="config/baseline")
    parser.add_argument("--output-dir", "--output", dest="output_dir", default=None)
    parser.add_argument("--force", dest="force", action="store_true", default=True)
    parser.add_argument("--no-force", dest="force", action="store_false")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    result = build_training_bundle(
        args.experiment,
        output_dir=Path(args.output_dir) if args.output_dir else None,
        force=args.force,
    )
    if args.json:
        print(dumps(_bundle_payload(result), indent=2))
    else:
        print(_format_bundle_summary(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
