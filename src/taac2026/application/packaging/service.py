"""Shared helpers for training and inference bundle packaging commands."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from taac2026.application.experiments.registry import load_experiment_package
from taac2026.infrastructure.bundles.zip_writer import (
    BundleResult,
    bundle_payload,
    format_bundle_summary,
    write_workspace_code_package,
)
from taac2026.infrastructure.bundles.manifest_store import BundleKind, build_bundle_manifest, get_bundle_definition
from taac2026.infrastructure.io.files import repo_root
from taac2026.infrastructure.io.json import dumps
from taac2026.infrastructure.io.streams import write_stdout_line


EntryPointWriter = Callable[[Path, Path], None]


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


@dataclass(frozen=True, slots=True)
class BundleCommand:
    kind: BundleKind
    description: str
    output_subdir: str
    output_suffix: str
    write_entrypoint: EntryPointWriter


def build_bundle(
    experiment: str,
    *,
    command: BundleCommand,
    output_dir: Path | None = None,
    output_path: Path | None = None,
    force: bool = False,
    root: Path | None = None,
) -> BundleResult:
    workspace_root = (root or repo_root()).resolve()
    experiment_path = resolve_experiment_path(experiment, workspace_root)
    definition = get_bundle_definition(command.kind)

    if output_path is not None and output_dir is not None:
        raise ValueError("output_path and output_dir cannot both be set")
    if output_dir is None:
        output_dir = output_path
    if output_dir is None:
        output_dir = workspace_root / "outputs" / command.output_subdir / f"{experiment_path.name}_{command.output_suffix}"

    resolved_output_dir = output_dir.expanduser().resolve()
    if resolved_output_dir.exists() and not resolved_output_dir.is_dir():
        raise NotADirectoryError(f"output path is not a directory: {resolved_output_dir}")
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    entrypoint_path = resolved_output_dir / definition.entrypoint
    code_package_path = resolved_output_dir / "code_package.zip"
    existing_targets = [path for path in (entrypoint_path, code_package_path) if path.exists()]
    if existing_targets and not force:
        names = ", ".join(path.name for path in existing_targets)
        raise FileExistsError(f"{command.kind} bundle file(s) already exist: {names}")

    manifest = build_bundle_manifest(kind=command.kind, experiment_path=experiment_path, root=workspace_root)
    if force:
        for target in (entrypoint_path, code_package_path):
            if target.exists():
                target.unlink()

    command.write_entrypoint(workspace_root, entrypoint_path)
    entrypoint_path.chmod(0o755)
    write_workspace_code_package(
        code_package_path=code_package_path,
        experiment_path=experiment_path,
        root=workspace_root,
        manifest=manifest,
        manifest_name=definition.manifest_name,
    )
    return BundleResult(
        output_dir=resolved_output_dir,
        run_script_path=entrypoint_path,
        code_package_path=code_package_path,
        manifest=manifest,
    )


def run_bundle_cli(
    argv: Sequence[str] | None,
    *,
    command: BundleCommand,
    builder: Callable[..., BundleResult],
) -> int:
    parser = argparse.ArgumentParser(description=command.description)
    parser.add_argument("--experiment", default="experiments/baseline")
    parser.add_argument("--output-dir", "--output", dest="output_dir", default=None)
    parser.add_argument("--force", dest="force", action="store_true", default=True)
    parser.add_argument("--no-force", dest="force", action="store_false")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    result = builder(
        args.experiment,
        output_dir=Path(args.output_dir) if args.output_dir else None,
        force=args.force,
    )
    if args.json:
        write_stdout_line(dumps(bundle_payload(result), indent=2))
    else:
        write_stdout_line(format_bundle_summary(result, kind=command.kind))
    return 0


__all__ = ["BundleCommand", "build_bundle", "resolve_experiment_path", "run_bundle_cli"]