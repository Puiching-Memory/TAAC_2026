"""Build uploadable online training files."""

from __future__ import annotations

import shutil
from collections.abc import Sequence
from pathlib import Path

from taac2026.application.maintenance._bundle_packaging import (
    BundleCommand,
    build_bundle,
    run_bundle_cli,
)
from taac2026.infrastructure.bundles.common import BundleResult


def _write_training_entrypoint(workspace_root: Path, target_path: Path) -> None:
    shutil.copy2(workspace_root / "run.sh", target_path)


_TRAINING_COMMAND = BundleCommand(
    kind="training",
    description="Build a TAAC online training bundle",
    output_subdir="training_bundles",
    output_suffix="training_bundle",
    write_entrypoint=_write_training_entrypoint,
)


def build_training_bundle(
    experiment: str,
    *,
    output_dir: Path | None = None,
    output_path: Path | None = None,
    force: bool = False,
    root: Path | None = None,
) -> BundleResult:
    return build_bundle(
        experiment,
        command=_TRAINING_COMMAND,
        output_dir=output_dir,
        output_path=output_path,
        force=force,
        root=root,
    )


def main(argv: Sequence[str] | None = None) -> int:
    return run_bundle_cli(argv, command=_TRAINING_COMMAND, builder=build_training_bundle)


if __name__ == "__main__":
    raise SystemExit(main())
