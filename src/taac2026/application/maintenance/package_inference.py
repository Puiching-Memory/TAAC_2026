"""Build uploadable online inference files."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from taac2026.application.maintenance._bundle_packaging import (
    BundleCommand,
    build_bundle,
    run_bundle_cli,
)
from taac2026.infrastructure.bundles.common import BundleResult
from taac2026.infrastructure.platform import inference_bundle_entrypoint


def _read_inference_entrypoint_source() -> str:
    return Path(inference_bundle_entrypoint.__file__).read_text(encoding="utf-8")


def _write_inference_entrypoint(_workspace_root: Path, target_path: Path) -> None:
    target_path.write_text(_read_inference_entrypoint_source(), encoding="utf-8")


_INFERENCE_COMMAND = BundleCommand(
    kind="inference",
    description="Build a TAAC online inference bundle",
    output_subdir="inference_bundles",
    output_suffix="inference_bundle",
    write_entrypoint=_write_inference_entrypoint,
)


def build_inference_bundle(
    experiment: str,
    *,
    output_dir: Path | None = None,
    output_path: Path | None = None,
    force: bool = False,
    root: Path | None = None,
) -> BundleResult:
    return build_bundle(
        experiment,
        command=_INFERENCE_COMMAND,
        output_dir=output_dir,
        output_path=output_path,
        force=force,
        root=root,
    )


def main(argv: Sequence[str] | None = None) -> int:
    return run_bundle_cli(argv, command=_INFERENCE_COMMAND, builder=build_inference_bundle)


if __name__ == "__main__":
    raise SystemExit(main())