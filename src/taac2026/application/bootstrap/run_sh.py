"""Python runtime for the top-level run.sh bootstrapper."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from taac2026.infrastructure.io.streams import write_stderr_line
from taac2026.infrastructure.platform.env import RuntimePlatform, select_run_sh_platform
from taac2026.infrastructure.platform.deps import (
    install_project_pip_dependencies,
    read_manifest,
    resolve_python,
)


SUPPORTED_CUDA_PROFILE = "cuda126"
SUPPORTED_COMMANDS = {"train", "val", "eval", "infer"}


@dataclass(frozen=True, slots=True)
class ParsedRunCommand:
    command: str
    args: list[str]


@dataclass(frozen=True, slots=True)
class CudaArgs:
    profile: str
    remaining_args: list[str]


def parse_run_command(argv: Sequence[str]) -> ParsedRunCommand:
    args = list(argv)
    command = "train"
    if args:
        first = args[0]
        if first in SUPPORTED_COMMANDS:
            command = first
            args = args[1:]
        elif first.startswith("--"):
            command = "train"
        else:
            command = first
            args = args[1:]
    return ParsedRunCommand(command=command, args=args)


def extract_cuda_profile(argv: Sequence[str], default_profile: str = SUPPORTED_CUDA_PROFILE) -> CudaArgs:
    profile = os.environ.get("TAAC_CUDA_PROFILE", default_profile)
    remaining_args: list[str] = []
    iterator = iter(argv)
    for arg in iterator:
        if arg == "--cuda-profile":
            try:
                profile = next(iterator)
            except StopIteration as error:
                raise ValueError("--cuda-profile requires a value") from error
        elif arg.startswith("--cuda-profile="):
            profile = arg.split("=", 1)[1]
        else:
            remaining_args.append(arg)
    if profile != SUPPORTED_CUDA_PROFILE:
        raise ValueError(
            f"unsupported TAAC_CUDA_PROFILE/--cuda-profile: {profile}; only '{SUPPORTED_CUDA_PROFILE}' is supported"
        )
    return CudaArgs(profile=profile, remaining_args=remaining_args)


def _ensure_uv() -> str:
    uv = shutil.which("uv")
    if uv is None:
        raise FileNotFoundError("uv is required but not found in PATH")
    return uv


def _runner_mode(platform: RuntimePlatform) -> str:
    return os.environ.get("TAAC_RUNNER") or platform.default_runner


def _manifest_experiment(project_dir: Path) -> str:
    manifest = read_manifest(project_dir / ".taac_training_manifest.json")
    experiment = manifest.get("bundled_experiment_path")
    return experiment if isinstance(experiment, str) else ""


def _sync_runtime(profile: str, *, runner_mode: str) -> None:
    if runner_mode != "uv" or os.environ.get("TAAC_SKIP_UV_SYNC") == "1":
        return
    subprocess.check_call([_ensure_uv(), "sync", "--locked", "--extra", profile])


def _run_module(module_name: str, args: Sequence[str]) -> int:
    return subprocess.call([resolve_python(), "-m", module_name, *args])


def _run_console_script(
    *,
    script_name: str,
    module_name: str,
    args: Sequence[str],
    runner_mode: str,
    project_dir: Path,
    platform: RuntimePlatform,
) -> int:
    if runner_mode == "uv":
        return subprocess.call([_ensure_uv(), "run", script_name, *args])
    if runner_mode == "python":
        install_project_pip_dependencies(project_dir, platform)
        return _run_module(module_name, args)
    write_stderr_line(f"unsupported TAAC_RUNNER: {runner_mode}; expected 'python' or 'uv'")
    return 2


def _experiment_args(manifest_experiment: str) -> list[str]:
    experiment = os.environ.get("TAAC_EXPERIMENT") or manifest_experiment
    if not experiment:
        return []
    return ["--experiment", experiment]


def _optional_path_arg(flag: str, env_name: str) -> list[str]:
    value = os.environ.get(env_name, "")
    if not value:
        return []
    return [flag, value]


def _training_args(manifest_experiment: str, remaining_args: Sequence[str]) -> list[str]:
    return [
        *_experiment_args(manifest_experiment),
        *_optional_path_arg("--dataset-path", "TRAIN_DATA_PATH"),
        *_optional_path_arg("--schema-path", "TAAC_SCHEMA_PATH"),
        *_optional_path_arg("--run-dir", "TRAIN_CKPT_PATH"),
        *remaining_args,
    ]


def _evaluation_args(manifest_experiment: str, remaining_args: Sequence[str]) -> list[str]:
    return [
        "single",
        *_experiment_args(manifest_experiment),
        *_optional_path_arg("--dataset-path", "TRAIN_DATA_PATH"),
        *_optional_path_arg("--schema-path", "TAAC_SCHEMA_PATH"),
        *_optional_path_arg("--run-dir", "TRAIN_CKPT_PATH"),
        *remaining_args,
    ]


def _inference_args(manifest_experiment: str, remaining_args: Sequence[str]) -> list[str]:
    return [
        "infer",
        *_experiment_args(manifest_experiment),
        *_optional_path_arg("--dataset-path", "EVAL_DATA_PATH"),
        *_optional_path_arg("--schema-path", "TAAC_SCHEMA_PATH"),
        *_optional_path_arg("--result-dir", "EVAL_RESULT_PATH"),
        *_optional_path_arg("--checkpoint", "MODEL_OUTPUT_PATH"),
        *remaining_args,
    ]


def main(argv: Sequence[str] | None = None) -> int:
    project_dir = Path(os.environ.get("TAAC_PROJECT_DIR", ".")).resolve()
    bundle_mode = os.environ.get("TAAC_BUNDLE_MODE") == "1"
    platform = select_run_sh_platform(bundle_mode=bundle_mode, platform_name=os.environ.get("TAAC_PLATFORM"))
    runner_mode = _runner_mode(platform)
    manifest_experiment = _manifest_experiment(project_dir)

    try:
        parsed = parse_run_command(list(argv or sys.argv[1:]))
        cuda_args = extract_cuda_profile(parsed.args)
    except ValueError as error:
        write_stderr_line(str(error))
        return 2

    os.chdir(project_dir)
    _sync_runtime(cuda_args.profile, runner_mode=runner_mode)

    if parsed.command == "train":
        args = _training_args(manifest_experiment, cuda_args.remaining_args)
        return _run_console_script(
            script_name="taac-train",
            module_name="taac2026.application.training.cli",
            args=args,
            runner_mode=runner_mode,
            project_dir=project_dir,
            platform=platform,
        )
    if parsed.command in {"val", "eval"}:
        args = _evaluation_args(manifest_experiment, cuda_args.remaining_args)
        return _run_console_script(
            script_name="taac-evaluate",
            module_name="taac2026.application.evaluation.cli",
            args=args,
            runner_mode=runner_mode,
            project_dir=project_dir,
            platform=platform,
        )
    if parsed.command == "infer":
        args = _inference_args(manifest_experiment, cuda_args.remaining_args)
        return _run_console_script(
            script_name="taac-evaluate",
            module_name="taac2026.application.evaluation.cli",
            args=args,
            runner_mode=runner_mode,
            project_dir=project_dir,
            platform=platform,
        )

    write_stderr_line(f"unknown command: {parsed.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "SUPPORTED_CUDA_PROFILE",
    "CudaArgs",
    "ParsedRunCommand",
    "extract_cuda_profile",
    "main",
    "parse_run_command",
]
