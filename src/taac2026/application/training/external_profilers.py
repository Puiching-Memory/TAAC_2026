from __future__ import annotations

import os
import shlex
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ...domain.config import TrainConfig
from ...infrastructure.io.files import ensure_dir, write_json
from .runtime_optimization import runtime_optimization_cli_args


EXTERNAL_PROFILER_SCHEMA_VERSION = 1
EXTERNAL_PROFILER_ARTIFACT_DIRNAME = "profiling"
EXTERNAL_PROFILER_VERSION_TIMEOUT_SECONDS = 5.0
EXTERNAL_PROFILER_PLACEHOLDER_PREFIX = "<"
EXTERNAL_PROFILER_PLACEHOLDER_SUFFIX = ">"


@dataclass(slots=True)
class ExternalProfilerToolPlan:
    tool: str
    available: bool
    applicable: bool
    executable: str | None
    version: str | None
    report_prefix: str
    artifact_candidates: list[str]
    existing_artifacts: list[str]
    suggested_command: list[str]
    suggested_command_string: str
    notes: list[str]


def _shell_join(command: list[str]) -> str:
    if os.name == "nt":
        return subprocess.list2cmdline(command)
    return shlex.join(command)


def _placeholder_argument(value: str | Path | None, placeholder_name: str) -> str:
    if value is None:
        return f"<{placeholder_name}>"
    return str(value)


def _is_placeholder(value: str) -> bool:
    return value.startswith(EXTERNAL_PROFILER_PLACEHOLDER_PREFIX) and value.endswith(EXTERNAL_PROFILER_PLACEHOLDER_SUFFIX)


def _resolve_profiler_executable(tool: str) -> str | None:
    return shutil.which(tool)


def _read_profiler_version(tool: str, executable: str | None) -> str | None:
    if executable is None:
        return None
    version_flag_variants = {
        "ncu": [["--version"], ["-V"]],
        "nsys": [["--version"]],
    }
    for flags in version_flag_variants.get(tool, [["--version"]]):
        try:
            completed = subprocess.run(
                [executable, *flags],
                capture_output=True,
                check=False,
                stdin=subprocess.DEVNULL,
                text=True,
                timeout=EXTERNAL_PROFILER_VERSION_TIMEOUT_SECONDS,
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
        output = (completed.stdout or completed.stderr or "").strip()
        if not output:
            continue
        for line in output.splitlines():
            stripped = line.strip()
            if stripped:
                return stripped
    return None


def _report_prefix(report_directory: Path, profile_label: str, tool: str) -> Path:
    return report_directory / f"{profile_label}_{tool}"


def _artifact_candidates(report_prefix: Path, tool: str) -> list[str]:
    suffixes = {
        "ncu": [".ncu-rep", ".csv", ".json"],
        "nsys": [".nsys-rep", ".qdrep", ".sqlite", ".json"],
    }
    return [str(report_prefix.with_suffix(suffix)) for suffix in suffixes[tool]]


def _existing_artifacts(candidates: list[str]) -> list[str]:
    return [candidate for candidate in candidates if Path(candidate).exists()]


def _tool_command(tool: str, report_prefix: Path, base_command: list[str]) -> list[str]:
    if tool == "ncu":
        return [
            "ncu",
            "--set",
            "full",
            "--target-processes",
            "all",
            "-o",
            str(report_prefix),
            *base_command,
        ]
    if tool == "nsys":
        return [
            "nsys",
            "profile",
            "--trace",
            "cuda,nvtx,osrt",
            "--sample",
            "none",
            "-o",
            str(report_prefix),
            *base_command,
        ]
    raise ValueError(f"Unsupported profiler tool: {tool}")


def _tool_notes(
    *,
    tool: str,
    available: bool,
    applicable: bool,
    base_command: list[str],
) -> list[str]:
    notes: list[str] = []
    if not applicable:
        notes.append("Current run used a non-CUDA device; ncu and nsys are most useful for CUDA execution.")
    if not available:
        notes.append("Profiler executable was not found on PATH.")
    if any(_is_placeholder(argument) for argument in base_command):
        notes.append("Replace placeholder arguments before running the suggested command.")
    if tool == "ncu":
        notes.append("ncu is best for kernel-level analysis after nsys identifies the hot CUDA regions.")
    if tool == "nsys":
        notes.append("nsys is best for system-level timelines, host-device overlap, and launch sequencing.")
    return notes


def collect_external_profiler_plan(
    *,
    device: str,
    output_dir: str | Path,
    profile_label: str,
    base_command: list[str],
) -> dict[str, Any]:
    report_directory = ensure_dir(Path(output_dir) / EXTERNAL_PROFILER_ARTIFACT_DIRNAME)
    applicable = str(device).startswith("cuda")
    tools: dict[str, Any] = {}
    for tool in ("ncu", "nsys"):
        executable = _resolve_profiler_executable(tool)
        version = _read_profiler_version(tool, executable)
        prefix = _report_prefix(report_directory, profile_label, tool)
        command = _tool_command(tool, prefix, base_command)
        candidates = _artifact_candidates(prefix, tool)
        tools[tool] = asdict(
            ExternalProfilerToolPlan(
                tool=tool,
                available=executable is not None,
                applicable=applicable,
                executable=executable,
                version=version,
                report_prefix=str(prefix),
                artifact_candidates=candidates,
                existing_artifacts=_existing_artifacts(candidates),
                suggested_command=command,
                suggested_command_string=_shell_join(command),
                notes=_tool_notes(
                    tool=tool,
                    available=executable is not None,
                    applicable=applicable,
                    base_command=base_command,
                ),
            )
        )
    return {
        "schema_version": EXTERNAL_PROFILER_SCHEMA_VERSION,
        "profile_label": profile_label,
        "device": str(device),
        "report_directory": str(report_directory),
        "tools": tools,
    }


def build_training_external_profiler_plan(
    *,
    device: str,
    output_dir: str | Path,
    experiment_path: str | Path | None,
    train_config: TrainConfig | None = None,
) -> dict[str, Any]:
    base_command = [
        "uv",
        "run",
        "taac-train",
        "--experiment",
        _placeholder_argument(experiment_path, "experiment_path"),
        "--run-dir",
        str(output_dir),
    ]
    if train_config is not None:
        base_command.extend(runtime_optimization_cli_args(train_config))
    return collect_external_profiler_plan(
        device=device,
        output_dir=output_dir,
        profile_label="training",
        base_command=base_command,
    )


def build_evaluation_external_profiler_plan(
    *,
    device: str,
    output_dir: str | Path,
    experiment_path: str | Path | None,
    checkpoint_path: str | Path | None,
    output_path: str | Path | None,
    run_dir: str | Path | None = None,
    train_config: TrainConfig | None = None,
) -> dict[str, Any]:
    base_command = [
        "uv",
        "run",
        "taac-evaluate",
        "single",
        "--experiment",
        _placeholder_argument(experiment_path, "experiment_path"),
    ]
    if checkpoint_path is not None:
        base_command.extend(["--checkpoint", str(checkpoint_path)])
    if output_path is not None:
        base_command.extend(["--output-path", str(output_path)])
    if run_dir is not None:
        base_command.extend(["--run-dir", str(run_dir)])
    if train_config is not None:
        base_command.extend(runtime_optimization_cli_args(train_config))
    return collect_external_profiler_plan(
        device=device,
        output_dir=output_dir,
        profile_label="evaluation",
        base_command=base_command,
    )


def write_external_profiler_plan_artifacts(plan: dict[str, Any]) -> dict[str, Any]:
    report_directory = ensure_dir(Path(plan["report_directory"]))
    script_extension = ".ps1" if os.name == "nt" else ".sh"
    generated_artifacts: dict[str, Any] = {
        "plan_json": str(report_directory / "external_profilers.json"),
        "scripts": {},
    }
    for tool, tool_plan in plan["tools"].items():
        script_path = report_directory / f"profile_{tool}{script_extension}"
        lines: list[str] = []
        if os.name == "nt":
            lines.append("$ErrorActionPreference = 'Stop'")
        else:
            lines.extend(["#!/usr/bin/env bash", "set -euo pipefail"])
        for note in tool_plan.get("notes", []):
            lines.append(f"# {note}")
        lines.append(tool_plan["suggested_command_string"])
        script_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        if os.name != "nt":
            script_path.chmod(0o755)
        generated_artifacts["scripts"][tool] = str(script_path)
    plan["generated_artifacts"] = generated_artifacts
    write_json(generated_artifacts["plan_json"], plan)
    return generated_artifacts


__all__ = [
    "EXTERNAL_PROFILER_ARTIFACT_DIRNAME",
    "EXTERNAL_PROFILER_SCHEMA_VERSION",
    "build_evaluation_external_profiler_plan",
    "build_training_external_profiler_plan",
    "collect_external_profiler_plan",
    "write_external_profiler_plan_artifacts",
]