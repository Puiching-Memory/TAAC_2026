"""Small runtime request objects used by the unified CLI entrypoints."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class TrainRequest:
    experiment: str
    dataset_path: Path | None
    schema_path: Path | None
    run_dir: Path
    extra_args: tuple[str, ...] = ()


@dataclass(slots=True)
class EvalRequest:
    experiment: str
    dataset_path: Path
    schema_path: Path | None
    run_dir: Path
    checkpoint_path: Path | None = None
    output_path: Path | None = None
    predictions_path: Path | None = None
    batch_size: int = 256
    num_workers: int = 0
    device: str = "cpu"
    amp: bool | None = None
    amp_dtype: str | None = None
    compile: bool | None = None
    is_training_data: bool = True
    extra_args: tuple[str, ...] = ()


@dataclass(slots=True)
class InferRequest:
    experiment: str
    dataset_path: Path
    schema_path: Path | None
    checkpoint_path: Path | None
    result_dir: Path
    batch_size: int = 256
    num_workers: int = 0
    device: str = "cpu"
    amp: bool | None = None
    amp_dtype: str | None = None
    compile: bool | None = None


def experiment_slug(value: str) -> str:
    cleaned = value.strip().replace("\\", "/").strip("/")
    if cleaned.startswith("./"):
        cleaned = cleaned[2:]
    return cleaned.replace("/", "_").replace(".", "_") or "experiment"


def default_run_dir(experiment: str) -> Path:
    return Path("outputs") / experiment_slug(experiment)
