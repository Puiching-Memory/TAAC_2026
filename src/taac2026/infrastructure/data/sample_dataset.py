"""Default local sample dataset resolution for PCVR workflows."""

from __future__ import annotations

import logging
from pathlib import Path

from datasets.utils.file_utils import huggingface_hub


_SAMPLE_DATASET_REPO_ID = "TAAC2026/data_sample_1000"
_SAMPLE_PARQUET_FILENAME = "demo_1000.parquet"


def _workspace_root() -> Path:
    return Path(__file__).resolve().parents[4]


def default_pcvr_sample_schema_path() -> Path:
    schema_path = _workspace_root() / "data" / "sample_1000_raw" / "schema.json"
    if not schema_path.exists():
        raise FileNotFoundError(
            "default PCVR sample schema.json not found in repository data/sample_1000_raw; "
            "pass --schema-path explicitly"
        )
    return schema_path.resolve()


def default_pcvr_sample_dataset_path() -> Path:
    dataset_path = huggingface_hub.hf_hub_download(
        repo_id=_SAMPLE_DATASET_REPO_ID,
        repo_type="dataset",
        filename=_SAMPLE_PARQUET_FILENAME,
    )
    return Path(dataset_path).expanduser().resolve()


def resolve_default_pcvr_sample_paths(
    dataset_path: Path | None,
    schema_path: Path | None,
) -> tuple[Path, Path | None]:
    if dataset_path is not None:
        resolved_dataset_path = dataset_path.expanduser().resolve()
        resolved_schema_path = schema_path.expanduser().resolve() if schema_path is not None else None
        return resolved_dataset_path, resolved_schema_path

    resolved_dataset_path = default_pcvr_sample_dataset_path()
    resolved_schema_path = schema_path.expanduser().resolve() if schema_path is not None else default_pcvr_sample_schema_path()
    logging.info(
        "Resolved default PCVR sample dataset from Hugging Face: dataset_path=%s schema_path=%s",
        resolved_dataset_path,
        resolved_schema_path,
    )
    return resolved_dataset_path, resolved_schema_path


__all__ = [
    "default_pcvr_sample_dataset_path",
    "default_pcvr_sample_schema_path",
    "resolve_default_pcvr_sample_paths",
]