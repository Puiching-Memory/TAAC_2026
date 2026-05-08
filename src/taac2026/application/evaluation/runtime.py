"""Composable PCVR runtime hooks for checkpoint, schema, and report handling."""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from taac2026.domain.requests import EvalRequest, InferRequest
from taac2026.infrastructure.checkpoints import resolve_checkpoint_path
from taac2026.infrastructure.io.files import read_json, write_json
import taac2026.infrastructure.data.dataset as pcvr_data
from taac2026.domain.config import REQUIRED_PCVR_TRAIN_CONFIG_KEYS
from taac2026.infrastructure.logging import logger
from taac2026.domain.sidecar import load_pcvr_train_config_sidecar
from taac2026.domain.model_contract import resolve_schema_path


def default_resolve_evaluation_checkpoint(experiment: Any, request: EvalRequest) -> Path:
    del experiment
    return resolve_checkpoint_path(request.run_dir, request.checkpoint_path)


def default_resolve_inference_checkpoint(experiment: Any, request: InferRequest) -> Path:
    del experiment
    checkpoint_root = Path(os.environ.get("MODEL_OUTPUT_PATH", "")).expanduser()
    checkpoint = resolve_checkpoint_path(Path.cwd(), request.checkpoint_path) if request.checkpoint_path else None
    if checkpoint is None and str(checkpoint_root) not in {"", "."} and checkpoint_root.exists():
        checkpoint = resolve_checkpoint_path(checkpoint_root)
    if checkpoint is None:
        checkpoint = resolve_checkpoint_path(Path.cwd())
    return checkpoint


def default_load_train_config(experiment: Any, checkpoint_dir: Path) -> dict[str, Any]:
    del experiment
    config_path = checkpoint_dir / "train_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"PCVR train_config.json not found in checkpoint directory: {checkpoint_dir}")
    config = load_pcvr_train_config_sidecar(read_json(config_path))
    missing_keys = sorted(REQUIRED_PCVR_TRAIN_CONFIG_KEYS - set(config))
    if missing_keys:
        joined = ", ".join(missing_keys)
        raise KeyError(f"PCVR train_config.json is missing required key(s): {joined}")
    return config


def default_load_runtime_schema(
    experiment: Any,
    *,
    dataset_path: Path,
    schema_path: Path | None,
    checkpoint_dir: Path,
    mode: str,
) -> tuple[Path, Any]:
    del experiment
    resolved_schema_path = resolve_schema_path(dataset_path, schema_path, checkpoint_dir)
    logger.info("Resolved PCVR {} schema.json: {}", mode, resolved_schema_path)
    return resolved_schema_path, read_json(resolved_schema_path)


def default_build_evaluation_data_diagnostics(experiment: Any, dataset_path: Path) -> dict[str, Any]:
    del experiment
    resolved_dataset_path = dataset_path.expanduser()
    warnings: list[str] = []
    try:
        rg_info = pcvr_data.collect_pcvr_row_groups(resolved_dataset_path)
        split_plan = pcvr_data.plan_pcvr_row_group_split(rg_info)
    except (FileNotFoundError, OSError, ValueError) as error:
        return {
            "dataset_path": str(resolved_dataset_path),
            "warnings": [f"row group diagnostics unavailable: {error}"],
        }

    files = sorted({path for path, _index, _rows in rg_info})
    if split_plan.reuse_train_for_valid:
        warnings.append("single Row Group dataset would reuse train rows for validation; treat as L0 smoke only")
    if not split_plan.is_l1_ready:
        warnings.append("row group split is not suitable for L1 model comparison")

    return {
        "dataset_path": str(resolved_dataset_path.resolve()),
        "file_count": len(files),
        "total_row_groups": split_plan.total_row_groups,
        "total_rows": int(sum(rows for _path, _index, rows in rg_info)),
        "row_group_split": {
            "train_row_groups": split_plan.train_row_groups,
            "valid_row_groups": split_plan.valid_row_groups,
            "train_row_group_range": list(split_plan.train_row_group_range),
            "valid_row_group_range": list(split_plan.valid_row_group_range),
            "train_rows": split_plan.train_rows,
            "valid_rows": split_plan.valid_rows,
            "reuse_train_for_valid": split_plan.reuse_train_for_valid,
            "is_disjoint": split_plan.is_disjoint,
            "is_l1_ready": split_plan.is_l1_ready,
        },
        "warnings": warnings,
    }


def default_write_observed_schema_report(
    experiment: Any,
    *,
    dataset_path: Path,
    schema_path: Path,
    output_path: Path,
    dataset_role: str,
    row_group_range: tuple[int, int] | None = None,
) -> Path:
    report = pcvr_data.build_pcvr_observed_schema_report(
        dataset_path,
        schema_path,
        row_group_range=row_group_range,
        dataset_role=dataset_role,
    )
    write_json(output_path, report)
    logger.info("Wrote PCVR observed schema report for {}: {}", dataset_role, output_path)
    return output_path


def default_write_train_split_observed_schema_reports(
    experiment: Any,
    *,
    dataset_path: Path,
    schema_path: Path,
    run_dir: Path,
    valid_ratio: float,
    train_ratio: float,
) -> dict[str, Any]:
    rg_info = pcvr_data.collect_pcvr_row_groups(dataset_path)
    split_plan = pcvr_data.plan_pcvr_row_group_split(
        rg_info,
        valid_ratio=valid_ratio,
        train_ratio=train_ratio,
    )
    observed_schema_paths = {
        "train_split": str(
            experiment.runtime_hooks.write_observed_schema_report(
                experiment,
                dataset_path=dataset_path,
                schema_path=schema_path,
                output_path=run_dir / "train_split_observed_schema.json",
                dataset_role="train_split",
                row_group_range=split_plan.train_row_group_range,
            )
        ),
        "valid_split": str(
            experiment.runtime_hooks.write_observed_schema_report(
                experiment,
                dataset_path=dataset_path,
                schema_path=schema_path,
                output_path=run_dir / "valid_split_observed_schema.json",
                dataset_role="valid_split",
                row_group_range=split_plan.valid_row_group_range,
            )
        ),
    }
    return {
        "observed_schema_paths": observed_schema_paths,
        "row_group_split": {
            "train_row_groups": split_plan.train_row_groups,
            "valid_row_groups": split_plan.valid_row_groups,
            "train_row_group_range": list(split_plan.train_row_group_range),
            "valid_row_group_range": list(split_plan.valid_row_group_range),
            "train_rows": split_plan.train_rows,
            "valid_rows": split_plan.valid_rows,
            "reuse_train_for_valid": split_plan.reuse_train_for_valid,
            "is_disjoint": split_plan.is_disjoint,
            "is_l1_ready": split_plan.is_l1_ready,
        },
    }


@dataclass(frozen=True, slots=True)
class PCVRRuntimeHooks:
    resolve_evaluation_checkpoint: Any = default_resolve_evaluation_checkpoint
    resolve_inference_checkpoint: Any = default_resolve_inference_checkpoint
    load_train_config: Any = default_load_train_config
    load_runtime_schema: Any = default_load_runtime_schema
    build_evaluation_data_diagnostics: Any = default_build_evaluation_data_diagnostics
    write_observed_schema_report: Any = default_write_observed_schema_report
    write_train_split_observed_schema_reports: Any = default_write_train_split_observed_schema_reports


DEFAULT_PCVR_RUNTIME_HOOKS = PCVRRuntimeHooks()


def build_pcvr_runtime_hooks(**overrides: Any) -> PCVRRuntimeHooks:
    return replace(DEFAULT_PCVR_RUNTIME_HOOKS, **overrides)


__all__ = [
    "DEFAULT_PCVR_RUNTIME_HOOKS",
    "PCVRRuntimeHooks",
    "build_pcvr_runtime_hooks",
    "default_build_evaluation_data_diagnostics",
    "default_load_runtime_schema",
    "default_load_train_config",
    "default_resolve_evaluation_checkpoint",
    "default_resolve_inference_checkpoint",
    "default_write_observed_schema_report",
    "default_write_train_split_observed_schema_reports",
]
