"""Platform-compatible inference entrypoint."""

from __future__ import annotations

import os
from pathlib import Path

from taac2026.application.evaluation.cli import main as evaluation_main


def _read_optional_bool_env(name: str) -> bool | None:
    value = os.environ.get(name)
    if value is None:
        return None
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise RuntimeError(f"{name} must be one of 1/0/true/false/yes/no/on/off")


def main() -> None:
    dataset_path = os.environ.get("EVAL_DATA_PATH")
    result_path = os.environ.get("EVAL_RESULT_PATH")
    model_path = os.environ.get("MODEL_OUTPUT_PATH")
    schema_path = os.environ.get("TAAC_SCHEMA_PATH")
    experiment_path = os.environ.get("TAAC_EXPERIMENT")
    infer_batch_size = os.environ.get("TAAC_INFER_BATCH_SIZE")
    infer_num_workers = os.environ.get("TAAC_INFER_NUM_WORKERS")
    infer_amp = _read_optional_bool_env("TAAC_INFER_AMP")
    infer_amp_dtype = os.environ.get("TAAC_INFER_AMP_DTYPE")
    infer_compile = _read_optional_bool_env("TAAC_INFER_COMPILE")
    if not dataset_path:
        raise RuntimeError("EVAL_DATA_PATH is required")
    if not result_path:
        raise RuntimeError("EVAL_RESULT_PATH is required")
    argv = [
        "infer",
    ]
    if experiment_path:
        argv.extend(["--experiment", experiment_path])
    argv.extend([
        "--dataset-path",
        dataset_path,
        "--result-dir",
        result_path,
    ])
    if model_path:
        argv.extend(["--checkpoint", str(Path(model_path))])
    if schema_path:
        argv.extend(["--schema-path", schema_path])
    if infer_batch_size:
        argv.extend(["--batch-size", infer_batch_size])
    if infer_num_workers:
        argv.extend(["--num-workers", infer_num_workers])
    if infer_amp is True:
        argv.append("--amp")
    elif infer_amp is False:
        argv.append("--no-amp")
    if infer_amp_dtype:
        argv.extend(["--amp-dtype", infer_amp_dtype])
    if infer_compile is True:
        argv.append("--compile")
    elif infer_compile is False:
        argv.append("--no-compile")
    evaluation_main(argv)


if __name__ == "__main__":
    main()
