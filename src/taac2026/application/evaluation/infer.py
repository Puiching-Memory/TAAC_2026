"""Platform-compatible inference entrypoint."""

from __future__ import annotations

import os
from pathlib import Path

from taac2026.application.evaluation.cli import main as evaluation_main


def main() -> None:
    dataset_path = os.environ.get("EVAL_DATA_PATH")
    result_path = os.environ.get("EVAL_RESULT_PATH")
    model_path = os.environ.get("MODEL_OUTPUT_PATH")
    schema_path = os.environ.get("TAAC_SCHEMA_PATH")
    experiment_path = os.environ.get("TAAC_EXPERIMENT")
    infer_batch_size = os.environ.get("TAAC_INFER_BATCH_SIZE")
    infer_num_workers = os.environ.get("TAAC_INFER_NUM_WORKERS")
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
    evaluation_main(argv)


if __name__ == "__main__":
    main()
