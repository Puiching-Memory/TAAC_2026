from __future__ import annotations

from pathlib import Path

import pytest

from taac2026.infrastructure.bundles.manifest_store import (
    BUNDLE_MANIFEST_VERSION,
    build_bundle_manifest,
    validate_bundle_manifest,
)


def test_build_training_bundle_manifest_contains_versioned_contract(tmp_path: Path) -> None:
    experiment_path = tmp_path / "experiments" / "baseline"
    experiment_path.mkdir(parents=True)

    manifest = build_bundle_manifest(kind="training", experiment_path=experiment_path, root=tmp_path)

    assert manifest["manifest_version"] == BUNDLE_MANIFEST_VERSION
    assert manifest["bundle_kind"] == "training"
    assert manifest["bundle_format"] == "taac2026-training-v2"
    assert manifest["bundle_format_version"] == 2
    assert manifest["framework"]["version"]
    assert manifest["bundled_experiment_path"] == "experiments/baseline"
    assert manifest["entrypoint"] == "run.sh"
    assert manifest["runtime_env"]["dataset_path"] == "TRAIN_DATA_PATH"
    assert manifest["compatibility"]["online_runner"] == "python"


def test_build_inference_bundle_manifest_contains_runtime_variables(tmp_path: Path) -> None:
    experiment_path = tmp_path / "experiments" / "baseline"
    experiment_path.mkdir(parents=True)

    manifest = build_bundle_manifest(kind="inference", experiment_path=experiment_path, root=tmp_path)

    assert manifest["bundle_kind"] == "inference"
    assert manifest["bundle_format"] == "taac2026-inference-v1"
    assert manifest["entrypoint"] == "infer.py"
    assert manifest["runtime_env"]["model_path"] == "MODEL_OUTPUT_PATH"
    assert manifest["runtime_env"]["result_path"] == "EVAL_RESULT_PATH"


def test_validate_bundle_manifest_rejects_missing_runtime_env_key(tmp_path: Path) -> None:
    experiment_path = tmp_path / "experiments" / "baseline"
    experiment_path.mkdir(parents=True)
    manifest = build_bundle_manifest(kind="training", experiment_path=experiment_path, root=tmp_path)
    manifest["runtime_env"] = {}

    with pytest.raises(KeyError, match="checkpoint_path"):
        validate_bundle_manifest(manifest, kind="training")


def test_validate_bundle_manifest_rejects_kind_mismatch(tmp_path: Path) -> None:
    experiment_path = tmp_path / "experiments" / "baseline"
    experiment_path.mkdir(parents=True)
    manifest = build_bundle_manifest(kind="training", experiment_path=experiment_path, root=tmp_path)

    with pytest.raises(ValueError, match="kind mismatch"):
        validate_bundle_manifest(manifest, kind="inference")
