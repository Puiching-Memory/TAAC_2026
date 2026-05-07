from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from taac2026.application.packaging.cli import build_training_bundle
from taac2026.infrastructure.io.json import loads
from tests.unit.application.packaging._bundle_test_support import (
    assert_pip_install_args,
    code_package_manifest,
    code_package_names,
    write_minimal_eval_runtime_package,
    write_minimal_training_runtime_package,
    write_fake_pip_package,
)
from tests.unit.experiments._experiment_matrix import discover_nonbaseline_pcvr_experiment_paths


NON_BASELINE_EXPERIMENTS = discover_nonbaseline_pcvr_experiment_paths()

def test_build_training_bundle_contains_runtime_sources(tmp_path: Path) -> None:
    output_dir = tmp_path / "baseline_bundle"

    result = build_training_bundle("experiments/baseline", output_dir=output_dir)

    assert result.output_dir == output_dir.resolve()
    assert result.run_script_path == output_dir.resolve() / "run.sh"
    assert result.code_package_path == output_dir.resolve() / "code_package.zip"
    assert sorted(path.name for path in output_dir.iterdir()) == ["code_package.zip", "run.sh"]
    assert result.run_script_path.exists()
    assert result.code_package_path.exists()
    run_script = result.run_script_path.read_text(encoding="utf-8")
    assert "TAAC_INSTALL_PROJECT_DEPS" in run_script
    assert "TAAC_BUNDLE_PIP_EXTRAS" in run_script
    assert "taac-train" in run_script

    manifest = code_package_manifest(result.code_package_path, ".taac_training_manifest.json")
    assert manifest["manifest_version"] == 1
    assert manifest["bundle_kind"] == "training"
    assert manifest["bundle_format"] == "taac2026-training-v2"
    assert manifest["bundle_format_version"] == 2
    assert manifest["framework"]["name"] == "taac2026"
    assert manifest["framework"]["version"]
    assert manifest["bundled_experiment_path"] == "experiments/baseline"
    assert manifest["entrypoint"] == "run.sh"
    assert manifest["code_package"] == "code_package.zip"
    assert manifest["runtime_env"]["pip_extras"].startswith("TAAC_BUNDLE_PIP_EXTRAS")
    assert manifest["compatibility"]["requires_uv_online"] is False

    names = code_package_names(result.code_package_path)
    assert "project/.taac_training_manifest.json" in names
    assert "project/pyproject.toml" in names
    assert "project/uv.lock" not in names
    assert "project/README.md" not in names
    assert "project/src/taac2026/application/training/cli.py" in names
    assert "project/src/taac2026/application/training/args.py" in names
    assert "project/src/taac2026/application/training/workflow.py" in names
    assert "project/src/taac2026/infrastructure/data/native/opt_cache.cpp" in names
    assert "project/src/taac2026/infrastructure/data/native/opt_cache.py" in names
    assert "project/src/taac2026/infrastructure/runtime/trainer.py" in names
    assert "project/experiments/baseline/__init__.py" in names
    assert "project/experiments/baseline/model.py" in names
    assert "project/experiments/baseline/ns_groups.json" not in names
    assert "project/run.sh" not in names
    assert "project/experiments/baseline/train.py" not in names
    assert "project/experiments/baseline/trainer.py" not in names
    assert "project/experiments/baseline/run.sh" not in names
    assert "project/tests/unit/test_package_training.py" not in names


@pytest.mark.parametrize(
    "experiment",
    NON_BASELINE_EXPERIMENTS,
)
def test_build_training_bundle_contains_experiment_ns_groups(tmp_path: Path, experiment: str) -> None:
    output_dir = tmp_path / f"{Path(experiment).name}_bundle"

    result = build_training_bundle(experiment, output_dir=output_dir)

    names = code_package_names(result.code_package_path)
    assert f"project/{experiment}/__init__.py" in names
    assert f"project/{experiment}/model.py" in names
    assert f"project/{experiment}/ns_groups.json" not in names


@pytest.mark.parametrize("experiment", ["experiments/host_device_info", "experiments/online_dataset_eda"])
def test_build_training_bundle_supports_maintenance_experiment_packages(tmp_path: Path, experiment: str) -> None:
    output_dir = tmp_path / f"{Path(experiment).name}_bundle"

    result = build_training_bundle(experiment, output_dir=output_dir)

    manifest = code_package_manifest(result.code_package_path, ".taac_training_manifest.json")
    names = code_package_names(result.code_package_path)
    assert manifest["bundled_experiment_path"] == experiment
    assert f"project/{experiment}/__init__.py" in names


def test_build_training_bundle_refuses_overwrite_without_force(tmp_path: Path) -> None:
    output_dir = tmp_path / "baseline_bundle"
    build_training_bundle("experiments/baseline", output_dir=output_dir)

    with pytest.raises(FileExistsError):
        build_training_bundle("experiments/baseline", output_dir=output_dir)


def test_build_training_bundle_force_replaces_two_file_output(tmp_path: Path) -> None:
    output_dir = tmp_path / "baseline_bundle"
    build_training_bundle("experiments/baseline", output_dir=output_dir)
    (output_dir / "run.sh").write_text("stale\n", encoding="utf-8")

    result = build_training_bundle("experiments/baseline", output_dir=output_dir, force=True)

    assert result.run_script_path.read_text(encoding="utf-8").startswith("#!/usr/bin/env bash")
    assert result.code_package_path.exists()


def test_training_run_script_installs_project_dependencies_before_entrypoint(tmp_path: Path) -> None:
    output_dir = tmp_path / "baseline_bundle"
    result = build_training_bundle("experiments/baseline", output_dir=output_dir)
    write_minimal_training_runtime_package(result.code_package_path)
    pip_args_path = tmp_path / "pip_args.json"
    fake_pip = write_fake_pip_package(tmp_path, pip_args_path)

    env = os.environ.copy()
    for variable in (
        "TAAC_BUNDLE_WORKDIR",
        "TAAC_CODE_PACKAGE",
        "TAAC_EXPERIMENT",
        "TAAC_INSTALL_PROJECT_DEPS",
        "TAAC_BUNDLE_PIP_EXTRAS",
        "TAAC_PIP_EXTRA_ARGS",
        "TAAC_PIP_EXTRAS",
        "TAAC_PIP_INDEX_URL",
        "TAAC_PYTHON",
        "TAAC_RUNNER",
        "TAAC_SKIP_PIP_INSTALL",
    ):
        env.pop(variable, None)
    env.update(
        {
            "TAAC_BUNDLE_WORKDIR": str(tmp_path / "bundle_workdir"),
            "TAAC_PIP_EXTRAS": "dev",
            "TAAC_PIP_EXTRA_ARGS": "-q",
            "TAAC_PIP_INDEX_URL": "",
            "TAAC_PYTHON": sys.executable,
            "TAAC_RUNNER": "python",
            "PYTHONPATH": str(fake_pip),
        }
    )
    completed = subprocess.run(
        ["bash", str(result.run_script_path), "--device", "cpu"],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )

    payload = loads(completed.stdout)
    pip_args = loads(pip_args_path.read_bytes())

    assert payload["cwd"].endswith("bundle_workdir/project")
    assert payload["experiment"] is None
    assert payload["argv"][:2] == ["--experiment", "experiments/minimal"]
    assert_pip_install_args(pip_args, expected_target=".")
    assert "Installing TAAC project dependencies from pyproject.toml" in completed.stderr


def test_training_run_script_accepts_explicit_bundle_pip_extras(tmp_path: Path) -> None:
    output_dir = tmp_path / "baseline_bundle"
    result = build_training_bundle("experiments/baseline", output_dir=output_dir)
    write_minimal_training_runtime_package(result.code_package_path)
    pip_args_path = tmp_path / "pip_args.json"
    fake_pip = write_fake_pip_package(tmp_path, pip_args_path)

    env = os.environ.copy()
    for variable in (
        "TAAC_BUNDLE_WORKDIR",
        "TAAC_CODE_PACKAGE",
        "TAAC_EXPERIMENT",
        "TAAC_INSTALL_PROJECT_DEPS",
        "TAAC_BUNDLE_PIP_EXTRAS",
        "TAAC_PIP_EXTRA_ARGS",
        "TAAC_PIP_EXTRAS",
        "TAAC_PIP_INDEX_URL",
        "TAAC_PYTHON",
        "TAAC_RUNNER",
        "TAAC_SKIP_PIP_INSTALL",
    ):
        env.pop(variable, None)
    env.update(
        {
            "TAAC_BUNDLE_WORKDIR": str(tmp_path / "bundle_workdir"),
            "TAAC_BUNDLE_PIP_EXTRAS": "dev",
            "TAAC_PIP_EXTRA_ARGS": "-q",
            "TAAC_PIP_INDEX_URL": "",
            "TAAC_PYTHON": sys.executable,
            "TAAC_RUNNER": "python",
            "PYTHONPATH": str(fake_pip),
        }
    )

    subprocess.run(
        ["bash", str(result.run_script_path), "--device", "cpu"],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )

    pip_args = loads(pip_args_path.read_bytes())

    assert_pip_install_args(pip_args, expected_target=".[dev]")


def test_training_run_script_reextracts_when_code_package_changes(tmp_path: Path) -> None:
    output_dir = tmp_path / "baseline_bundle"
    result = build_training_bundle("experiments/baseline", output_dir=output_dir)
    write_minimal_training_runtime_package(result.code_package_path, bundled_experiment_path="experiments/first")

    env = os.environ.copy()
    for variable in (
        "TAAC_BUNDLE_WORKDIR",
        "TAAC_CODE_PACKAGE",
        "TAAC_EXPERIMENT",
        "TAAC_INSTALL_PROJECT_DEPS",
        "TAAC_BUNDLE_PIP_EXTRAS",
        "TAAC_PIP_EXTRA_ARGS",
        "TAAC_PIP_EXTRAS",
        "TAAC_PIP_INDEX_URL",
        "TAAC_PYTHON",
        "TAAC_RUNNER",
        "TAAC_SKIP_PIP_INSTALL",
    ):
        env.pop(variable, None)
    env.update(
        {
            "TAAC_BUNDLE_WORKDIR": str(tmp_path / "bundle_workdir"),
            "TAAC_PYTHON": sys.executable,
            "TAAC_RUNNER": "python",
            "TAAC_SKIP_PIP_INSTALL": "1",
        }
    )

    first = subprocess.run(
        ["bash", str(result.run_script_path), "--device", "cpu"],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )

    write_minimal_training_runtime_package(result.code_package_path, bundled_experiment_path="experiments/second")
    second = subprocess.run(
        ["bash", str(result.run_script_path), "--device", "cpu"],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )

    assert loads(first.stdout)["argv"][:2] == ["--experiment", "experiments/first"]
    assert loads(second.stdout)["argv"][:2] == ["--experiment", "experiments/second"]


def test_training_run_script_does_not_inject_default_experiment(tmp_path: Path) -> None:
    output_dir = tmp_path / "baseline_bundle"
    result = build_training_bundle("experiments/baseline", output_dir=output_dir)
    write_minimal_training_runtime_package(result.code_package_path, bundled_experiment_path=None)
    pip_args_path = tmp_path / "pip_args.json"
    fake_pip = write_fake_pip_package(tmp_path, pip_args_path)

    env = os.environ.copy()
    for variable in (
        "TAAC_BUNDLE_WORKDIR",
        "TAAC_CODE_PACKAGE",
        "TAAC_EXPERIMENT",
        "TAAC_INSTALL_PROJECT_DEPS",
        "TAAC_BUNDLE_PIP_EXTRAS",
        "TAAC_PIP_EXTRA_ARGS",
        "TAAC_PIP_EXTRAS",
        "TAAC_PIP_INDEX_URL",
        "TAAC_PYTHON",
        "TAAC_RUNNER",
        "TAAC_SKIP_PIP_INSTALL",
    ):
        env.pop(variable, None)
    env.update(
        {
            "TAAC_BUNDLE_WORKDIR": str(tmp_path / "bundle_workdir"),
            "TAAC_PIP_EXTRA_ARGS": "-q",
            "TAAC_PIP_INDEX_URL": "",
            "TAAC_PYTHON": sys.executable,
            "TAAC_RUNNER": "python",
            "PYTHONPATH": str(fake_pip),
        }
    )
    completed = subprocess.run(
        ["bash", str(result.run_script_path), "--device", "cpu"],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )

    payload = loads(completed.stdout)

    assert payload["experiment"] is None
    assert payload["argv"] == ["--device", "cpu"]


def test_training_run_script_uses_platform_train_env_paths(tmp_path: Path) -> None:
    output_dir = tmp_path / "baseline_bundle"
    result = build_training_bundle("experiments/baseline", output_dir=output_dir)
    write_minimal_training_runtime_package(result.code_package_path)
    pip_args_path = tmp_path / "pip_args.json"
    fake_pip = write_fake_pip_package(tmp_path, pip_args_path)

    env = os.environ.copy()
    for variable in (
        "TAAC_BUNDLE_WORKDIR",
        "TAAC_CODE_PACKAGE",
        "TAAC_EXPERIMENT",
        "TAAC_INSTALL_PROJECT_DEPS",
        "TAAC_BUNDLE_PIP_EXTRAS",
        "TAAC_PIP_EXTRA_ARGS",
        "TAAC_PIP_EXTRAS",
        "TAAC_PIP_INDEX_URL",
        "TAAC_PYTHON",
        "TAAC_RUNNER",
        "TAAC_SKIP_PIP_INSTALL",
        "TAAC_DATASET_PATH",
        "TAAC_OUTPUT_DIR",
        "TRAIN_DATA_PATH",
        "TRAIN_CKPT_PATH",
    ):
        env.pop(variable, None)
    env.update(
        {
            "TAAC_BUNDLE_WORKDIR": str(tmp_path / "bundle_workdir"),
            "TAAC_PIP_EXTRA_ARGS": "-q",
            "TAAC_PIP_INDEX_URL": "",
            "TAAC_PYTHON": sys.executable,
            "TAAC_RUNNER": "python",
            "TRAIN_DATA_PATH": "/platform/train.parquet",
            "TAAC_SCHEMA_PATH": "/platform/schema.json",
            "TRAIN_CKPT_PATH": "/platform/output",
            "PYTHONPATH": str(fake_pip),
        }
    )
    completed = subprocess.run(
        ["bash", str(result.run_script_path), "--device", "cpu"],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )

    payload = loads(completed.stdout)

    assert payload["argv"][:6] == [
        "--experiment",
        "experiments/minimal",
        "--dataset-path",
        "/platform/train.parquet",
        "--schema-path",
        "/platform/schema.json",
    ]
    assert "--run-dir" in payload["argv"]
    assert "/platform/output" in payload["argv"]


def test_training_run_script_infer_uses_platform_eval_env_paths(tmp_path: Path) -> None:
    output_dir = tmp_path / "baseline_bundle"
    result = build_training_bundle("experiments/baseline", output_dir=output_dir)
    write_minimal_eval_runtime_package(result.code_package_path)

    env = os.environ.copy()
    for variable in (
        "TAAC_BUNDLE_WORKDIR",
        "TAAC_CODE_PACKAGE",
        "TAAC_EXPERIMENT",
        "TAAC_INSTALL_PROJECT_DEPS",
        "TAAC_BUNDLE_PIP_EXTRAS",
        "TAAC_PIP_EXTRA_ARGS",
        "TAAC_PIP_EXTRAS",
        "TAAC_PIP_INDEX_URL",
        "TAAC_PYTHON",
        "TAAC_RUNNER",
        "TAAC_SKIP_PIP_INSTALL",
        "TAAC_DATASET_PATH",
        "TAAC_RESULT_DIR",
        "EVAL_DATA_PATH",
        "EVAL_RESULT_PATH",
        "MODEL_OUTPUT_PATH",
    ):
        env.pop(variable, None)
    env.update(
        {
            "TAAC_BUNDLE_WORKDIR": str(tmp_path / "bundle_workdir"),
            "TAAC_PYTHON": sys.executable,
            "TAAC_RUNNER": "python",
            "TAAC_SKIP_PIP_INSTALL": "1",
            "EVAL_DATA_PATH": "/platform/eval.parquet",
            "TAAC_SCHEMA_PATH": "/platform/schema.json",
            "EVAL_RESULT_PATH": "/platform/results",
            "MODEL_OUTPUT_PATH": "/platform/model.safetensors",
        }
    )
    completed = subprocess.run(
        ["bash", str(result.run_script_path), "infer"],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )

    payload = loads(completed.stdout)

    assert payload["argv"] == [
        "infer",
        "--experiment",
        "experiments/minimal",
        "--dataset-path",
        "/platform/eval.parquet",
        "--schema-path",
        "/platform/schema.json",
        "--result-dir",
        "/platform/results",
        "--checkpoint",
        "/platform/model.safetensors",
    ]
