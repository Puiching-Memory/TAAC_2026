from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from taac2026.application.maintenance.bundle_packaging import build_inference_bundle
from taac2026.infrastructure.io.json_utils import loads
from tests.unit.application._bundle_test_support import (
    assert_pip_install_args,
    code_package_manifest,
    code_package_names,
    write_minimal_inference_runtime_package,
    write_fake_pip_package,
)
from tests.unit.infrastructure.pcvr._pcvr_experiment_matrix import discover_nonbaseline_pcvr_experiment_paths


NON_BASELINE_EXPERIMENTS = discover_nonbaseline_pcvr_experiment_paths()

def test_build_inference_bundle_contains_runtime_sources(tmp_path: Path) -> None:
    output_dir = tmp_path / "baseline_bundle"

    result = build_inference_bundle("experiments/pcvr/baseline", output_dir=output_dir)

    assert result.output_dir == output_dir.resolve()
    assert result.run_script_path == output_dir.resolve() / "infer.py"
    assert result.code_package_path == output_dir.resolve() / "code_package.zip"
    assert sorted(path.name for path in output_dir.iterdir()) == ["code_package.zip", "infer.py"]
    assert result.run_script_path.exists()
    assert result.code_package_path.exists()
    infer_script = result.run_script_path.read_text(encoding="utf-8")
    assert "TAAC_INSTALL_PROJECT_DEPS" in infer_script
    assert "TAAC_BUNDLE_PIP_EXTRAS" in infer_script
    assert "run_inference_bundle" in infer_script

    manifest = code_package_manifest(result.code_package_path, ".taac_inference_manifest.json")
    assert manifest["manifest_version"] == 1
    assert manifest["bundle_kind"] == "inference"
    assert manifest["bundle_format"] == "taac2026-inference-v1"
    assert manifest["bundle_format_version"] == 1
    assert manifest["framework"]["name"] == "taac2026"
    assert manifest["framework"]["version"]
    assert manifest["bundled_experiment_path"] == "experiments/pcvr/baseline"
    assert manifest["entrypoint"] == "infer.py"
    assert manifest["code_package"] == "code_package.zip"
    assert manifest["runtime_env"]["pip_extras"].startswith("TAAC_BUNDLE_PIP_EXTRAS")
    assert manifest["compatibility"]["requires_uv_online"] is False

    names = code_package_names(result.code_package_path)
    assert "project/.taac_inference_manifest.json" in names
    assert "project/pyproject.toml" in names
    assert "project/uv.lock" not in names
    assert "project/README.md" not in names
    assert "project/src/taac2026/application/evaluation/cli.py" in names
    assert "project/src/taac2026/application/evaluation/infer.py" in names
    assert "project/experiments/pcvr/baseline/__init__.py" in names
    assert "project/experiments/pcvr/baseline/model.py" in names
    assert "project/experiments/pcvr/baseline/ns_groups.json" not in names
    assert "project/infer.py" not in names
    assert "project/run.sh" not in names
    assert "project/tests/unit/test_package_inference.py" not in names


@pytest.mark.parametrize(
    "experiment",
    NON_BASELINE_EXPERIMENTS,
)
def test_build_inference_bundle_contains_experiment_ns_groups(tmp_path: Path, experiment: str) -> None:
    output_dir = tmp_path / f"{Path(experiment).name}_bundle"

    result = build_inference_bundle(experiment, output_dir=output_dir)

    names = code_package_names(result.code_package_path)
    assert f"project/{experiment}/__init__.py" in names
    assert f"project/{experiment}/model.py" in names
    assert f"project/{experiment}/ns_groups.json" not in names


def test_build_inference_bundle_refuses_overwrite_without_force(tmp_path: Path) -> None:
    output_dir = tmp_path / "baseline_bundle"
    build_inference_bundle("experiments/pcvr/baseline", output_dir=output_dir)

    with pytest.raises(FileExistsError):
        build_inference_bundle("experiments/pcvr/baseline", output_dir=output_dir)


def test_build_inference_bundle_force_replaces_two_file_output(tmp_path: Path) -> None:
    output_dir = tmp_path / "baseline_bundle"
    build_inference_bundle("experiments/pcvr/baseline", output_dir=output_dir)
    (output_dir / "infer.py").write_text("stale\n", encoding="utf-8")

    result = build_inference_bundle("experiments/pcvr/baseline", output_dir=output_dir, force=True)

    assert result.run_script_path.read_text(encoding="utf-8").startswith("#!/usr/bin/env python3")
    assert result.code_package_path.exists()


def test_generated_infer_script_requires_user_cache_path_without_workdir_override(tmp_path: Path) -> None:
    output_dir = tmp_path / "baseline_bundle"
    result = build_inference_bundle("experiments/pcvr/baseline", output_dir=output_dir)
    write_minimal_inference_runtime_package(result.code_package_path)

    env = os.environ.copy()
    for variable in (
        "TAAC_BUNDLE_WORKDIR",
        "TAAC_CODE_PACKAGE",
        "TAAC_EXPERIMENT",
        "TAAC_FORCE_EXTRACT",
        "TAAC_INSTALL_PROJECT_DEPS",
        "TAAC_BUNDLE_PIP_EXTRAS",
        "TAAC_PIP_EXTRA_ARGS",
        "TAAC_PIP_EXTRAS",
        "TAAC_PIP_INDEX_URL",
        "TAAC_SKIP_PIP_INSTALL",
        "USER_CACHE_PATH",
    ):
        env.pop(variable, None)
    completed = subprocess.run(
        [sys.executable, str(result.run_script_path)],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    assert completed.returncode != 0
    assert completed.stdout == ""
    assert "USER_CACHE_PATH is required unless TAAC_BUNDLE_WORKDIR is set" in completed.stderr


def test_generated_infer_script_prefers_user_cache_path_when_available(tmp_path: Path) -> None:
    output_dir = tmp_path / "baseline_bundle"
    user_cache_path = tmp_path / "user_cache"
    result = build_inference_bundle("experiments/pcvr/baseline", output_dir=output_dir)
    write_minimal_inference_runtime_package(result.code_package_path)

    env = os.environ.copy()
    for variable in (
        "TAAC_BUNDLE_WORKDIR",
        "TAAC_CODE_PACKAGE",
        "TAAC_EXPERIMENT",
        "TAAC_FORCE_EXTRACT",
        "TAAC_INSTALL_PROJECT_DEPS",
        "TAAC_BUNDLE_PIP_EXTRAS",
        "TAAC_PIP_EXTRA_ARGS",
        "TAAC_PIP_EXTRAS",
        "TAAC_PIP_INDEX_URL",
        "TAAC_SKIP_PIP_INSTALL",
    ):
        env.pop(variable, None)
    env["USER_CACHE_PATH"] = str(user_cache_path)
    env["TAAC_SKIP_PIP_INSTALL"] = "1"
    completed = subprocess.run(
        [sys.executable, str(result.run_script_path)],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )

    payload = loads(completed.stdout)
    extracted_project_dir = Path(payload["cwd"]).resolve()

    assert payload["experiment"] == "experiments/pcvr/minimal"
    assert extracted_project_dir.name == "project"
    assert extracted_project_dir.is_relative_to(user_cache_path.resolve())
    assert not extracted_project_dir.is_relative_to(output_dir.resolve())


def test_generated_infer_script_installs_project_dependencies_before_entrypoint(tmp_path: Path) -> None:
    output_dir = tmp_path / "baseline_bundle"
    user_cache_path = tmp_path / "user_cache"
    result = build_inference_bundle("experiments/pcvr/baseline", output_dir=output_dir)
    write_minimal_inference_runtime_package(result.code_package_path)
    pip_args_path = tmp_path / "pip_args.json"
    fake_pip = write_fake_pip_package(tmp_path, pip_args_path)

    env = os.environ.copy()
    for variable in (
        "TAAC_BUNDLE_WORKDIR",
        "TAAC_CODE_PACKAGE",
        "TAAC_EXPERIMENT",
        "TAAC_FORCE_EXTRACT",
        "TAAC_INSTALL_PROJECT_DEPS",
        "TAAC_BUNDLE_PIP_EXTRAS",
        "TAAC_PIP_EXTRA_ARGS",
        "TAAC_PIP_EXTRAS",
        "TAAC_PIP_INDEX_URL",
        "TAAC_SKIP_PIP_INSTALL",
    ):
        env.pop(variable, None)
    env.update(
        {
            "TAAC_PIP_EXTRAS": "dev",
            "TAAC_PIP_EXTRA_ARGS": "-q",
            "TAAC_PIP_INDEX_URL": "",
            "PYTHONPATH": str(fake_pip),
            "USER_CACHE_PATH": str(user_cache_path),
        }
    )
    completed = subprocess.run(
        [sys.executable, str(result.run_script_path)],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )

    payload = loads(completed.stdout)
    pip_args = loads(pip_args_path.read_bytes())

    assert payload["experiment"] == "experiments/pcvr/minimal"
    assert Path(payload["cwd"]).resolve().is_relative_to(user_cache_path.resolve())
    assert_pip_install_args(pip_args, expected_target=".")
    assert "Installing TAAC project dependencies from pyproject.toml" in completed.stderr


def test_generated_infer_script_accepts_explicit_bundle_pip_extras(tmp_path: Path) -> None:
    output_dir = tmp_path / "baseline_bundle"
    user_cache_path = tmp_path / "user_cache"
    result = build_inference_bundle("experiments/pcvr/baseline", output_dir=output_dir)
    write_minimal_inference_runtime_package(result.code_package_path)
    pip_args_path = tmp_path / "pip_args.json"
    fake_pip = write_fake_pip_package(tmp_path, pip_args_path)

    env = os.environ.copy()
    for variable in (
        "TAAC_BUNDLE_WORKDIR",
        "TAAC_CODE_PACKAGE",
        "TAAC_EXPERIMENT",
        "TAAC_FORCE_EXTRACT",
        "TAAC_INSTALL_PROJECT_DEPS",
        "TAAC_BUNDLE_PIP_EXTRAS",
        "TAAC_PIP_EXTRA_ARGS",
        "TAAC_PIP_EXTRAS",
        "TAAC_PIP_INDEX_URL",
        "TAAC_SKIP_PIP_INSTALL",
    ):
        env.pop(variable, None)
    env.update(
        {
            "TAAC_BUNDLE_PIP_EXTRAS": "dev",
            "TAAC_PIP_EXTRA_ARGS": "-q",
            "TAAC_PIP_INDEX_URL": "",
            "PYTHONPATH": str(fake_pip),
            "USER_CACHE_PATH": str(user_cache_path),
        }
    )
    subprocess.run(
        [sys.executable, str(result.run_script_path)],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )

    pip_args = loads(pip_args_path.read_bytes())

    assert_pip_install_args(pip_args, expected_target=".[dev]")