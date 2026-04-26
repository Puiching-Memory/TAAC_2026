from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from taac2026.application.maintenance.package_training import build_training_bundle


def _code_package_names(code_package_path: Path) -> set[str]:
    with zipfile.ZipFile(code_package_path) as code_archive:
        return set(code_archive.namelist())


def _code_package_manifest(code_package_path: Path) -> dict[str, object]:
    with zipfile.ZipFile(code_package_path) as code_archive:
        payload = code_archive.read("project/.taac_training_manifest.json")
    return json.loads(payload.decode("utf-8"))


def test_build_training_bundle_contains_runtime_sources(tmp_path: Path) -> None:
    output_dir = tmp_path / "baseline_bundle"

    result = build_training_bundle("config/baseline", output_dir=output_dir)

    assert result.output_dir == output_dir.resolve()
    assert result.run_script_path == output_dir.resolve() / "run.sh"
    assert result.code_package_path == output_dir.resolve() / "code_package.zip"
    assert sorted(path.name for path in output_dir.iterdir()) == ["code_package.zip", "run.sh"]
    assert result.run_script_path.exists()
    assert result.code_package_path.exists()
    run_script = result.run_script_path.read_text(encoding="utf-8")
    assert "RUNNER_MODE=\"python\"" in run_script
    assert "python -m taac2026.application.training.cli" not in run_script
    assert "run_console_script taac-train taac2026.application.training.cli" in run_script

    manifest = _code_package_manifest(result.code_package_path)
    assert manifest["bundle_format"] == "taac2026-training-v2"
    assert manifest["bundled_experiment_path"] == "config/baseline"
    assert manifest["entrypoint"] == "run.sh"
    assert manifest["code_package"] == "code_package.zip"

    names = _code_package_names(result.code_package_path)
    assert "project/.taac_training_manifest.json" in names
    assert "project/pyproject.toml" in names
    assert "project/uv.lock" in names
    assert "project/src/taac2026/application/training/cli.py" in names
    assert "project/src/taac2026/infrastructure/pcvr/training.py" in names
    assert "project/src/taac2026/infrastructure/pcvr/trainer.py" in names
    assert "project/config/baseline/model.py" in names
    assert "project/config/baseline/ns_groups.json" in names
    assert "project/run.sh" not in names
    assert "project/config/baseline/train.py" not in names
    assert "project/config/baseline/trainer.py" not in names
    assert "project/config/baseline/run.sh" not in names
    assert "project/tests/unit/test_package_training.py" not in names


@pytest.mark.parametrize(
    "experiment",
    [
        "config/symbiosis",
        "config/ctr_baseline",
        "config/deepcontextnet",
        "config/interformer",
        "config/onetrans",
        "config/hyformer",
        "config/unirec",
        "config/uniscaleformer",
    ],
)
def test_build_training_bundle_contains_experiment_ns_groups(tmp_path: Path, experiment: str) -> None:
    output_dir = tmp_path / f"{Path(experiment).name}_bundle"

    result = build_training_bundle(experiment, output_dir=output_dir)

    names = _code_package_names(result.code_package_path)
    assert f"project/{experiment}/model.py" in names
    assert f"project/{experiment}/ns_groups.json" in names


def test_build_training_bundle_refuses_overwrite_without_force(tmp_path: Path) -> None:
    output_dir = tmp_path / "baseline_bundle"
    build_training_bundle("config/baseline", output_dir=output_dir)

    with pytest.raises(FileExistsError):
        build_training_bundle("config/baseline", output_dir=output_dir)


def test_build_training_bundle_force_replaces_two_file_output(tmp_path: Path) -> None:
    output_dir = tmp_path / "baseline_bundle"
    build_training_bundle("config/baseline", output_dir=output_dir)
    (output_dir / "run.sh").write_text("stale\n", encoding="utf-8")

    result = build_training_bundle("config/baseline", output_dir=output_dir, force=True)

    assert result.run_script_path.read_text(encoding="utf-8").startswith("#!/usr/bin/env bash")
    assert result.code_package_path.exists()
