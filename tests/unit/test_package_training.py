from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

import taac2026.application.maintenance.package_training as package_training
from taac2026.application.maintenance.package_training import BundleResult, build_training_bundle


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


def test_package_training_main_prints_human_readable_summary_by_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output_dir = tmp_path / "bundle"
    result = BundleResult(
        output_dir=output_dir,
        run_script_path=output_dir / "run.sh",
        code_package_path=output_dir / "code_package.zip",
        manifest={
            "bundle_format": "taac2026-training-v2",
            "bundled_experiment_path": "config/baseline",
            "runtime_env": {
                "dataset_path": "TAAC_DATASET_PATH or TRAIN_DATA_PATH",
                "schema_path": "TAAC_SCHEMA_PATH",
                "checkpoint_path": "TAAC_OUTPUT_DIR or TRAIN_CKPT_PATH",
                "cuda_profile": "TAAC_CUDA_PROFILE",
            },
        },
    )
    monkeypatch.setattr(package_training, "build_training_bundle", lambda *args, **kwargs: result)

    exit_code = package_training.main(["--experiment", "config/baseline"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Built TAAC online training bundle" in captured.out
    assert "Experiment: config/baseline" in captured.out
    assert f"run.sh: {result.run_script_path}" in captured.out
    assert f"code_package.zip: {result.code_package_path}" in captured.out
    assert "Upload the two files above: run.sh and code_package.zip" in captured.out


def test_package_training_main_overwrites_existing_bundle_by_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_dir = tmp_path / "bundle"
    calls: list[dict[str, object]] = []
    result = BundleResult(
        output_dir=output_dir,
        run_script_path=output_dir / "run.sh",
        code_package_path=output_dir / "code_package.zip",
        manifest={
            "bundle_format": "taac2026-training-v2",
            "bundled_experiment_path": "config/baseline",
            "runtime_env": {},
        },
    )

    def fake_build_training_bundle(experiment: str, *, output_dir: Path | None = None, force: bool = False, root: Path | None = None):
        calls.append({
            "experiment": experiment,
            "output_dir": output_dir,
            "force": force,
            "root": root,
        })
        return result

    monkeypatch.setattr(package_training, "build_training_bundle", fake_build_training_bundle)

    exit_code = package_training.main(["--experiment", "config/baseline"])

    assert exit_code == 0
    assert calls == [{
        "experiment": "config/baseline",
        "output_dir": None,
        "force": True,
        "root": None,
    }]


def test_package_training_main_honors_no_force(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_dir = tmp_path / "bundle"
    calls: list[dict[str, object]] = []
    result = BundleResult(
        output_dir=output_dir,
        run_script_path=output_dir / "run.sh",
        code_package_path=output_dir / "code_package.zip",
        manifest={
            "bundle_format": "taac2026-training-v2",
            "bundled_experiment_path": "config/baseline",
            "runtime_env": {},
        },
    )

    def fake_build_training_bundle(experiment: str, *, output_dir: Path | None = None, force: bool = False, root: Path | None = None):
        calls.append({
            "experiment": experiment,
            "output_dir": output_dir,
            "force": force,
            "root": root,
        })
        return result

    monkeypatch.setattr(package_training, "build_training_bundle", fake_build_training_bundle)

    exit_code = package_training.main(["--experiment", "config/baseline", "--no-force"])

    assert exit_code == 0
    assert calls == [{
        "experiment": "config/baseline",
        "output_dir": None,
        "force": False,
        "root": None,
    }]


def test_package_training_main_prints_json_when_requested(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output_dir = tmp_path / "bundle"
    result = BundleResult(
        output_dir=output_dir,
        run_script_path=output_dir / "run.sh",
        code_package_path=output_dir / "code_package.zip",
        manifest={
            "bundle_format": "taac2026-training-v2",
            "bundled_experiment_path": "config/baseline",
            "runtime_env": {},
        },
    )
    monkeypatch.setattr(package_training, "build_training_bundle", lambda *args, **kwargs: result)

    exit_code = package_training.main(["--experiment", "config/baseline", "--json"])

    captured = capsys.readouterr()
    assert exit_code == 0
    payload = json.loads(captured.out)
    assert payload["output_dir"] == str(result.output_dir)
    assert payload["run_script_path"] == str(result.run_script_path)
    assert payload["code_package_path"] == str(result.code_package_path)
    assert payload["manifest"]["bundled_experiment_path"] == "config/baseline"
