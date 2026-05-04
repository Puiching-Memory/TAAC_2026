from __future__ import annotations

import os
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

import taac2026.application.maintenance.package_training as package_training
from taac2026.application.maintenance.package_training import BundleResult, build_training_bundle
from taac2026.infrastructure.io.json_utils import dump_bytes, loads
from tests.unit.infrastructure.pcvr._pcvr_experiment_matrix import discover_nonbaseline_pcvr_experiment_paths


NON_BASELINE_EXPERIMENTS = discover_nonbaseline_pcvr_experiment_paths()


def _code_package_names(code_package_path: Path) -> set[str]:
    with zipfile.ZipFile(code_package_path) as code_archive:
        return set(code_archive.namelist())


def _code_package_manifest(code_package_path: Path) -> dict[str, object]:
    with zipfile.ZipFile(code_package_path) as code_archive:
        payload = code_archive.read("project/.taac_training_manifest.json")
    return loads(payload)


def _write_minimal_training_runtime_package(code_package_path: Path, *, bundled_experiment_path: str | None = "experiments/pcvr/minimal") -> None:
    with zipfile.ZipFile(code_package_path, mode="w", compression=zipfile.ZIP_DEFLATED) as code_archive:
        manifest: dict[str, object] = {}
        if bundled_experiment_path is not None:
            manifest["bundled_experiment_path"] = bundled_experiment_path
        code_archive.writestr(
            "project/.taac_training_manifest.json",
            dump_bytes(manifest, indent=2, trailing_newline=True),
        )
        code_archive.writestr(
            "project/pyproject.toml",
            "[project]\nname = \"minimal\"\nversion = \"0.0.0\"\n",
        )
        for package_init in (
            "project/src/taac2026/__init__.py",
            "project/src/taac2026/application/__init__.py",
            "project/src/taac2026/application/training/__init__.py",
        ):
            code_archive.writestr(package_init, "")
        code_archive.writestr(
            "project/src/taac2026/application/training/cli.py",
            "from __future__ import annotations\n"
            "\n"
            "import orjson\n"
            "import os\n"
            "import sys\n"
            "from pathlib import Path\n"
            "\n"
            "\n"
            "def main() -> None:\n"
            "    print(orjson.dumps({\"cwd\": str(Path.cwd()), \"argv\": sys.argv[1:], "
            "\"experiment\": os.environ.get(\"TAAC_EXPERIMENT\")}).decode())\n"
            "\n"
            "\n"
            "if __name__ == \"__main__\":\n"
            "    main()\n",
        )


def _write_minimal_eval_runtime_package(code_package_path: Path, *, bundled_experiment_path: str | None = "experiments/pcvr/minimal") -> None:
    with zipfile.ZipFile(code_package_path, mode="w", compression=zipfile.ZIP_DEFLATED) as code_archive:
        manifest: dict[str, object] = {}
        if bundled_experiment_path is not None:
            manifest["bundled_experiment_path"] = bundled_experiment_path
        code_archive.writestr(
            "project/.taac_training_manifest.json",
            dump_bytes(manifest, indent=2, trailing_newline=True),
        )
        code_archive.writestr(
            "project/pyproject.toml",
            "[project]\nname = \"minimal\"\nversion = \"0.0.0\"\n",
        )
        for package_init in (
            "project/src/taac2026/__init__.py",
            "project/src/taac2026/application/__init__.py",
            "project/src/taac2026/application/evaluation/__init__.py",
        ):
            code_archive.writestr(package_init, "")
        code_archive.writestr(
            "project/src/taac2026/application/evaluation/cli.py",
            "from __future__ import annotations\n"
            "\n"
            "import json\n"
            "import sys\n"
            "from pathlib import Path\n"
            "\n"
            "\n"
            "def main() -> None:\n"
            "    print(json.dumps({\"cwd\": str(Path.cwd()), \"argv\": sys.argv[1:]}))\n"
            "\n"
            "\n"
            "if __name__ == \"__main__\":\n"
            "    main()\n",
        )


def _write_fake_pip_package(root: Path, log_path: Path) -> Path:
    fake_pip = root / "fake_pip"
    pip_package = fake_pip / "pip"
    pip_package.mkdir(parents=True)
    (pip_package / "__init__.py").write_text("", encoding="utf-8")
    (pip_package / "__main__.py").write_text(
        "from __future__ import annotations\n"
        "\n"
        "import orjson\n"
        "import sys\n"
        "from pathlib import Path\n"
        "\n"
        f"Path({str(log_path)!r}).write_bytes(orjson.dumps(sys.argv[1:]))\n",
        encoding="utf-8",
    )
    return fake_pip


def _assert_pip_install_args(pip_args: list[str], *, expected_target: str) -> None:
    assert pip_args[:2] == ["install", "--disable-pip-version-check"]
    assert "-q" in pip_args
    assert expected_target in pip_args


def test_build_training_bundle_contains_runtime_sources(tmp_path: Path) -> None:
    output_dir = tmp_path / "baseline_bundle"

    result = build_training_bundle("experiments/pcvr/baseline", output_dir=output_dir)

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

    manifest = _code_package_manifest(result.code_package_path)
    assert manifest["bundle_format"] == "taac2026-training-v2"
    assert manifest["bundled_experiment_path"] == "experiments/pcvr/baseline"
    assert manifest["entrypoint"] == "run.sh"
    assert manifest["code_package"] == "code_package.zip"
    assert manifest["runtime_env"]["pip_extras"].startswith("TAAC_BUNDLE_PIP_EXTRAS")

    names = _code_package_names(result.code_package_path)
    assert "project/.taac_training_manifest.json" in names
    assert "project/pyproject.toml" in names
    assert "project/uv.lock" not in names
    assert "project/README.md" not in names
    assert "project/src/taac2026/application/training/cli.py" in names
    assert "project/src/taac2026/infrastructure/pcvr/training.py" in names
    assert "project/src/taac2026/infrastructure/pcvr/trainer.py" in names
    assert "project/experiments/pcvr/baseline/__init__.py" in names
    assert "project/experiments/pcvr/baseline/model.py" in names
    assert "project/experiments/pcvr/baseline/ns_groups.json" not in names
    assert "project/run.sh" not in names
    assert "project/experiments/pcvr/baseline/train.py" not in names
    assert "project/experiments/pcvr/baseline/trainer.py" not in names
    assert "project/experiments/pcvr/baseline/run.sh" not in names
    assert "project/tests/unit/test_package_training.py" not in names


@pytest.mark.parametrize(
    "experiment",
    NON_BASELINE_EXPERIMENTS,
)
def test_build_training_bundle_contains_experiment_ns_groups(tmp_path: Path, experiment: str) -> None:
    output_dir = tmp_path / f"{Path(experiment).name}_bundle"

    result = build_training_bundle(experiment, output_dir=output_dir)

    names = _code_package_names(result.code_package_path)
    assert f"project/{experiment}/__init__.py" in names
    assert f"project/{experiment}/model.py" in names
    assert f"project/{experiment}/ns_groups.json" not in names


@pytest.mark.parametrize("experiment", ["experiments/maintenance/host_device_info", "experiments/maintenance/online_dataset_eda"])
def test_build_training_bundle_supports_maintenance_experiment_packages(tmp_path: Path, experiment: str) -> None:
    output_dir = tmp_path / f"{Path(experiment).name}_bundle"

    result = build_training_bundle(experiment, output_dir=output_dir)

    manifest = _code_package_manifest(result.code_package_path)
    names = _code_package_names(result.code_package_path)
    assert manifest["bundled_experiment_path"] == experiment
    assert f"project/{experiment}/__init__.py" in names


def test_build_training_bundle_refuses_overwrite_without_force(tmp_path: Path) -> None:
    output_dir = tmp_path / "baseline_bundle"
    build_training_bundle("experiments/pcvr/baseline", output_dir=output_dir)

    with pytest.raises(FileExistsError):
        build_training_bundle("experiments/pcvr/baseline", output_dir=output_dir)


def test_build_training_bundle_force_replaces_two_file_output(tmp_path: Path) -> None:
    output_dir = tmp_path / "baseline_bundle"
    build_training_bundle("experiments/pcvr/baseline", output_dir=output_dir)
    (output_dir / "run.sh").write_text("stale\n", encoding="utf-8")

    result = build_training_bundle("experiments/pcvr/baseline", output_dir=output_dir, force=True)

    assert result.run_script_path.read_text(encoding="utf-8").startswith("#!/usr/bin/env bash")
    assert result.code_package_path.exists()


def test_training_run_script_installs_project_dependencies_before_entrypoint(tmp_path: Path) -> None:
    output_dir = tmp_path / "baseline_bundle"
    result = build_training_bundle("experiments/pcvr/baseline", output_dir=output_dir)
    _write_minimal_training_runtime_package(result.code_package_path)
    pip_args_path = tmp_path / "pip_args.json"
    fake_pip = _write_fake_pip_package(tmp_path, pip_args_path)

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
    assert payload["argv"][:2] == ["--experiment", "experiments/pcvr/minimal"]
    _assert_pip_install_args(pip_args, expected_target=".")
    assert "Installing TAAC project dependencies from pyproject.toml" in completed.stderr


def test_training_run_script_accepts_explicit_bundle_pip_extras(tmp_path: Path) -> None:
    output_dir = tmp_path / "baseline_bundle"
    result = build_training_bundle("experiments/pcvr/baseline", output_dir=output_dir)
    _write_minimal_training_runtime_package(result.code_package_path)
    pip_args_path = tmp_path / "pip_args.json"
    fake_pip = _write_fake_pip_package(tmp_path, pip_args_path)

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

    _assert_pip_install_args(pip_args, expected_target=".[dev]")


def test_training_run_script_does_not_inject_default_experiment(tmp_path: Path) -> None:
    output_dir = tmp_path / "baseline_bundle"
    result = build_training_bundle("experiments/pcvr/baseline", output_dir=output_dir)
    _write_minimal_training_runtime_package(result.code_package_path, bundled_experiment_path=None)
    pip_args_path = tmp_path / "pip_args.json"
    fake_pip = _write_fake_pip_package(tmp_path, pip_args_path)

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
    result = build_training_bundle("experiments/pcvr/baseline", output_dir=output_dir)
    _write_minimal_training_runtime_package(result.code_package_path)
    pip_args_path = tmp_path / "pip_args.json"
    fake_pip = _write_fake_pip_package(tmp_path, pip_args_path)

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
        "experiments/pcvr/minimal",
        "--dataset-path",
        "/platform/train.parquet",
        "--schema-path",
        "/platform/schema.json",
    ]
    assert "--run-dir" in payload["argv"]
    assert "/platform/output" in payload["argv"]


def test_training_run_script_infer_uses_platform_eval_env_paths(tmp_path: Path) -> None:
    output_dir = tmp_path / "baseline_bundle"
    result = build_training_bundle("experiments/pcvr/baseline", output_dir=output_dir)
    _write_minimal_eval_runtime_package(result.code_package_path)

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
        "experiments/pcvr/minimal",
        "--dataset-path",
        "/platform/eval.parquet",
        "--schema-path",
        "/platform/schema.json",
        "--result-dir",
        "/platform/results",
        "--checkpoint",
        "/platform/model.safetensors",
    ]


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
            "bundled_experiment_path": "experiments/pcvr/baseline",
            "runtime_env": {
                "dataset_path": "TRAIN_DATA_PATH",
                "schema_path": "TAAC_SCHEMA_PATH",
                "checkpoint_path": "TRAIN_CKPT_PATH",
                "cuda_profile": "TAAC_CUDA_PROFILE",
                "pip_extras": "TAAC_BUNDLE_PIP_EXTRAS (optional; defaults to runtime-only install with no dev extra)",
            },
        },
    )
    monkeypatch.setattr(package_training, "build_training_bundle", lambda *args, **kwargs: result)

    exit_code = package_training.main(["--experiment", "experiments/pcvr/baseline"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Built TAAC online training bundle" in captured.out
    assert "Experiment: experiments/pcvr/baseline" in captured.out
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
            "bundled_experiment_path": "experiments/pcvr/baseline",
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

    exit_code = package_training.main(["--experiment", "experiments/pcvr/baseline"])

    assert exit_code == 0
    assert calls == [{
        "experiment": "experiments/pcvr/baseline",
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
            "bundled_experiment_path": "experiments/pcvr/baseline",
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

    exit_code = package_training.main(["--experiment", "experiments/pcvr/baseline", "--no-force"])

    assert exit_code == 0
    assert calls == [{
        "experiment": "experiments/pcvr/baseline",
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
            "bundled_experiment_path": "experiments/pcvr/baseline",
            "runtime_env": {},
        },
    )
    monkeypatch.setattr(package_training, "build_training_bundle", lambda *args, **kwargs: result)

    exit_code = package_training.main(["--experiment", "experiments/pcvr/baseline", "--json"])

    captured = capsys.readouterr()
    assert exit_code == 0
    payload = loads(captured.out)
    assert payload["output_dir"] == str(result.output_dir)
    assert payload["run_script_path"] == str(result.run_script_path)
    assert payload["code_package_path"] == str(result.code_package_path)
    assert payload["manifest"]["bundled_experiment_path"] == "experiments/pcvr/baseline"
