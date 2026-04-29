from __future__ import annotations

import os
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

import taac2026.application.maintenance.package_inference as package_inference
from taac2026.application.maintenance.package_inference import BundleResult, build_inference_bundle
from taac2026.infrastructure.io.json_utils import dump_bytes, loads
from tests.unit._pcvr_experiment_matrix import discover_nonbaseline_pcvr_experiment_paths


NON_BASELINE_EXPERIMENTS = discover_nonbaseline_pcvr_experiment_paths()


def _code_package_names(code_package_path: Path) -> set[str]:
    with zipfile.ZipFile(code_package_path) as code_archive:
        return set(code_archive.namelist())


def _code_package_manifest(code_package_path: Path) -> dict[str, object]:
    with zipfile.ZipFile(code_package_path) as code_archive:
        payload = code_archive.read("project/.taac_inference_manifest.json")
    return loads(payload)


def _write_minimal_runtime_package(code_package_path: Path) -> None:
    with zipfile.ZipFile(code_package_path, mode="w", compression=zipfile.ZIP_DEFLATED) as code_archive:
        code_archive.writestr(
            "project/.taac_inference_manifest.json",
            dump_bytes({"bundled_experiment_path": "config/minimal"}, indent=2, trailing_newline=True),
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
            "project/src/taac2026/application/evaluation/infer.py",
            "from __future__ import annotations\n"
            "\n"
            "import orjson\n"
            "import os\n"
            "from pathlib import Path\n"
            "\n"
            "\n"
            "def main() -> None:\n"
            "    print(orjson.dumps({\"cwd\": str(Path.cwd()), \"experiment\": os.environ.get(\"TAAC_EXPERIMENT\")}).decode())\n",
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


def test_build_inference_bundle_contains_runtime_sources(tmp_path: Path) -> None:
    output_dir = tmp_path / "baseline_bundle"

    result = build_inference_bundle("config/baseline", output_dir=output_dir)

    assert result.output_dir == output_dir.resolve()
    assert result.run_script_path == output_dir.resolve() / "infer.py"
    assert result.code_package_path == output_dir.resolve() / "code_package.zip"
    assert sorted(path.name for path in output_dir.iterdir()) == ["code_package.zip", "infer.py"]
    assert result.run_script_path.exists()
    assert result.code_package_path.exists()
    infer_script = result.run_script_path.read_text(encoding="utf-8")
    assert "code_package.zip" in infer_script
    assert ".taac_inference_manifest.json" in infer_script
    assert "TAAC_INSTALL_PROJECT_DEPS" in infer_script
    assert "mirrors.cloud.tencent.com/pypi/simple" in infer_script
    assert "taac2026.application.evaluation.infer" in infer_script

    manifest = _code_package_manifest(result.code_package_path)
    assert manifest["bundle_format"] == "taac2026-inference-v1"
    assert manifest["bundled_experiment_path"] == "config/baseline"
    assert manifest["entrypoint"] == "infer.py"
    assert manifest["code_package"] == "code_package.zip"

    names = _code_package_names(result.code_package_path)
    assert "project/.taac_inference_manifest.json" in names
    assert "project/pyproject.toml" in names
    assert "project/uv.lock" in names
    assert "project/src/taac2026/application/evaluation/cli.py" in names
    assert "project/src/taac2026/application/evaluation/infer.py" in names
    assert "project/config/baseline/model.py" in names
    assert "project/config/baseline/ns_groups.json" in names
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

    names = _code_package_names(result.code_package_path)
    assert f"project/{experiment}/model.py" in names
    assert f"project/{experiment}/ns_groups.json" in names


def test_build_inference_bundle_refuses_overwrite_without_force(tmp_path: Path) -> None:
    output_dir = tmp_path / "baseline_bundle"
    build_inference_bundle("config/baseline", output_dir=output_dir)

    with pytest.raises(FileExistsError):
        build_inference_bundle("config/baseline", output_dir=output_dir)


def test_build_inference_bundle_force_replaces_two_file_output(tmp_path: Path) -> None:
    output_dir = tmp_path / "baseline_bundle"
    build_inference_bundle("config/baseline", output_dir=output_dir)
    (output_dir / "infer.py").write_text("stale\n", encoding="utf-8")

    result = build_inference_bundle("config/baseline", output_dir=output_dir, force=True)

    assert result.run_script_path.read_text(encoding="utf-8").startswith("#!/usr/bin/env python3")
    assert result.code_package_path.exists()


def test_generated_infer_script_requires_user_cache_path_without_workdir_override(tmp_path: Path) -> None:
    output_dir = tmp_path / "baseline_bundle"
    result = build_inference_bundle("config/baseline", output_dir=output_dir)
    _write_minimal_runtime_package(result.code_package_path)

    env = os.environ.copy()
    for variable in (
        "TAAC_BUNDLE_WORKDIR",
        "TAAC_CODE_PACKAGE",
        "TAAC_EXPERIMENT",
        "TAAC_FORCE_EXTRACT",
        "TAAC_INSTALL_PROJECT_DEPS",
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
    result = build_inference_bundle("config/baseline", output_dir=output_dir)
    _write_minimal_runtime_package(result.code_package_path)

    env = os.environ.copy()
    for variable in (
        "TAAC_BUNDLE_WORKDIR",
        "TAAC_CODE_PACKAGE",
        "TAAC_EXPERIMENT",
        "TAAC_FORCE_EXTRACT",
        "TAAC_INSTALL_PROJECT_DEPS",
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

    assert payload["experiment"] == "config/minimal"
    assert extracted_project_dir.name == "project"
    assert extracted_project_dir.is_relative_to(user_cache_path.resolve())
    assert not extracted_project_dir.is_relative_to(output_dir.resolve())


def test_generated_infer_script_installs_project_dependencies_before_entrypoint(tmp_path: Path) -> None:
    output_dir = tmp_path / "baseline_bundle"
    user_cache_path = tmp_path / "user_cache"
    result = build_inference_bundle("config/baseline", output_dir=output_dir)
    _write_minimal_runtime_package(result.code_package_path)
    pip_args_path = tmp_path / "pip_args.json"
    fake_pip = _write_fake_pip_package(tmp_path, pip_args_path)

    env = os.environ.copy()
    for variable in (
        "TAAC_BUNDLE_WORKDIR",
        "TAAC_CODE_PACKAGE",
        "TAAC_EXPERIMENT",
        "TAAC_FORCE_EXTRACT",
        "TAAC_INSTALL_PROJECT_DEPS",
        "TAAC_PIP_EXTRA_ARGS",
        "TAAC_PIP_EXTRAS",
        "TAAC_PIP_INDEX_URL",
        "TAAC_SKIP_PIP_INSTALL",
    ):
        env.pop(variable, None)
    env.update(
        {
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

    assert payload["experiment"] == "config/minimal"
    assert Path(payload["cwd"]).resolve().is_relative_to(user_cache_path.resolve())
    assert pip_args == ["install", "--disable-pip-version-check", "-q", "."]
    assert "Installing TAAC project dependencies from pyproject.toml" in completed.stderr


def test_package_inference_main_prints_human_readable_summary_by_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output_dir = tmp_path / "bundle"
    result = BundleResult(
        output_dir=output_dir,
        run_script_path=output_dir / "infer.py",
        code_package_path=output_dir / "code_package.zip",
        manifest={
            "bundle_format": "taac2026-inference-v1",
            "bundled_experiment_path": "config/baseline",
            "runtime_env": {
                "model_path": "MODEL_OUTPUT_PATH",
                "dataset_path": "EVAL_DATA_PATH",
                "result_path": "EVAL_RESULT_PATH",
                "schema_path": "TAAC_SCHEMA_PATH",
            },
        },
    )
    monkeypatch.setattr(package_inference, "build_inference_bundle", lambda *args, **kwargs: result)

    exit_code = package_inference.main(["--experiment", "config/baseline"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Built TAAC online inference bundle" in captured.out
    assert "Experiment: config/baseline" in captured.out
    assert f"infer.py: {result.run_script_path}" in captured.out
    assert f"code_package.zip: {result.code_package_path}" in captured.out
    assert "Upload the two files above: infer.py and code_package.zip" in captured.out


def test_package_inference_main_overwrites_existing_bundle_by_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_dir = tmp_path / "bundle"
    calls: list[dict[str, object]] = []
    result = BundleResult(
        output_dir=output_dir,
        run_script_path=output_dir / "infer.py",
        code_package_path=output_dir / "code_package.zip",
        manifest={
            "bundle_format": "taac2026-inference-v1",
            "bundled_experiment_path": "config/baseline",
            "runtime_env": {},
        },
    )

    def fake_build_inference_bundle(experiment: str, *, output_dir: Path | None = None, force: bool = False, root: Path | None = None):
        calls.append({
            "experiment": experiment,
            "output_dir": output_dir,
            "force": force,
            "root": root,
        })
        return result

    monkeypatch.setattr(package_inference, "build_inference_bundle", fake_build_inference_bundle)

    exit_code = package_inference.main(["--experiment", "config/baseline"])

    assert exit_code == 0
    assert calls == [{
        "experiment": "config/baseline",
        "output_dir": None,
        "force": True,
        "root": None,
    }]


def test_package_inference_main_honors_no_force(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_dir = tmp_path / "bundle"
    calls: list[dict[str, object]] = []
    result = BundleResult(
        output_dir=output_dir,
        run_script_path=output_dir / "infer.py",
        code_package_path=output_dir / "code_package.zip",
        manifest={
            "bundle_format": "taac2026-inference-v1",
            "bundled_experiment_path": "config/baseline",
            "runtime_env": {},
        },
    )

    def fake_build_inference_bundle(experiment: str, *, output_dir: Path | None = None, force: bool = False, root: Path | None = None):
        calls.append({
            "experiment": experiment,
            "output_dir": output_dir,
            "force": force,
            "root": root,
        })
        return result

    monkeypatch.setattr(package_inference, "build_inference_bundle", fake_build_inference_bundle)

    exit_code = package_inference.main(["--experiment", "config/baseline", "--no-force"])

    assert exit_code == 0
    assert calls == [{
        "experiment": "config/baseline",
        "output_dir": None,
        "force": False,
        "root": None,
    }]


def test_package_inference_main_prints_json_when_requested(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output_dir = tmp_path / "bundle"
    result = BundleResult(
        output_dir=output_dir,
        run_script_path=output_dir / "infer.py",
        code_package_path=output_dir / "code_package.zip",
        manifest={
            "bundle_format": "taac2026-inference-v1",
            "bundled_experiment_path": "config/baseline",
            "runtime_env": {},
        },
    )
    monkeypatch.setattr(package_inference, "build_inference_bundle", lambda *args, **kwargs: result)

    exit_code = package_inference.main(["--experiment", "config/baseline", "--json"])

    captured = capsys.readouterr()
    assert exit_code == 0
    payload = loads(captured.out)
    assert payload["output_dir"] == str(result.output_dir)
    assert payload["run_script_path"] == str(result.run_script_path)
    assert payload["code_package_path"] == str(result.code_package_path)
    assert payload["manifest"]["bundled_experiment_path"] == "config/baseline"