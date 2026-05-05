from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import pytest

import taac2026.application.maintenance.package_inference as package_inference
import taac2026.application.maintenance.package_training as package_training
from taac2026.infrastructure.bundles.common import BundleResult
from taac2026.infrastructure.io.json_utils import loads


@dataclass(frozen=True, slots=True)
class BundleCliCase:
    name: str
    module: ModuleType
    builder_name: str
    entrypoint_name: str
    bundle_format: str
    runtime_env: dict[str, str]


CASES = (
    BundleCliCase(
        name="training",
        module=package_training,
        builder_name="build_training_bundle",
        entrypoint_name="run.sh",
        bundle_format="taac2026-training-v2",
        runtime_env={
            "dataset_path": "TRAIN_DATA_PATH",
            "schema_path": "TAAC_SCHEMA_PATH",
            "checkpoint_path": "TRAIN_CKPT_PATH",
            "cuda_profile": "TAAC_CUDA_PROFILE",
            "pip_extras": "TAAC_BUNDLE_PIP_EXTRAS (optional; defaults to runtime-only install with no dev extra)",
        },
    ),
    BundleCliCase(
        name="inference",
        module=package_inference,
        builder_name="build_inference_bundle",
        entrypoint_name="infer.py",
        bundle_format="taac2026-inference-v1",
        runtime_env={
            "model_path": "MODEL_OUTPUT_PATH",
            "dataset_path": "EVAL_DATA_PATH",
            "result_path": "EVAL_RESULT_PATH",
            "schema_path": "TAAC_SCHEMA_PATH",
            "pip_extras": "TAAC_BUNDLE_PIP_EXTRAS (optional; defaults to runtime-only install with no dev extra)",
        },
    ),
)


def _make_result(case: BundleCliCase, output_dir: Path, *, runtime_env: dict[str, str] | None = None) -> BundleResult:
    return BundleResult(
        output_dir=output_dir,
        run_script_path=output_dir / case.entrypoint_name,
        code_package_path=output_dir / "code_package.zip",
        manifest={
            "bundle_format": case.bundle_format,
            "bundled_experiment_path": "experiments/pcvr/baseline",
            "runtime_env": runtime_env if runtime_env is not None else {},
        },
    )


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_package_main_prints_human_readable_summary_by_default(
    case: BundleCliCase,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output_dir = tmp_path / "bundle"
    result = _make_result(case, output_dir, runtime_env=case.runtime_env)
    monkeypatch.setattr(case.module, case.builder_name, lambda *args, **kwargs: result)

    exit_code = case.module.main(["--experiment", "experiments/pcvr/baseline"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert f"Built TAAC online {case.name} bundle" in captured.out
    assert "Experiment: experiments/pcvr/baseline" in captured.out
    assert f"{case.entrypoint_name}: {result.run_script_path}" in captured.out
    assert f"code_package.zip: {result.code_package_path}" in captured.out
    assert f"Upload the two files above: {case.entrypoint_name} and code_package.zip" in captured.out


@pytest.mark.parametrize("case,force_flag", [(case, True) for case in CASES] + [(case, False) for case in CASES], ids=lambda item: item.name if isinstance(item, BundleCliCase) else str(item))
def test_package_main_passes_force_flag(
    case: BundleCliCase,
    force_flag: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_dir = tmp_path / "bundle"
    calls: list[dict[str, object]] = []
    result = _make_result(case, output_dir)

    def fake_build_bundle(experiment: str, *, output_dir: Path | None = None, force: bool = False, root: Path | None = None) -> BundleResult:
        calls.append(
            {
                "experiment": experiment,
                "output_dir": output_dir,
                "force": force,
                "root": root,
            }
        )
        return result

    monkeypatch.setattr(case.module, case.builder_name, fake_build_bundle)

    argv = ["--experiment", "experiments/pcvr/baseline"]
    if not force_flag:
        argv.append("--no-force")
    exit_code = case.module.main(argv)

    assert exit_code == 0
    assert calls == [
        {
            "experiment": "experiments/pcvr/baseline",
            "output_dir": None,
            "force": force_flag,
            "root": None,
        }
    ]


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_package_main_prints_json_when_requested(
    case: BundleCliCase,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output_dir = tmp_path / "bundle"
    result = _make_result(case, output_dir)
    monkeypatch.setattr(case.module, case.builder_name, lambda *args, **kwargs: result)

    exit_code = case.module.main(["--experiment", "experiments/pcvr/baseline", "--json"])

    captured = capsys.readouterr()
    assert exit_code == 0
    payload = loads(captured.out)
    assert payload["output_dir"] == str(result.output_dir)
    assert payload["run_script_path"] == str(result.run_script_path)
    assert payload["code_package_path"] == str(result.code_package_path)
    assert payload["manifest"]["bundled_experiment_path"] == "experiments/pcvr/baseline"