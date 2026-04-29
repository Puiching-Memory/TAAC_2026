from __future__ import annotations

from dataclasses import replace
import importlib.util
from pathlib import Path
import sys

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from taac2026.domain.config import TrainRequest
from taac2026.infrastructure.experiments.loader import load_experiment_package
from taac2026.infrastructure.io.json_utils import dumps


REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_package_module(package_name: str):
    package_dir = REPO_ROOT / "config" / package_name
    module_name = f"test_{package_name}_experiment"
    spec = importlib.util.spec_from_file_location(
        module_name,
        package_dir / "__init__.py",
        submodule_search_locations=[str(package_dir)],
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _load_host_device_info_runner_module():
    package_dir = REPO_ROOT / "config" / "host_device_info"
    module_name = "test_host_device_info_runner"
    spec = importlib.util.spec_from_file_location(module_name, package_dir / "runner.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _write_schema(path: Path) -> None:
    payload = {
        "user_int": [[1, 10, 1], [2, 20, 1]],
        "item_int": [[3, 20, 1]],
        "user_dense": [[4, 2]],
        "seq": {
            "seq_a": {"prefix": "domain_a_seq", "ts_fid": 10, "features": [[10, 100], [11, 20]]},
        },
    }
    path.write_text(dumps(payload, indent=2, trailing_newline=True), encoding="utf-8")


def _write_dataset(path: Path) -> None:
    pq.write_table(
        pa.table(
            {
                "user_int_feats_1": [1, 1, 2, 3],
                "user_int_feats_2": [10, None, 11, 10],
                "item_int_feats_3": [100, 101, 100, None],
                "user_dense_feats_4": [[0.1, 0.2], [0.0, 0.0], [0.5, 0.4], []],
                "domain_a_seq_10": [[1, 2, 3], [2], [], [4, 5]],
                "domain_a_seq_11": [[100, 100, 101], [102], [], [103, 103]],
            }
        ),
        path,
    )


@pytest.mark.parametrize(
    ("experiment_path", "requires_dataset"),
    [("config/host_device_info", False), ("config/online_dataset_eda", True)],
)
def test_load_maintenance_experiment_packages(experiment_path: str, requires_dataset: bool) -> None:
    experiment = load_experiment_package(experiment_path)

    assert experiment.metadata["kind"] == "maintenance"
    assert experiment.metadata["requires_dataset"] is requires_dataset


def test_host_device_info_experiment_runs_without_writing_log_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    host_device_info_package = _load_package_module("host_device_info")
    experiment = host_device_info_package.EXPERIMENT
    captured: dict[str, object] = {}

    def fake_collect(config):
        captured["config"] = config
        return {
            "repo_root": str(config.repo_root),
            "requested_profile": None,
            "requested_python": None,
        }

    monkeypatch.setattr(host_device_info_package, "collect_host_device_info", fake_collect)

    run_dir = tmp_path / "outputs"
    result = experiment.train(
        TrainRequest(
            experiment="config/host_device_info",
            dataset_path=None,
            schema_path=None,
            run_dir=run_dir,
        )
    )

    assert result["run_dir"] == str(run_dir.resolve())
    assert "log_path" not in result
    assert captured["config"].repo_root == host_device_info_package.PROJECT_ROOT
    assert captured["config"].requested_profile is None
    assert not run_dir.exists()


def test_host_device_info_experiment_rejects_extra_args(tmp_path: Path) -> None:
    host_device_info_package = _load_package_module("host_device_info")
    experiment = host_device_info_package.EXPERIMENT

    run_dir = tmp_path / "host_device_info"
    with pytest.raises(ValueError, match="does not accept extra_args"):
        experiment.train(
            TrainRequest(
                experiment="config/host_device_info",
                dataset_path=None,
                schema_path=None,
                run_dir=run_dir,
                extra_args=("--requested-profile", "cpu"),
            )
        )


def test_host_device_info_runner_converts_timeout_to_failure_result() -> None:
    runner_module = _load_host_device_info_runner_module()

    return_code, output = runner_module._run_command(
        [sys.executable, "-c", "import time; time.sleep(1)"],
        timeout=0.01,
    )

    assert return_code == runner_module.TIMEOUT_EXIT_CODE
    assert "timed out" in output


def test_online_dataset_eda_experiment_prints_report_to_stdout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    online_dataset_eda_package = _load_package_module("online_dataset_eda")
    experiment = online_dataset_eda_package.EXPERIMENT
    schema_path = tmp_path / "schema.json"
    dataset_path = tmp_path / "demo.parquet"
    run_dir = tmp_path / "outputs"
    _write_schema(schema_path)
    _write_dataset(dataset_path)
    monkeypatch.setattr(
        online_dataset_eda_package,
        "ONLINE_DATASET_EDA_CONFIG",
        replace(online_dataset_eda_package.ONLINE_DATASET_EDA_CONFIG, max_rows=2),
    )

    result = experiment.train(
        TrainRequest(
            experiment="config/online_dataset_eda",
            dataset_path=dataset_path,
            schema_path=schema_path,
            run_dir=run_dir,
        )
    )

    captured = capsys.readouterr()
    assert result["dataset_role"] == "online"
    assert result["row_count"] == 2
    assert result["sampled"] is True
    assert "report_path" not in result
    assert "chart_dir" not in result
    assert "[online-eda] sink=stdout" in captured.out
    assert "== Dataset ==" in captured.out
    assert not run_dir.exists()


def test_online_dataset_eda_experiment_rejects_extra_args(tmp_path: Path) -> None:
    experiment = _load_package_module("online_dataset_eda").EXPERIMENT
    schema_path = tmp_path / "schema.json"
    dataset_path = tmp_path / "demo.parquet"
    _write_schema(schema_path)
    _write_dataset(dataset_path)

    with pytest.raises(ValueError, match="does not accept extra_args"):
        experiment.train(
            TrainRequest(
                experiment="config/online_dataset_eda",
                dataset_path=dataset_path,
                schema_path=schema_path,
                run_dir=tmp_path / "outputs",
                extra_args=("--max-rows", "2"),
            )
        )