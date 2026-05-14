from __future__ import annotations

from dataclasses import replace
import importlib.util
import logging
from pathlib import Path
import sys
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from tests.support.paths import locate_repo_root
from taac2026.domain.requests import InferRequest, TrainRequest
from taac2026.application.experiments.registry import load_experiment_package
from taac2026.infrastructure.io.json import dumps, loads


REPO_ROOT = locate_repo_root(Path(__file__))
RESULT_PREFIX = "ONLINE_DATASET_EDA_RESULT="


def _online_eda_stdout_payload(output: str) -> dict[str, object]:
    for line in output.splitlines():
        if line.startswith(RESULT_PREFIX):
            return loads(line.removeprefix(RESULT_PREFIX))
    raise AssertionError(f"missing {RESULT_PREFIX!r} line in stdout: {output}")


def _load_package_module(package_name: str):
    package_dir = REPO_ROOT / "experiments" / package_name
    module_name = f"test_{package_name}_experiment"
    spec = importlib.util.spec_from_file_location(
        module_name,
        package_dir / "__init__.py",
        submodule_search_locations=[str(package_dir)],
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    previous_module = sys.modules.get(module_name)
    try:
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    finally:
        if previous_module is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = previous_module
    return module


def _load_host_device_info_runner_module():
    package_dir = REPO_ROOT / "experiments" / "host_device_info"
    module_name = "test_host_device_info_runner"
    spec = importlib.util.spec_from_file_location(module_name, package_dir / "runner.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    previous_module = sys.modules.get(module_name)
    try:
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    finally:
        if previous_module is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = previous_module
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
                "label_type": [2, 1, 2, None],
            }
        ),
        path,
    )


@pytest.mark.parametrize(
    ("experiment_path", "requires_dataset"),
    [("experiments/host_device_info", False), ("experiments/online_dataset_eda", True)],
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
            experiment="experiments/host_device_info",
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
                experiment="experiments/host_device_info",
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


def test_host_device_info_runner_logs_deduplicated_active_python_packages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner_module = _load_host_device_info_runner_module()
    messages: list[str] = []

    class FakeSink:
        def close(self) -> None:
            return None

        def log(self, message: str) -> None:
            messages.append(message)

    class FakeDistribution:
        def __init__(self, name: str, version: str, root: str, top_levels: tuple[str, ...]) -> None:
            self.metadata = {"Name": name}
            self.name = name
            self.version = version
            self.files = [Path(f"{top_level}/__init__.py") for top_level in top_levels]
            self._root = Path(root)
            self._top_levels = top_levels

        def read_text(self, filename: str) -> str | None:
            if filename != "top_level.txt":
                return None
            return "\n".join(self._top_levels)

        def locate_file(self, path: str) -> Path:
            return self._root / path

    active_distributions = {
        "numpy": FakeDistribution("numpy", "2.2.6", "/env/new/site-packages", ("numpy",)),
        "Pillow": FakeDistribution("Pillow", "12.2.0", "/env/new/site-packages", ("PIL",)),
        "torch": FakeDistribution("torch", "2.7.1", "/env/site-packages", ("torch",)),
    }
    discovered_distributions = [
        FakeDistribution("numpy", "2.2.5", "/env/old/site-packages", ("numpy",)),
        active_distributions["numpy"],
        FakeDistribution("Pillow", "12.1.1", "/env/old/site-packages", ("PIL",)),
        active_distributions["Pillow"],
        active_distributions["torch"],
    ]

    def fake_distribution(name: str):
        try:
            return active_distributions[name]
        except KeyError as error:
            raise runner_module.metadata.PackageNotFoundError(name) from error

    def fake_find_spec(name: str):
        specs = {
            "numpy": SimpleNamespace(submodule_search_locations=["/env/new/site-packages/numpy"], origin=None),
            "PIL": SimpleNamespace(submodule_search_locations=["/env/new/site-packages/PIL"], origin=None),
            "torch": SimpleNamespace(submodule_search_locations=["/env/site-packages/torch"], origin=None),
        }
        return specs.get(name)

    monkeypatch.setattr(runner_module.metadata, "distributions", lambda: iter(discovered_distributions))
    monkeypatch.setattr(runner_module.metadata, "distribution", fake_distribution)
    monkeypatch.setattr(runner_module.importlib.util, "find_spec", fake_find_spec)

    runner_module._log_python_packages(FakeSink())

    assert messages[0] == "---- python packages ----"
    assert "installed_python_packages=3" in messages
    assert "numpy==2.2.6 source=/env/new/site-packages/numpy" in messages
    assert "Pillow==12.2.0 source=/env/new/site-packages/PIL" in messages
    assert "torch==2.7.1 source=/env/site-packages/torch" in messages
    assert all("2.2.5" not in message for message in messages)
    assert all("12.1.1" not in message for message in messages)


def test_host_device_info_runner_continues_after_unexpected_section_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner_module = _load_host_device_info_runner_module()
    events: list[str] = []
    messages: list[str] = []

    class FakeSink:
        def close(self) -> None:
            messages.append("__closed__")

        def log(self, message: str) -> None:
            messages.append(message)

    def record(name: str):
        def _handler(*args, **kwargs) -> None:
            events.append(name)

        return _handler

    def fail_conda_search(*args, **kwargs) -> None:
        events.append("conda_search_probes")
        raise RuntimeError("conda probe exploded")

    monkeypatch.setattr(runner_module, "LogSink", FakeSink)
    monkeypatch.setattr(runner_module, "_log_os_release", record("os_release"))
    monkeypatch.setattr(runner_module, "_log_proxy_environment", record("proxy_environment"))
    monkeypatch.setattr(runner_module, "_log_command", lambda sink, title, command, timeout=None: events.append(title))
    monkeypatch.setattr(runner_module, "_log_network_info", record("network"))
    monkeypatch.setattr(runner_module, "_log_device_nodes", lambda sink, pattern, title, missing_message: events.append(title))
    monkeypatch.setattr(runner_module, "_log_uv_bootstrap_status", record("uv_bootstrap"))
    monkeypatch.setattr(runner_module, "_log_dependency_index_status", record("dependency_indexes"))
    monkeypatch.setattr(runner_module, "_log_connectivity_matrix", record("connectivity_matrix"))
    monkeypatch.setattr(runner_module, "_log_pip_download_probes", record("pip_download_probes"))
    monkeypatch.setattr(runner_module, "_log_conda_search_probes", fail_conda_search)
    monkeypatch.setattr(runner_module, "_log_build_tools", record("build_tools"))
    monkeypatch.setattr(runner_module, "_log_python_info", record("python_info"))
    monkeypatch.setattr(runner_module, "_log_python_packages", record("python_packages"))

    summary = runner_module.collect_host_device_info(runner_module.HostDeviceInfoConfig(repo_root=REPO_ROOT))

    assert summary["repo_root"] == str(REPO_ROOT)
    assert "conda_search_probes" in events
    assert "build_tools" in events
    assert "python_info" in events
    assert "python_packages" in events
    assert any("conda_search_probes_failure_class=RuntimeError" in message for message in messages)
    assert any("conda_search_probes_failure_detail=conda probe exploded" in message for message in messages)
    assert messages[-1] == "__closed__"


def test_online_dataset_eda_experiment_prints_report_to_stdout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    log_capture,
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

    with log_capture.at_level(logging.INFO):
        result = experiment.train(
            TrainRequest(
                experiment="experiments/online_dataset_eda",
                dataset_path=dataset_path,
                schema_path=schema_path,
                run_dir=run_dir,
            )
        )

    captured = capsys.readouterr()
    assert result["dataset_role"] == "train"
    assert result["row_count"] == 2
    assert result["sampled"] is True
    assert result["stdout_result"] is True
    assert "[online-eda] scan=arrow-profile" in log_capture.text
    payload = _online_eda_stdout_payload(captured.out)
    assert payload["dataset_role"] == "train"
    assert payload["row_count"] == 2
    assert not run_dir.exists()


def test_online_dataset_eda_experiment_runs_as_inference_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    log_capture,
) -> None:
    online_dataset_eda_package = _load_package_module("online_dataset_eda")
    experiment = online_dataset_eda_package.EXPERIMENT
    schema_path = tmp_path / "schema.json"
    dataset_path = tmp_path / "demo.parquet"
    train_dir = tmp_path / "train_outputs"
    result_dir = tmp_path / "infer_outputs"
    _write_schema(schema_path)
    _write_dataset(dataset_path)
    monkeypatch.setattr(
        online_dataset_eda_package,
        "ONLINE_DATASET_EDA_CONFIG",
        replace(online_dataset_eda_package.ONLINE_DATASET_EDA_CONFIG, max_rows=2),
    )

    train_result = experiment.train(
        TrainRequest(
            experiment="experiments/online_dataset_eda",
            dataset_path=dataset_path,
            schema_path=schema_path,
            run_dir=train_dir,
        )
    )
    train_payload = _online_eda_stdout_payload(capsys.readouterr().out)
    reference_profile = tmp_path / "train-profile.json"
    reference_profile.write_text(dumps(train_payload, trailing_newline=True), encoding="utf-8")

    with log_capture.at_level(logging.INFO):
        infer_result = experiment.infer(
            InferRequest(
                experiment="experiments/online_dataset_eda",
                dataset_path=dataset_path,
                schema_path=schema_path,
                checkpoint_path=reference_profile,
                result_dir=result_dir,
            )
        )

    captured = capsys.readouterr()
    assert infer_result["dataset_role"] == "infer"
    assert infer_result["stdout_result"] is True
    assert infer_result["reference_profile_path"] == str(reference_profile.resolve())
    assert infer_result["risk_flags"] == []
    payload = _online_eda_stdout_payload(captured.out)
    assert payload["comparison"]["schema_signature_match"] is True
    assert train_result["stdout_result"] is True


def test_online_dataset_eda_experiment_rejects_extra_args(tmp_path: Path) -> None:
    experiment = _load_package_module("online_dataset_eda").EXPERIMENT
    schema_path = tmp_path / "schema.json"
    dataset_path = tmp_path / "demo.parquet"
    _write_schema(schema_path)
    _write_dataset(dataset_path)

    with pytest.raises(ValueError, match="does not accept extra_args"):
        experiment.train(
            TrainRequest(
                experiment="experiments/online_dataset_eda",
                dataset_path=dataset_path,
                schema_path=schema_path,
                run_dir=tmp_path / "outputs",
                extra_args=("--max-rows", "2"),
            )
        )
