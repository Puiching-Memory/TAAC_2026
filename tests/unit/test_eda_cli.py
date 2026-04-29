from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from taac2026.application.reporting.eda_cli import main, resolve_dataset_role


def _load_online_eda_runner_module():
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "config" / "online_dataset_eda" / "runner.py"
    module_name = "test_online_dataset_eda_runner"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
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
            "seq_b": {"prefix": "domain_b_seq", "ts_fid": 20, "features": [[20, 100], [21, 20]]},
            "seq_c": {"prefix": "domain_c_seq", "ts_fid": 30, "features": [[30, 100], [31, 20]]},
            "seq_d": {"prefix": "domain_d_seq", "ts_fid": 40, "features": [[40, 100], [41, 20]]},
        },
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_dataset(path: Path, *, include_labels: bool) -> None:
    columns: dict[str, list[object]] = {
        "user_int_feats_1": [1, 1, 2, 3],
        "user_int_feats_2": [10, None, 11, 10],
        "item_int_feats_3": [100, 101, 100, None],
        "user_dense_feats_4": [[0.1, 0.2], [0.0, 0.0], [0.5, 0.4], []],
        "domain_a_seq_10": [[1, 2, 3], [2], [], [4, 5]],
        "domain_a_seq_11": [[100, 100, 101], [102], [], [103, 103]],
        "domain_b_seq_20": [[1], [], [3, 4], [5]],
        "domain_b_seq_21": [[11], [], [12, 13], [13]],
        "domain_c_seq_30": [[], [1, 2], [3], [4, 5, 6]],
        "domain_c_seq_31": [[], [21, 21], [22], [23, 24, 24]],
        "domain_d_seq_40": [[7, 8], [], [9], []],
        "domain_d_seq_41": [[31, 31], [], [32], []],
    }
    if include_labels:
        columns["label_type"] = [1, 2, 1, 2]
        columns["label_action_type"] = [1, 2, 1, 2]
    pq.write_table(pa.table(columns), path)


def _run_online_eda_runner(
    *,
    dataset_path: Path,
    schema_path: Path,
    max_rows: int | None = None,
    sample_percent: float | None = None,
) -> dict[str, object]:
    runner_module = _load_online_eda_runner_module()
    config = runner_module.OnlineDatasetEDAConfig(
        dataset_path=dataset_path.resolve(),
        schema_path=schema_path.resolve(),
        max_rows=max_rows,
        sample_percent=sample_percent,
    )
    return runner_module.run_online_dataset_eda(config)


def test_resolve_dataset_role_uses_labels_for_auto() -> None:
    role, label_columns = resolve_dataset_role("auto", {"label_type", "user_int_feats_1"})

    assert role.value == "test"
    assert label_columns == ("label_type",)


def test_resolve_dataset_role_rejects_unlabeled_test_dataset() -> None:
    with pytest.raises(ValueError):
        resolve_dataset_role("test", {"user_int_feats_1"})


def test_main_generates_label_charts_for_test_dataset(tmp_path: Path) -> None:
    schema_path = tmp_path / "schema.json"
    dataset_path = tmp_path / "demo.parquet"
    output_path = tmp_path / "dataset_eda.json"
    chart_dir = tmp_path / "charts"
    _write_schema(schema_path)
    _write_dataset(dataset_path, include_labels=True)

    exit_code = main(
        [
            "--dataset-path",
            str(dataset_path),
            "--schema-path",
            str(schema_path),
            "--dataset-role",
            "test",
            "--output",
            str(output_path),
            "--chart-dir",
            str(chart_dir),
        ]
    )

    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert report["dataset_role"] == "test"
    assert report["label_dependent_analyses_enabled"] is True
    assert (chart_dir / "label_distribution.echarts.json").exists()
    assert (chart_dir / "feature_auc.echarts.json").exists()


def test_main_skips_label_charts_for_online_dataset(tmp_path: Path) -> None:
    schema_path = tmp_path / "schema.json"
    dataset_path = tmp_path / "demo.parquet"
    output_path = tmp_path / "online_dataset_eda.json"
    chart_dir = tmp_path / "online_charts"
    _write_schema(schema_path)
    _write_dataset(dataset_path, include_labels=False)

    exit_code = main(
        [
            "--dataset-path",
            str(dataset_path),
            "--schema-path",
            str(schema_path),
            "--dataset-role",
            "online",
            "--output",
            str(output_path),
            "--chart-dir",
            str(chart_dir),
        ]
    )

    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert report["dataset_role"] == "online"
    assert report["label_dependent_analyses_enabled"] is False
    assert "label_distribution" in report["skipped_charts"]
    assert not (chart_dir / "label_distribution.echarts.json").exists()


def test_online_dataset_eda_runner_prints_summary(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    schema_path = tmp_path / "schema.json"
    dataset_path = tmp_path / "demo.parquet"
    _write_schema(schema_path)
    _write_dataset(dataset_path, include_labels=False)

    report = _run_online_eda_runner(
        dataset_path=dataset_path,
        schema_path=schema_path,
        max_rows=4,
    )
    captured = capsys.readouterr()

    assert report["row_count"] == 4
    assert "[online-eda] dataset=" in captured.out
    assert "== Dataset ==" in captured.out
    assert "== Top Null Rates ==" in captured.out


def test_online_dataset_eda_runner_streams_full_dataset_by_default(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    schema_path = tmp_path / "schema.json"
    dataset_path = tmp_path / "demo.parquet"
    _write_schema(schema_path)
    _write_dataset(dataset_path, include_labels=False)

    report = _run_online_eda_runner(
        dataset_path=dataset_path,
        schema_path=schema_path,
    )
    captured = capsys.readouterr()

    assert report["row_count"] == 4
    assert "scan=streaming full" in captured.out
    assert "summary: online dataset, 4 rows" in captured.out


def test_online_dataset_eda_runner_honors_explicit_max_rows(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    schema_path = tmp_path / "schema.json"
    dataset_path = tmp_path / "demo.parquet"
    _write_schema(schema_path)
    _write_dataset(dataset_path, include_labels=False)

    report = _run_online_eda_runner(
        dataset_path=dataset_path,
        schema_path=schema_path,
        max_rows=2,
    )
    captured = capsys.readouterr()

    assert report["row_count"] == 2
    assert "scan=streaming max_rows=2" in captured.out
    assert "summary: online dataset, scanned 2/4 rows" in captured.out
    assert "progress first-pass:" in captured.out
    assert "progress second-pass:" in captured.out


def test_online_dataset_eda_runner_honors_sample_percent(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    schema_path = tmp_path / "schema.json"
    dataset_path = tmp_path / "demo.parquet"
    _write_schema(schema_path)
    _write_dataset(dataset_path, include_labels=False)

    report = _run_online_eda_runner(
        dataset_path=dataset_path,
        schema_path=schema_path,
        sample_percent=50.0,
    )
    captured = capsys.readouterr()

    assert report["row_count"] == 2
    assert "scan=streaming sample_percent=50.0 max_rows=2" in captured.out
    assert "summary: online dataset, scanned 2/4 rows" in captured.out