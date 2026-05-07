from __future__ import annotations

import importlib.util
import logging
from pathlib import Path
import sys

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from tests.support.paths import locate_repo_root
from taac2026.infrastructure.io.json import dumps


def _load_online_eda_runner_module():
    repo_root = locate_repo_root(Path(__file__))
    module_path = repo_root / "experiments" / "online_dataset_eda" / "runner.py"
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
    path.write_text(dumps(payload, indent=2, trailing_newline=True), encoding="utf-8")


def _write_dataset(path: Path) -> None:
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
        "label_type": [2, 1, 2, None],
    }
    pq.write_table(pa.table(columns), path)


def _run_online_eda_runner(
    *,
    dataset_path: Path,
    schema_path: Path,
    max_rows: int | None = None,
    sample_percent: float | None = None,
    config_overrides: dict[str, object] | None = None,
) -> dict[str, object]:
    runner_module = _load_online_eda_runner_module()
    overrides = config_overrides or {}
    config = runner_module.OnlineDatasetEDAConfig(
        dataset_path=dataset_path.resolve(),
        schema_path=schema_path.resolve(),
        max_rows=max_rows,
        sample_percent=sample_percent,
        **overrides,
    )
    return runner_module.run_online_dataset_eda(config)


def test_online_dataset_eda_runner_prints_summary(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    log_capture,
) -> None:
    schema_path = tmp_path / "schema.json"
    dataset_path = tmp_path / "demo.parquet"
    _write_schema(schema_path)
    _write_dataset(dataset_path)

    with log_capture.at_level(logging.INFO):
        report = _run_online_eda_runner(
            dataset_path=dataset_path,
            schema_path=schema_path,
            max_rows=4,
        )
    captured = capsys.readouterr()

    assert report["row_count"] == 4
    assert "[online-eda] dataset=" in log_capture.text
    assert "== Dataset ==" in captured.out
    assert "== Top Null Rates ==" in captured.out


def test_online_dataset_eda_runner_reports_first_layer_stats(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    log_capture,
) -> None:
    schema_path = tmp_path / "schema.json"
    dataset_path = tmp_path / "demo.parquet"
    _write_schema(schema_path)
    _write_dataset(dataset_path)

    with log_capture.at_level(logging.INFO):
        report = _run_online_eda_runner(
            dataset_path=dataset_path,
            schema_path=schema_path,
            max_rows=4,
        )
    captured = capsys.readouterr()
    stats = report["stats"]

    assert report["label_columns"] == ["label_type"]
    assert report["label_dependent_analyses_enabled"] is True
    assert stats["label_distribution"] == [
        {
            "name": "label_type",
            "total": 4,
            "observed": 3,
            "positive": 2,
            "negative": 1,
            "missing": 1,
            "positive_rate": 0.666667,
        }
    ]
    assert stats["dense_distributions"] == [
        {
            "name": "user_dense_feats_4",
            "mean": 0.2,
            "variance": 0.036667,
            "std": 0.191485,
            "zero_frac": 0.333333,
        }
    ]
    assert {row["name"] for row in stats["sequence_token_cardinality"]} == {
        "domain_a_seq_11",
        "domain_b_seq_21",
        "domain_c_seq_31",
        "domain_d_seq_41",
    }
    assert all(row["cardinality"] > 0 for row in stats["sequence_token_cardinality"])
    coverage_by_domain = {row["domain"]: row for row in stats["cross_domain_coverage"]}
    assert coverage_by_domain["seq_a"] == {"domain": "seq_a", "sampled_users": 3, "covered_users": 2, "coverage": 0.666667}
    assert coverage_by_domain["seq_d"] == {"domain": "seq_d", "sampled_users": 3, "covered_users": 2, "coverage": 0.666667}
    assert "== Label Distribution ==" in captured.out
    assert "label_type: positive=2 negative=1 observed=3 missing=1 positive_rate=0.666667" in captured.out
    assert "== Top Sequence Token Cardinalities ==" in captured.out
    assert "== Sampled Cross-Domain Coverage ==" in captured.out


def test_online_dataset_eda_runner_reports_second_layer_stats(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    log_capture,
) -> None:
    schema_path = tmp_path / "schema.json"
    dataset_path = tmp_path / "demo.parquet"
    _write_schema(schema_path)
    _write_dataset(dataset_path)

    with log_capture.at_level(logging.INFO):
        report = _run_online_eda_runner(
            dataset_path=dataset_path,
            schema_path=schema_path,
            max_rows=4,
            config_overrides={
                "label_feature_min_support": 1,
                "label_feature_top_k": 10,
                "categorical_pair_max_columns": 3,
                "categorical_pair_max_cardinality": 20,
                "categorical_pair_sample_rows": 4,
                "categorical_pair_top_k": 5,
            },
        )
    captured = capsys.readouterr()
    stats = report["stats"]

    null_by_label = {row["name"]: row for row in stats["null_rate_by_label"]}
    assert null_by_label["user_int_feats_2"] == {
        "name": "user_int_feats_2",
        "positive_null_rate": 0.0,
        "negative_null_rate": 1.0,
        "delta": -1.0,
        "positive_rows": 2,
        "negative_rows": 1,
    }
    assert null_by_label["domain_b_seq_20"] == {
        "name": "domain_b_seq_20",
        "positive_null_rate": 0.0,
        "negative_null_rate": 1.0,
        "delta": -1.0,
        "positive_rows": 2,
        "negative_rows": 1,
    }

    user_lift_by_token = {
        (row["feature"], row["token"]): row
        for row in stats["user_feature_label_lift"]
    }
    assert user_lift_by_token[("user_int_feats_1", 2)] == {
        "group": "user_int",
        "feature": "user_int_feats_1",
        "token": 2,
        "support": 1,
        "positive": 1,
        "negative": 0,
        "positive_rate": 1.0,
        "baseline_positive_rate": 0.666667,
        "lift": 1.5,
        "log_odds": 1.098612,
    }
    item_lift_by_token = {
        (row["feature"], row["token"]): row
        for row in stats["item_feature_label_lift"]
    }
    assert item_lift_by_token[("item_int_feats_3", 100)] == {
        "group": "item_int",
        "feature": "item_int_feats_3",
        "token": 100,
        "support": 2,
        "positive": 2,
        "negative": 0,
        "positive_rate": 1.0,
        "baseline_positive_rate": 0.666667,
        "lift": 1.5,
        "log_odds": 1.609438,
    }

    assert stats["categorical_pair_associations"]
    assert all(row["sample_rows"] > 0 for row in stats["categorical_pair_associations"])
    assert "== Null Rate By Label ==" in captured.out
    assert "== Top User Feature Label Lift ==" in captured.out
    assert "== Top Item Feature Label Lift ==" in captured.out
    assert "== Top Categorical Pair Associations ==" in captured.out


def test_online_dataset_eda_runner_streams_full_dataset_by_default(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    log_capture,
) -> None:
    schema_path = tmp_path / "schema.json"
    dataset_path = tmp_path / "demo.parquet"
    _write_schema(schema_path)
    _write_dataset(dataset_path)

    with log_capture.at_level(logging.INFO):
        report = _run_online_eda_runner(
            dataset_path=dataset_path,
            schema_path=schema_path,
        )
    captured = capsys.readouterr()

    assert report["row_count"] == 4
    assert "scan=streaming full" in log_capture.text
    assert "summary: online dataset, 4 rows" in captured.out


def test_online_dataset_eda_runner_honors_explicit_max_rows(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    log_capture,
) -> None:
    schema_path = tmp_path / "schema.json"
    dataset_path = tmp_path / "demo.parquet"
    _write_schema(schema_path)
    _write_dataset(dataset_path)

    with log_capture.at_level(logging.INFO):
        report = _run_online_eda_runner(
            dataset_path=dataset_path,
            schema_path=schema_path,
            max_rows=2,
        )
    captured = capsys.readouterr()

    assert report["row_count"] == 2
    assert "scan=streaming max_rows=2" in log_capture.text
    assert "summary: online dataset, scanned 2/4 rows" in captured.out
    assert "progress first-pass:" in log_capture.text
    assert "progress second-pass:" in log_capture.text


def test_online_dataset_eda_runner_honors_sample_percent(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    log_capture,
) -> None:
    schema_path = tmp_path / "schema.json"
    dataset_path = tmp_path / "demo.parquet"
    _write_schema(schema_path)
    _write_dataset(dataset_path)

    with log_capture.at_level(logging.INFO):
        report = _run_online_eda_runner(
            dataset_path=dataset_path,
            schema_path=schema_path,
            sample_percent=50.0,
        )
    captured = capsys.readouterr()

    assert report["row_count"] == 2
    assert "scan=streaming sample_percent=50.0 max_rows=2" in log_capture.text
    assert "summary: online dataset, scanned 2/4 rows" in captured.out


def test_resolve_schema_path_honors_taac_schema_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    runner_module = _load_online_eda_runner_module()
    dataset_path = tmp_path / "demo.parquet"
    env_schema_path = tmp_path / "env-schema.json"
    _write_schema(env_schema_path)
    dataset_path.write_bytes(b"parquet-placeholder")
    monkeypatch.setenv("TAAC_SCHEMA_PATH", str(env_schema_path))

    resolved = runner_module.resolve_schema_path(dataset_path, None)

    assert resolved == env_schema_path.resolve()
