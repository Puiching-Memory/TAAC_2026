from __future__ import annotations

import importlib.util
import logging
from pathlib import Path
import sys

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from tests.support.paths import locate_repo_root
from taac2026.infrastructure.io.json import dumps, loads


RESULT_PREFIX = "ONLINE_DATASET_EDA_RESULT="


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


def _stdout_payload(captured: pytest.CaptureFixture[str] | str) -> dict[str, object]:
    output = captured if isinstance(captured, str) else captured.readouterr().out
    for line in output.splitlines():
        if line.startswith(RESULT_PREFIX):
            return loads(line.removeprefix(RESULT_PREFIX))
    raise AssertionError(f"missing {RESULT_PREFIX!r} line in stdout: {output}")


def _run_online_eda_runner(
    *,
    dataset_path: Path,
    schema_path: Path,
    output_dir: Path | None = None,
    reference_profile_path: Path | None = None,
    dataset_role: str = "online",
    max_rows: int | None = None,
    sample_percent: float | None = None,
    config_overrides: dict[str, object] | None = None,
) -> dict[str, object]:
    runner_module = _load_online_eda_runner_module()
    overrides = config_overrides or {}
    config = runner_module.OnlineDatasetEDAConfig(
        dataset_path=dataset_path.resolve(),
        schema_path=schema_path.resolve(),
        output_dir=output_dir.resolve() if output_dir is not None else None,
        reference_profile_path=reference_profile_path.resolve() if reference_profile_path is not None else None,
        dataset_role=dataset_role,
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
    payload = _stdout_payload(captured.out)
    assert payload["row_count"] == 4
    assert payload["stats"]["null_rates"]
    assert payload["stats"]["token_overlap_sketch"]
    assert payload["approximation"]["token_overlap"]["k"] == 256


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
    payload = _stdout_payload(captured.out)
    assert payload["stats"]["label_distribution"] == stats["label_distribution"]
    assert payload["stats"]["sequence_token_cardinality"] == stats["sequence_token_cardinality"]
    assert payload["stats"]["cross_domain_coverage"] == stats["cross_domain_coverage"]


def test_online_dataset_eda_runner_can_enable_label_lift(
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
                "enable_label_lift": True,
                "label_feature_min_support": 1,
                "label_feature_top_k": 10,
            },
        )
    captured = capsys.readouterr()
    stats = report["stats"]

    user_lift_by_token = {
        (row["feature"], row["token"]): row
        for row in stats["user_feature_label_lift"]
    }
    assert user_lift_by_token[("user_int_feats_1", 2)] == {
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

    payload = _stdout_payload(captured.out)
    assert payload["stats"]["user_feature_label_lift"] == stats["user_feature_label_lift"]
    assert payload["stats"]["item_feature_label_lift"] == stats["item_feature_label_lift"]
    assert stats["categorical_pair_associations"] == []


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
    assert "scan=arrow-profile full" in log_capture.text
    assert _stdout_payload(captured.out)["row_count"] == 4


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
    assert "scan=arrow-profile max_rows=2" in log_capture.text
    payload = _stdout_payload(captured.out)
    assert payload["row_count"] == 2
    assert payload["total_rows"] == 4
    assert "progress profile-scan:" in log_capture.text


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
    assert "scan=arrow-profile sample_percent=50.0 max_rows=2" in log_capture.text
    assert _stdout_payload(captured.out)["row_count"] == 2


def test_online_dataset_eda_runner_prints_profile_and_compares_reference(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    log_capture,
) -> None:
    schema_path = tmp_path / "schema.json"
    dataset_path = tmp_path / "demo.parquet"
    train_dir = tmp_path / "train_profile"
    infer_dir = tmp_path / "infer_profile"
    _write_schema(schema_path)
    _write_dataset(dataset_path)

    with log_capture.at_level(logging.INFO):
        train_report = _run_online_eda_runner(
            dataset_path=dataset_path,
            schema_path=schema_path,
            output_dir=train_dir,
            dataset_role="train",
            max_rows=4,
        )
    capsys.readouterr()

    profile_path = tmp_path / "train-profile.json"
    profile_path.write_text(dumps(train_report, trailing_newline=True), encoding="utf-8")

    with log_capture.at_level(logging.INFO):
        infer_report = _run_online_eda_runner(
            dataset_path=dataset_path,
            schema_path=schema_path,
            output_dir=infer_dir,
            reference_profile_path=profile_path,
            dataset_role="infer",
            max_rows=4,
        )
    captured = capsys.readouterr()

    assert infer_report["dataset_role"] == "infer"
    assert infer_report["comparison"]["schema_signature_match"] is True
    assert infer_report["comparison"]["risk_flags"] == []
    assert infer_report["comparison"]["token_overlap_drift"]
    payload = _stdout_payload(captured.out)
    assert payload["comparison"]["schema_signature_match"] is True


def test_online_dataset_eda_runner_compares_reference_from_stdout_json(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    schema_path = tmp_path / "schema.json"
    dataset_path = tmp_path / "dataset.parquet"
    _write_schema(schema_path)
    _write_dataset(dataset_path)

    _run_online_eda_runner(
        dataset_path=dataset_path,
        schema_path=schema_path,
        output_dir=tmp_path / "train",
        dataset_role="train",
    )
    train_payload = _stdout_payload(capsys.readouterr().out)

    infer_report = _run_online_eda_runner(
        dataset_path=dataset_path,
        schema_path=schema_path,
        output_dir=tmp_path / "infer",
        dataset_role="infer",
        config_overrides={"reference_profile_json": RESULT_PREFIX + dumps(train_payload)},
    )

    captured = capsys.readouterr()
    assert infer_report["reference_profile_source"] == "ONLINE_EDA_REFERENCE_PROFILE_JSON"
    assert infer_report["comparison"]["schema_signature_match"] is True
    assert _stdout_payload(captured.out)["reference_profile_source"] == "ONLINE_EDA_REFERENCE_PROFILE_JSON"


def test_resolve_schema_path_honors_taac_schema_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    runner_module = _load_online_eda_runner_module()
    dataset_path = tmp_path / "demo.parquet"
    env_schema_path = tmp_path / "env-schema.json"
    _write_schema(env_schema_path)
    dataset_path.write_bytes(b"parquet-placeholder")
    monkeypatch.setenv("TAAC_SCHEMA_PATH", str(env_schema_path))

    resolved = runner_module.resolve_schema_path(dataset_path, None)

    assert resolved == env_schema_path.resolve()
