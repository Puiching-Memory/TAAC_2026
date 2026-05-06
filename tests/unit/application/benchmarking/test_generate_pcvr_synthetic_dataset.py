from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from taac2026.application.benchmarking.generate_pcvr_synthetic_dataset import (
    MIN_SYNTHETIC_MULTIPLIER,
    generate_dataset,
    parse_args,
)


def _write_source_dataset(source_dir: Path) -> None:
    source_dir.mkdir(parents=True)
    table = pa.table(
        {
            "timestamp": [100, 200],
            "label_type": [1, 2],
            "user_id": [10, 20],
            "item_id": [30, 40],
        }
    )
    pq.write_table(table, source_dir / "demo_1000.parquet", row_group_size=2)
    (source_dir / "schema.json").write_text("{}", encoding="utf-8")


def test_parse_args_defaults_to_300x_augmented_dataset() -> None:
    args = parse_args([])

    assert args.multiplier == MIN_SYNTHETIC_MULTIPLIER
    assert args.output_dir == Path("outputs/perf/pcvr_synthetic_300x")


def test_generate_dataset_enforces_minimum_online_perf_multiplier(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    _write_source_dataset(source_dir)

    with pytest.raises(ValueError, match="at least 300x"):
        generate_dataset(
            source_dir=source_dir,
            output_dir=tmp_path / "too_small",
            multiplier=MIN_SYNTHETIC_MULTIPLIER - 1,
            row_group_size=2,
            compression="snappy",
            jitter_ids=True,
            force=False,
        )


def test_generate_dataset_writes_300x_rows_and_schema(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    output_dir = tmp_path / "synthetic_300x"
    _write_source_dataset(source_dir)

    summary = generate_dataset(
        source_dir=source_dir,
        output_dir=output_dir,
        multiplier=MIN_SYNTHETIC_MULTIPLIER,
        row_group_size=2,
        compression="snappy",
        jitter_ids=True,
        force=False,
    )

    assert summary["source_rows"] == 2
    assert summary["rows"] == 2 * MIN_SYNTHETIC_MULTIPLIER
    assert summary["multiplier"] == MIN_SYNTHETIC_MULTIPLIER
    assert Path(summary["schema_path"]).read_text(encoding="utf-8") == "{}"
    parquet_file = pq.ParquetFile(summary["parquet_path"])
    assert parquet_file.metadata.num_rows == 2 * MIN_SYNTHETIC_MULTIPLIER
