"""Generate an amplified synthetic PCVR parquet dataset for loader benchmarks."""

from __future__ import annotations

import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import tyro

from taac2026.infrastructure.io.json import dumps
from taac2026.infrastructure.io.rich_output import print_rich_summary
from taac2026.infrastructure.io.streams import write_stdout_line

MIN_SYNTHETIC_MULTIPLIER = 300


@dataclass(slots=True)
class GeneratePCVRSyntheticDatasetArgs:
    source_dir: Path = Path("data/sample_1000_raw")
    output_dir: Path = Path("outputs/perf/pcvr_synthetic_300x")
    multiplier: int = MIN_SYNTHETIC_MULTIPLIER
    row_group_size: int = 0
    compression: str = "snappy"
    no_jitter_ids: bool = False
    force: bool = False
    json: bool = False


def _replace_column(table: pa.Table, name: str, values: pa.ChunkedArray) -> pa.Table:
    index = table.schema.get_field_index(name)
    if index < 0:
        return table
    field = table.schema.field(index)
    return table.set_column(index, field, values.cast(field.type))


def _offset_scalar_column(table: pa.Table, name: str, offset: int) -> pa.Table:
    index = table.schema.get_field_index(name)
    if index < 0 or offset == 0:
        return table
    field = table.schema.field(index)
    if not pa.types.is_integer(field.type):
        return table
    values = pc.add(table.column(name), pa.scalar(offset, type=field.type))
    return _replace_column(table, name, values)


def _make_repeat_chunk(
    table: pa.Table, repeat_index: int, *, jitter_ids: bool
) -> pa.Table:
    if not jitter_ids or repeat_index == 0:
        return table

    rows = table.num_rows
    id_offset = repeat_index * rows
    time_offset = repeat_index * 86_400
    chunk = table
    chunk = _offset_scalar_column(chunk, "user_id", id_offset)
    chunk = _offset_scalar_column(chunk, "item_id", id_offset)
    chunk = _offset_scalar_column(chunk, "timestamp", time_offset)
    chunk = _offset_scalar_column(chunk, "label_time", time_offset)
    return chunk


def generate_dataset(
    *,
    source_dir: Path,
    output_dir: Path,
    multiplier: int,
    row_group_size: int | None,
    compression: str,
    jitter_ids: bool,
    force: bool,
) -> dict[str, object]:
    source_parquet = source_dir / "demo_1000.parquet"
    source_schema = source_dir / "schema.json"
    if not source_parquet.exists():
        raise FileNotFoundError(f"source parquet not found: {source_parquet}")
    if not source_schema.exists():
        raise FileNotFoundError(f"source schema not found: {source_schema}")
    if multiplier < MIN_SYNTHETIC_MULTIPLIER:
        raise ValueError(
            f"synthetic PCVR benchmark dataset multiplier must be at least "
            f"{MIN_SYNTHETIC_MULTIPLIER}x"
        )

    if output_dir.exists():
        if not force:
            raise FileExistsError(f"output directory already exists: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    table = pq.read_table(source_parquet)
    output_parquet = output_dir / f"demo_{table.num_rows * multiplier}.parquet"
    resolved_row_group_size = row_group_size or table.num_rows

    with pq.ParquetWriter(
        output_parquet, table.schema, compression=compression
    ) as writer:
        for repeat_index in range(multiplier):
            chunk = _make_repeat_chunk(table, repeat_index, jitter_ids=jitter_ids)
            writer.write_table(chunk, row_group_size=resolved_row_group_size)

    shutil.copy2(source_schema, output_dir / "schema.json")
    parquet_file = pq.ParquetFile(output_parquet)
    return {
        "output_dir": str(output_dir),
        "parquet_path": str(output_parquet),
        "schema_path": str(output_dir / "schema.json"),
        "rows": parquet_file.metadata.num_rows,
        "row_groups": parquet_file.metadata.num_row_groups,
        "source_rows": table.num_rows,
        "multiplier": multiplier,
        "row_group_size": resolved_row_group_size,
        "compression": compression,
    }


def parse_args(argv: Sequence[str] | None = None) -> GeneratePCVRSyntheticDatasetArgs:
    return tyro.cli(GeneratePCVRSyntheticDatasetArgs, description=__doc__, args=argv)


def _format_dataset_summary(summary: dict[str, object]) -> None:
    fields = [
        ("Output dir", str(summary.get("output_dir", "<unknown>"))),
        ("Parquet", str(summary.get("parquet_path", "<unknown>"))),
        ("Schema", str(summary.get("schema_path", "<unknown>"))),
        ("Rows", f"{summary.get('rows', 0):,}"),
        ("Row groups", str(summary.get("row_groups", 0))),
        ("Source rows", f"{summary.get('source_rows', 0):,}"),
        ("Multiplier", f"{summary.get('multiplier', 0)}x"),
        ("Row group sz", str(summary.get("row_group_size", 0))),
        ("Compression", str(summary.get("compression", "<unknown>"))),
    ]
    print_rich_summary("PCVR synthetic dataset generated", fields, border_style="blue")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    summary = generate_dataset(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        multiplier=args.multiplier,
        row_group_size=args.row_group_size or None,
        compression=args.compression,
        jitter_ids=not args.no_jitter_ids,
        force=args.force,
    )
    if args.json:
        write_stdout_line(dumps(summary, indent=2))
    else:
        _format_dataset_summary(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
