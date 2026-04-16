from __future__ import annotations

"""CLI for dataset exploratory analysis.

Usage:
    uv run taac-dataset-eda                         # sample dataset
    uv run taac-dataset-eda --dataset path/to/data  # custom path
    uv run taac-dataset-eda --max-rows 5000         # limit rows scanned
"""

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
DEFAULT_FIGURES_DIR = ROOT / "docs" / "assets" / "figures" / "eda"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run dataset EDA and generate ECharts JSON")
    p.add_argument("--dataset", default="TAAC2026/data_sample_1000", help="HF Hub name or local path")
    p.add_argument("--max-rows", type=int, default=10000, help="Max rows to scan (0 = all)")
    p.add_argument("--figures-dir", default=str(DEFAULT_FIGURES_DIR), help="Output directory for ECharts JSON")
    p.add_argument("--json-path", default="", help="Optional JSON stats output path")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    from taac2026.infrastructure.io.console import logger, stderr_console
    from taac2026.infrastructure.io.datasets import iter_dataset_rows
    from taac2026.reporting.dataset_eda import (
        serialize_echarts,
        classify_columns,
        compute_cardinality_ranking,
        compute_column_stats,
        compute_label_distribution,
        compute_sequence_lengths,
        echarts_cardinality,
        echarts_column_layout,
        echarts_coverage_heatmap,
        echarts_cross_edition,
        echarts_edition_comparison,
        echarts_label_distribution,
        echarts_ndcg_decay,
        echarts_null_rates,
        echarts_seq_length_summary,
        echarts_sequence_lengths,
    )

    # ---- Load data --------------------------------------------------------
    logger.info("Loading dataset: {}", args.dataset)
    rows_iter = iter_dataset_rows(args.dataset)

    # Materialize rows (optionally capped) so we can do multiple passes
    materialized: list[dict] = []
    for i, row in enumerate(rows_iter):
        materialized.append(dict(row))
        if args.max_rows and i + 1 >= args.max_rows:
            break
    logger.info("Loaded {} rows", len(materialized))

    # ---- Analyse ----------------------------------------------------------
    logger.info("Classifying columns …")
    col_names = list(materialized[0].keys()) if materialized else []
    groups = classify_columns(col_names)
    stderr_console.print(
        f"[bold]Schema:[/] {groups.total} columns — "
        f"scalar={len(groups.scalar)}, user_int={len(groups.user_int)}, "
        f"user_dense={len(groups.user_dense)}, item_int={len(groups.item_int)}, "
        f"domain_seq={sum(len(v) for v in groups.domain_seq.values())}"
    )

    logger.info("Computing column stats …")
    col_stats = compute_column_stats(iter(materialized))

    logger.info("Computing label distribution …")
    label_dist = compute_label_distribution(iter(materialized))
    for r in label_dist.as_table():
        stderr_console.print(f"  label_type={r['label_type']} ({r['name']}): {r['count']:,}  ({r['ratio']:.2%})")

    logger.info("Computing sequence lengths …")
    seq_stats = compute_sequence_lengths(iter(materialized))
    for domain, st in seq_stats.items():
        s = st.summary()
        if s["count"]:
            stderr_console.print(f"  {domain}: mean={s['mean']:.1f}, median={s['median']:.0f}, p95={s['p95']:.0f}")

    cardinality = compute_cardinality_ranking(col_stats, groups)
    logger.info("Top-5 cardinality: {}", [(r["column"], r["n_unique"]) for r in cardinality[:5]])

    # ---- Generate ECharts JSON ---------------------------------------------
    logger.info("Generating ECharts JSON → {}", figures_dir)

    def _write_ec(name: str, opt: dict) -> None:
        (figures_dir / f"{name}.echarts.json").write_text(
            serialize_echarts(opt), encoding="utf-8",
        )

    _write_ec("label_distribution", echarts_label_distribution(label_dist))
    _write_ec("null_rates", echarts_null_rates(col_stats))
    _write_ec("cardinality", echarts_cardinality(cardinality))
    _write_ec("sequence_lengths", echarts_sequence_lengths(seq_stats))
    _write_ec("coverage_heatmap", echarts_coverage_heatmap(col_stats, groups))
    _write_ec("column_layout", echarts_column_layout(groups))
    _write_ec("ndcg_decay", echarts_ndcg_decay())
    _write_ec("label_cross_edition", echarts_cross_edition())
    _write_ec("edition_comparison", echarts_edition_comparison())
    _write_ec("seq_length_summary", echarts_seq_length_summary(seq_stats))
    logger.info("  ✓ 10 ECharts JSON files")

    # ---- Optional JSON stats ----------------------------------------------
    if args.json_path:
        json_path = Path(args.json_path)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        stats_dict = {
            "rows": len(materialized),
            "columns": groups.total,
            "groups": {
                "scalar": len(groups.scalar),
                "user_int": len(groups.user_int),
                "user_dense": len(groups.user_dense),
                "item_int": len(groups.item_int),
                "domain_seq": {d: len(c) for d, c in groups.domain_seq.items()},
            },
            "label_distribution": label_dist.as_table(),
            "sequence_lengths": {d: st.summary() for d, st in seq_stats.items()},
            "cardinality_top20": cardinality[:20],
        }
        json_path.write_text(json.dumps(stats_dict, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("JSON stats → {}", json_path)

    logger.info("Done ✓")
    return 0
