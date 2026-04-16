from __future__ import annotations

"""Reusable dataset analysis primitives for TAAC 2026.

This module provides composable functions that inspect a loaded dataset and
return plain-data summary dicts.  ECharts option generators accept those
summaries and produce JSON-serialisable dicts for interactive visualisation.
Both layers are designed for use by CLI scripts *and* notebook cells.
"""

import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np

from taac2026.domain.config import DEFAULT_SEQUENCE_NAMES

# ---------------------------------------------------------------------------
# Constants – column naming conventions mirrored from tests/support.py
# ---------------------------------------------------------------------------

_USER_INT_PREFIX = "user_int_feats_"
_USER_DENSE_PREFIX = "user_dense_feats_"
_ITEM_INT_PREFIX = "item_int_feats_"
_TIMESTAMP_FEATURE_SUFFIX = "_99"
_DOMAIN_SEQ_PREFIXES: dict[str, str] = {
    d: f"{d}_seq_" for d in DEFAULT_SEQUENCE_NAMES
}

# ---------------------------------------------------------------------------
# Caps – prevent OOM / excessive compute during streaming scans
# ---------------------------------------------------------------------------
_MAX_AUC_PAIRS = 50_000          # max pos×neg pairs in single-feature AUC
_MAX_AUC_PER_COL = 50_000        # max (value, label) samples per column for AUC
_MAX_DENSE_VALUES = 100_000      # max dense-feature values collected per column
_MAX_MISSING_VECTORS = 50_000    # max per-row missing-set vectors kept
_REPEAT_RATE_CAP = 500           # max items inspected for repeat-rate calc
_DEFAULT_LABEL_COL = "label_type"

# ---------------------------------------------------------------------------
# 1. Schema summary
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ColumnGroups:
    """Column names classified into semantic groups."""

    scalar: list[str] = field(default_factory=list)
    user_int: list[str] = field(default_factory=list)
    user_dense: list[str] = field(default_factory=list)
    item_int: list[str] = field(default_factory=list)
    domain_seq: dict[str, list[str]] = field(default_factory=dict)

    @property
    def total(self) -> int:
        domain_count = sum(len(v) for v in self.domain_seq.values())
        return len(self.scalar) + len(self.user_int) + len(self.user_dense) + len(self.item_int) + domain_count


def classify_columns(column_names: Sequence[str]) -> ColumnGroups:
    """Split dataset column names into semantic groups."""
    groups = ColumnGroups(domain_seq={d: [] for d in DEFAULT_SEQUENCE_NAMES})
    for col in column_names:
        if col.startswith(_USER_INT_PREFIX):
            groups.user_int.append(col)
        elif col.startswith(_USER_DENSE_PREFIX):
            groups.user_dense.append(col)
        elif col.startswith(_ITEM_INT_PREFIX):
            groups.item_int.append(col)
        else:
            matched = False
            for domain, prefix in _DOMAIN_SEQ_PREFIXES.items():
                if col.startswith(prefix):
                    groups.domain_seq.setdefault(domain, []).append(col)
                    matched = True
                    break
            if not matched:
                groups.scalar.append(col)
    for key in groups.domain_seq:
        groups.domain_seq[key].sort()
    groups.user_int.sort()
    groups.user_dense.sort()
    groups.item_int.sort()
    groups.scalar.sort()
    return groups


# ---------------------------------------------------------------------------
# 2. Per-column statistics (streaming, single-pass)
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class ColumnStats:
    name: str
    count: int = 0
    non_null: int = 0
    n_unique: int = 0
    unique_capped: bool = False  # True when n_unique is a lower bound (cap reached)
    is_list: bool = False
    min_val: float | None = None
    max_val: float | None = None
    sum_val: float = 0.0
    sum_len: int = 0  # for list columns

    @property
    def null_rate(self) -> float:
        return 1.0 - self.non_null / self.count if self.count else 0.0

    @property
    def mean_val(self) -> float | None:
        return self.sum_val / self.non_null if self.non_null else None

    @property
    def mean_list_len(self) -> float | None:
        return self.sum_len / self.non_null if self.non_null and self.is_list else None


# Prefixes whose columns should skip unique-value tracking (dense vectors, sequences)
_SKIP_UNIQUE_PREFIXES: tuple[str, ...] = (_USER_DENSE_PREFIX,) + tuple(_DOMAIN_SEQ_PREFIXES.values())
# High-cardinality identifier/time columns that should always skip unique tracking
_SKIP_UNIQUE_EXACT: frozenset[str] = frozenset({"user_id", "item_id", "timestamp", "label_time"})


def _should_skip_unique(col: str) -> bool:
    """Return True if *col* should skip unique-value tracking.

    Skips dense/sequence prefix columns as well as known high-cardinality
    identifier/time columns (``user_id``, ``item_id``, ``timestamp``,
    ``label_time``).
    """
    return col in _SKIP_UNIQUE_EXACT or any(col.startswith(p) for p in _SKIP_UNIQUE_PREFIXES)


def _update_col_stat(
    col: str,
    value: Any,
    stats: ColumnStats,
    unique_set: set[Any] | None,
    max_unique_track: int,
    *,
    _isfinite: Any = math.isfinite,
) -> None:
    """Shared per-value column-stats accumulation logic."""
    stats.count += 1
    if value is None:
        return
    stats.non_null += 1
    if isinstance(value, (list, tuple)):
        stats.is_list = True
        stats.sum_len += len(value)
        if unique_set is not None and len(unique_set) < max_unique_track:
            for elem in value:
                if len(unique_set) >= max_unique_track:
                    break
                try:
                    unique_set.add(elem)
                except TypeError:
                    continue
    else:
        if unique_set is not None and len(unique_set) < max_unique_track:
            unique_set.add(value)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if _isfinite(value):
                stats.sum_val += value
                if stats.min_val is None or value < stats.min_val:
                    stats.min_val = value
                if stats.max_val is None or value > stats.max_val:
                    stats.max_val = value


def _probe_seq_row(
    row: dict[str, Any],
    domain: str,
    prefix: str,
    probe_cols: dict[str, str | None],
    seq_stats: dict[str, SequenceLengthStats],
) -> None:
    """Shared per-row sequence-length probing logic."""
    probe = probe_cols[domain]
    if probe is None:
        for col in row:
            if col.startswith(prefix):
                probe_cols[domain] = col
                probe = col
                break
    if probe is None:
        seq_stats[domain].lengths.append(0)
        return
    seq = row.get(probe)
    if seq is None or not isinstance(seq, (list, tuple)):
        seq_stats[domain].lengths.append(0)
    else:
        seq_stats[domain].lengths.append(len(seq))


def compute_column_stats(
    rows: Iterable[dict[str, Any]],
    columns: Sequence[str] | None = None,
    *,
    max_unique_track: int = 200_000,
) -> dict[str, ColumnStats]:
    """Compute per-column statistics in a single streaming pass."""
    accumulators: dict[str, ColumnStats] = {}
    unique_sets: dict[str, set[Any] | None] = {}

    for row in rows:
        if columns is None:
            row_items = row.items()
        else:
            row_items = ((col, row.get(col)) for col in columns)

        for col, value in row_items:
            if col not in accumulators:
                accumulators[col] = ColumnStats(name=col)
                unique_sets[col] = None if _should_skip_unique(col) else set()
            _update_col_stat(col, value, accumulators[col], unique_sets[col], max_unique_track)

    for col, uniques in unique_sets.items():
        if uniques is not None:
            accumulators[col].n_unique = len(uniques)
            if len(uniques) >= max_unique_track:
                accumulators[col].unique_capped = True

    return accumulators


# ---------------------------------------------------------------------------
# 3. Label / action-type distribution
# ---------------------------------------------------------------------------

_LABEL_NAMES: dict[int, str] = {0: "曝光", 1: "点击", 2: "转化"}


@dataclass(slots=True)
class LabelDistribution:
    counts: Counter[int] = field(default_factory=Counter)
    total: int = 0

    def add(self, label: int | None) -> None:
        self.total += 1
        if label is not None:
            self.counts[label] = self.counts.get(label, 0) + 1

    def as_table(self) -> list[dict[str, Any]]:
        rows = []
        for label in sorted(self.counts):
            count = self.counts[label]
            rows.append({
                "label_type": label,
                "name": _LABEL_NAMES.get(label, f"unknown_{label}"),
                "count": count,
                "ratio": count / self.total if self.total else 0.0,
            })
        return rows


def compute_label_distribution(rows: Iterable[dict[str, Any]], label_col: str = _DEFAULT_LABEL_COL) -> LabelDistribution:
    dist = LabelDistribution()
    for row in rows:
        dist.add(row.get(label_col))
    return dist


# ---------------------------------------------------------------------------
# 4. Sequence-length analysis
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class SequenceLengthStats:
    domain: str
    lengths: list[int] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.lengths)

    def summary(self) -> dict[str, Any]:
        if not self.lengths:
            return {"domain": self.domain, "count": 0}
        arr = np.array(self.lengths)
        return {
            "domain": self.domain,
            "count": len(arr),
            "min": int(arr.min()),
            "max": int(arr.max()),
            "mean": float(arr.mean()),
            "median": float(np.median(arr)),
            "p25": float(np.percentile(arr, 25)),
            "p75": float(np.percentile(arr, 75)),
            "p95": float(np.percentile(arr, 95)),
            "zero_rate": float((arr == 0).mean()),
        }


def compute_sequence_lengths(
    rows: Iterable[dict[str, Any]],
    domain_prefixes: dict[str, str] | None = None,
) -> dict[str, SequenceLengthStats]:
    prefixes = domain_prefixes or _DOMAIN_SEQ_PREFIXES
    result: dict[str, SequenceLengthStats] = {d: SequenceLengthStats(domain=d) for d in prefixes}
    _probe_cols: dict[str, str | None] = {d: None for d in prefixes}

    for row in rows:
        for domain, prefix in prefixes.items():
            _probe_seq_row(row, domain, prefix, _probe_cols, result)

    return result


# ---------------------------------------------------------------------------
# 5. Feature cardinality ranking
# ---------------------------------------------------------------------------

def compute_cardinality_ranking(
    stats: dict[str, ColumnStats],
    groups: ColumnGroups,
) -> list[dict[str, Any]]:
    """Rank sparse features by unique-value count (descending)."""
    sparse_cols = groups.user_int + groups.item_int
    ranking = []
    for col in sparse_cols:
        st = stats.get(col)
        if st is None:
            continue
        ranking.append({
            "column": col,
            "n_unique": st.n_unique,
            "coverage": 1.0 - st.null_rate,
            "group": "user" if col.startswith(_USER_INT_PREFIX) else "item",
        })
    ranking.sort(key=lambda r: r["n_unique"], reverse=True)
    return ranking


# ---------------------------------------------------------------------------
# 6. ECharts interactive option generators
# ---------------------------------------------------------------------------

_EC_COLORS = ["#89b4fa", "#f38ba8", "#a6e3a1", "#fab387", "#cba6f7", "#f9e2af"]


def echarts_label_distribution(dist: LabelDistribution) -> dict[str, Any]:
    """ECharts option for label-type pie chart."""
    table = dist.as_table()
    return {
        "tooltip": {"trigger": "item", "formatter": "{b}: {c} ({d}%)"},
        "legend": {"bottom": 0},
        "series": [{
            "type": "pie",
            "radius": ["35%", "65%"],
            "avoidLabelOverlap": True,
            "itemStyle": {"borderRadius": 6, "borderWidth": 2},
            "label": {"formatter": "{b}\n{d}%"},
            "data": [
                {"name": r["name"], "value": r["count"]}
                for r in table
            ],
            "color": _EC_COLORS,
        }],
    }


def echarts_cardinality(ranking: list[dict[str, Any]], *, top_n: int = 25) -> dict[str, Any]:
    """ECharts option for horizontal bar chart of sparse-feature cardinalities."""
    # Filter out zero-cardinality entries (log axis cannot display 0)
    data = list(reversed([r for r in ranking[:top_n] if r.get("n_unique", 0) > 0]))
    names = [
        r["column"].replace("user_int_feats_", "u_").replace("item_int_feats_", "i_")
        for r in data
    ]
    values = [r["n_unique"] for r in data]
    colors = [_EC_COLORS[0] if r["group"] == "user" else _EC_COLORS[2] for r in data]
    return {
        "_height": "600px",
        "tooltip": {"trigger": "axis", "axisPointer": {"type": "shadow"}},
        "grid": {"left": 90, "right": 30, "top": 40, "bottom": 30},
        "xAxis": {"type": "log", "name": "基数 (unique values)"},
        "yAxis": {"type": "category", "data": names, "axisLabel": {"fontSize": 10}},
        "series": [{
            "type": "bar",
            "data": [{"value": v, "itemStyle": {"color": c}} for v, c in zip(values, colors)],
            "label": {"show": True, "position": "right", "formatter": "{c}"},
        }],
    }


def echarts_sequence_lengths(seq_stats: dict[str, SequenceLengthStats]) -> dict[str, Any]:
    """ECharts option for per-domain sequence-length box-style chart."""
    domains = [d for d in seq_stats if seq_stats[d].lengths]
    summaries = [seq_stats[d].summary() for d in domains]
    return {
        "tooltip": {"trigger": "axis"},
        "legend": {"data": domains},
        "grid": {"left": 60, "right": 30, "top": 60, "bottom": 40},
        "xAxis": {"type": "category", "data": ["min", "P25", "median", "mean", "P75", "P95", "max"]},
        "yAxis": {"type": "value", "name": "序列长度"},
        "series": [
            {
                "name": d,
                "type": "line",
                "smooth": True,
                "symbol": "circle",
                "symbolSize": 8,
                "data": [
                    s["min"], int(s["p25"]), int(s["median"]),
                    int(s["mean"]), int(s["p75"]), int(s["p95"]), s["max"],
                ],
                "itemStyle": {"color": _EC_COLORS[i % len(_EC_COLORS)]},
                "areaStyle": {"opacity": 0.08},
            }
            for i, (d, s) in enumerate(zip(domains, summaries))
            if s["count"] > 0
        ],
    }


def echarts_coverage_heatmap(
    stats: dict[str, ColumnStats], groups: ColumnGroups,
) -> dict[str, Any]:
    """ECharts option for sparse-feature coverage heatmap."""
    cols = groups.user_int + groups.item_int
    names = [c.replace("user_int_feats_", "u_").replace("item_int_feats_", "i_") for c in cols]
    coverages = [round(1.0 - stats[c].null_rate, 3) if c in stats else 0.0 for c in cols]
    data = [[i, 0, coverages[i]] for i in range(len(cols))]
    return {
        "_height": "180px",
        "tooltip": {},
        "grid": {"left": 60, "right": 30, "top": 10, "bottom": 60},
        "xAxis": {"type": "category", "data": names, "axisLabel": {"rotate": 60, "fontSize": 7}},
        "yAxis": {"type": "category", "data": ["coverage"], "show": False},
        "visualMap": {
            "min": 0, "max": 1, "show": True, "orient": "horizontal",
            "left": "center", "bottom": 0,
            "inRange": {"color": ["#f38ba8", "#f9e2af", "#a6e3a1"]},
            "text": ["100%", "0%"],
        },
        "series": [{
            "type": "heatmap",
            "data": data,
            "label": {"show": len(cols) <= 30},
        }],
    }


def echarts_ndcg_decay(k_max: int = 10) -> dict[str, Any]:
    """ECharts option for NDCG@K position-discount curve."""
    ks = list(range(1, k_max + 1))
    gains = [round(1.0 / math.log2(k + 1), 4) for k in ks]
    return {
        "tooltip": {"trigger": "axis"},
        "grid": {"left": 60, "right": 30, "top": 50, "bottom": 40},
        "xAxis": {"type": "category", "data": [str(k) for k in ks], "name": "排名位置 k"},
        "yAxis": {"type": "value", "name": "1/log₂(k+1)", "max": 1.15},
        "series": [{
            "type": "line",
            "data": gains,
            "smooth": False,
            "symbol": "circle",
            "symbolSize": 10,
            "label": {"show": True, "position": "top", "fontSize": 9},
            "areaStyle": {"opacity": 0.1},
            "itemStyle": {"color": _EC_COLORS[0]},
        }],
    }


def echarts_cross_edition(
    current_distribution: dict[str, float] | None = None,
    *,
    current_label: str = "本届 sample",
) -> dict[str, Any]:
    """ECharts option for cross-edition label distribution comparison.

    When *current_distribution* is provided, the last category is populated
    from the current run's label percentages (keys: ``曝光`` / ``点击`` / ``转化``).
    If omitted, the chart falls back to static sample values for compatibility.
    """
    current = current_distribution or {"曝光": 0.0, "点击": 87.6, "转化": 12.4}
    return {
        "tooltip": {"trigger": "axis", "axisPointer": {"type": "shadow"}},
        "legend": {"data": ["曝光", "点击", "转化"]},
        "grid": {"left": 60, "right": 30, "top": 50, "bottom": 30},
        "xAxis": {"type": "category", "data": ["上届 1M", "上届 10M", current_label]},
        "yAxis": {"type": "value", "name": "占比 (%)", "max": 100},
        "series": [
            {"name": "曝光", "type": "bar", "data": [90.19, 94.63, float(current.get("曝光", 0.0))], "itemStyle": {"color": _EC_COLORS[0]}},
            {"name": "点击", "type": "bar", "data": [9.81, 2.85, float(current.get("点击", 0.0))], "itemStyle": {"color": _EC_COLORS[1]}},
            {"name": "转化", "type": "bar", "data": [0, 2.52, float(current.get("转化", 0.0))], "itemStyle": {"color": _EC_COLORS[2]}},
        ],
    }


def echarts_column_layout(groups: ColumnGroups) -> dict[str, Any]:
    """ECharts option for column-group donut chart."""
    pieces = [
        ("标量列", len(groups.scalar)),
        ("user_int", len(groups.user_int)),
        ("user_dense", len(groups.user_dense)),
        ("item_int", len(groups.item_int)),
    ]
    for domain, cols in sorted(groups.domain_seq.items()):
        pieces.append((f"{domain}_seq", len(cols)))
    return {
        "tooltip": {"trigger": "item", "formatter": "{b}: {c} 列 ({d}%)"},
        "legend": {"bottom": 0, "type": "scroll"},
        "series": [{
            "type": "pie",
            "radius": ["35%", "65%"],
            "avoidLabelOverlap": True,
            "itemStyle": {"borderRadius": 6, "borderWidth": 2},
            "label": {"formatter": "{b}\n{c} ({d}%)"},
            "data": [{"name": n, "value": v} for n, v in pieces],
            "color": _EC_COLORS,
        }],
    }


def echarts_null_rates(stats: dict[str, ColumnStats], *, top_n: int = 30) -> dict[str, Any]:
    """ECharts option for horizontal bar chart of columns with highest null rates."""
    items = [(s.name, round(s.null_rate, 4)) for s in stats.values() if s.null_rate > 0]
    items.sort(key=lambda x: x[1], reverse=True)
    items = items[:top_n]
    names = [n for n, _ in reversed(items)]
    rates = [r for _, r in reversed(items)]
    return {
        "_height": f"{max(300, len(items) * 22)}px",
        "tooltip": {"trigger": "axis", "axisPointer": {"type": "shadow"}},
        "grid": {"left": 120, "right": 40, "top": 30, "bottom": 30},
        "xAxis": {"type": "value", "name": "缺失率", "max": 1.0},
        "yAxis": {"type": "category", "data": names, "axisLabel": {"fontSize": 8}},
        "series": [{
            "type": "bar",
            "data": rates,
            "itemStyle": {"color": _EC_COLORS[1]},
            "label": {"show": True, "position": "right", "formatter": "{c}"},
            "markLine": {
                "silent": True,
                "data": [{"xAxis": 0.5, "lineStyle": {"color": _EC_COLORS[4], "type": "dashed"}}],
            },
        }],
    }


def echarts_edition_comparison(
    groups: ColumnGroups | None = None,
    seq_stats: dict[str, SequenceLengthStats] | None = None,
) -> dict[str, Any]:
    """ECharts option for cross-edition dataset dimension comparison.

    When *groups* and *seq_stats* are supplied the TAAC 2026 values are
    computed dynamically from the scan results.  Otherwise static fallback
    values are used (with a subtitle note).
    """
    metrics = ["用户特征数", "物品特征数", "行为域数", "总列数", "序列最大长度", "序列均值(主域)"]
    taac2025 = [8, 12, 1, 20, 100, 94]  # fixed historical baseline

    if groups is not None and seq_stats is not None:
        # Derive TAAC 2026 values dynamically from scan results
        seq_max = max(
            (s.summary().get("max", 0) for s in seq_stats.values()),
            default=0,
        )
        primary = next((n for n in DEFAULT_SEQUENCE_NAMES if n in seq_stats), None)
        if primary is None and seq_stats:
            primary = next(iter(seq_stats))
        primary_mean = round(seq_stats[primary].summary().get("mean", 0)) if primary and primary in seq_stats else 0
        taac2026 = [
            len(groups.user_int) + len(groups.user_dense),
            len(groups.item_int),
            len(groups.domain_seq),
            groups.total,
            int(seq_max),
            int(primary_mean),
        ]
        subtitle = "TAAC 2025 为固定对比基线；TAAC 2026 指标根据当前扫描结果动态生成"
    else:
        taac2026 = [56, 14, 4, 120, 3951, 1099]  # static fallback
        subtitle = "TAAC 2025 为固定对比基线；TAAC 2026 为静态示例值"

    all_values = [*taac2025, *taac2026]
    use_log = all(v > 0 for v in all_values)

    return {
        "title": {"text": "TAAC 赛题版本数据规模对比", "subtext": subtitle},
        "tooltip": {"trigger": "axis", "axisPointer": {"type": "shadow"}},
        "legend": {"data": ["TAAC 2025", "TAAC 2026"]},
        "grid": {"left": 100, "right": 40, "top": 70, "bottom": 30},
        "xAxis": {"type": "log" if use_log else "value", "name": "数值 (log)" if use_log else "数值"},
        "yAxis": {"type": "category", "data": metrics, "axisLabel": {"fontSize": 9}},
        "series": [
            {
                "name": "TAAC 2025",
                "type": "bar",
                "data": taac2025,
                "itemStyle": {"color": _EC_COLORS[4]},
                "label": {"show": True, "position": "right", "formatter": "{c}"},
            },
            {
                "name": "TAAC 2026",
                "type": "bar",
                "data": taac2026,
                "itemStyle": {"color": _EC_COLORS[0]},
                "label": {"show": True, "position": "right", "formatter": "{c}"},
            },
        ],
    }


def echarts_seq_length_summary(seq_stats: dict[str, SequenceLengthStats]) -> dict[str, Any]:
    """ECharts option for per-domain sequence-length box-style summary."""
    domains = [d for d in seq_stats if seq_stats[d].lengths]
    summaries = [seq_stats[d].summary() for d in domains]
    if not domains or not summaries:
        return {
            "tooltip": {},
            "legend": {"data": []},
            "radar": {"indicator": []},
            "series": [{"type": "radar", "data": []}],
        }
    # Radar chart for multi-domain comparison
    radar_max = max(s["max"] for s in summaries) * 1.1
    indicators = [
        {"name": "min", "max": radar_max},
        {"name": "P25", "max": radar_max},
        {"name": "median", "max": radar_max},
        {"name": "mean", "max": radar_max},
        {"name": "P75", "max": radar_max},
        {"name": "P95", "max": radar_max},
        {"name": "max", "max": radar_max},
    ]
    radar_data = []
    for i, (d, s) in enumerate(zip(domains, summaries)):
        if s["count"] == 0:
            continue
        radar_data.append({
            "name": d,
            "value": [s["min"], int(s["p25"]), int(s["median"]),
                       int(s["mean"]), int(s["p75"]), int(s["p95"]), s["max"]],
            "lineStyle": {"color": _EC_COLORS[i % len(_EC_COLORS)]},
            "itemStyle": {"color": _EC_COLORS[i % len(_EC_COLORS)]},
            "areaStyle": {"opacity": 0.1},
        })
    return {
        "tooltip": {},
        "legend": {"data": [d for d, s in zip(domains, summaries) if s["count"] > 0]},
        "radar": {"indicator": indicators},
        "series": [{
            "type": "radar",
            "data": radar_data,
        }],
    }


# ---------------------------------------------------------------------------
# 7. User-level aggregation
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class UserStats:
    """Per-user behaviour counts and activity distribution."""

    user_behavior_counts: Counter[str]  # user_id → total rows
    user_label_counts: dict[str, Counter[int]]  # user_id → {label → count}
    domain_activity: dict[str, set[str]]  # domain → set of active user_ids

    @property
    def n_users(self) -> int:
        return len(self.user_behavior_counts)

    def activity_distribution(self) -> dict[str, Any]:
        """Return activity-level bins and counts."""
        counts = list(self.user_behavior_counts.values())
        if not counts:
            return {
                "bins": {},
                "total_users": 0,
                "mean_behaviors": 0.0,
                "median_behaviors": 0.0,
                "max_behaviors": 0,
                "min_behaviors": 0,
            }
        arr = np.array(counts)
        bins = {"1": 0, "2-5": 0, "6-20": 0, "21-100": 0, "100+": 0}
        for c in counts:
            if c == 1:
                bins["1"] += 1
            elif c <= 5:
                bins["2-5"] += 1
            elif c <= 20:
                bins["6-20"] += 1
            elif c <= 100:
                bins["21-100"] += 1
            else:
                bins["100+"] += 1
        return {
            "bins": bins,
            "total_users": len(counts),
            "mean_behaviors": float(arr.mean()),
            "median_behaviors": float(np.median(arr)),
            "max_behaviors": int(arr.max()),
            "min_behaviors": int(arr.min()),
        }

    def cross_domain_overlap(self) -> dict[str, Any]:
        """Compute user overlap between domains."""
        domains = sorted(self.domain_activity.keys())
        overlap_matrix: list[list[int]] = []
        for d1 in domains:
            row = []
            for d2 in domains:
                s1 = self.domain_activity.get(d1, set())
                s2 = self.domain_activity.get(d2, set())
                row.append(len(s1 & s2))
            overlap_matrix.append(row)
        # Users active in N domains
        user_domain_counts: Counter[int] = Counter()
        all_users = set()
        for s in self.domain_activity.values():
            all_users |= s
        for uid in all_users:
            n = sum(1 for s in self.domain_activity.values() if uid in s)
            user_domain_counts[n] += 1
        return {
            "domains": domains,
            "overlap_matrix": overlap_matrix,
            "user_domain_count_dist": dict(sorted(user_domain_counts.items())),
        }


# ---------------------------------------------------------------------------
# 8. Label-conditional feature analysis
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class LabelConditionalStats:
    """Feature statistics conditioned on positive/negative label."""

    # col → {0: (null_count, total), 1: (null_count, total)}
    conditional_nulls: dict[str, dict[int, tuple[int, int]]]
    # single-feature AUC: col → auc_value
    feature_auc: dict[str, float]

    def null_rate_diff(self, *, top_n: int = 30) -> list[dict[str, Any]]:
        """Return features with largest null-rate difference between labels."""
        diffs = []
        for col, label_map in self.conditional_nulls.items():
            rates = {}
            for label, (null_count, total) in label_map.items():
                rates[label] = null_count / total if total else 0.0
            if len(rates) < 2:
                continue
            labels_sorted = sorted(rates.keys())
            diff = abs(rates[labels_sorted[-1]] - rates[labels_sorted[0]])
            diffs.append({
                "column": col,
                "null_rate_positive": rates.get(1, 0.0),
                "null_rate_negative": rates.get(0, 0.0),
                "diff": round(diff, 4),
            })
        diffs.sort(key=lambda x: x["diff"], reverse=True)
        return diffs[:top_n]


def _compute_single_feature_auc(values: list[float], labels: list[int]) -> float:
    """Compute AUC for a single numeric feature vs binary label."""
    if not values or not labels:
        return 0.5
    arr_v = np.array(values, dtype=np.float64)
    arr_l = np.array(labels, dtype=np.float64)
    pos = arr_v[arr_l > 0.5]
    neg = arr_v[arr_l <= 0.5]
    if pos.size == 0 or neg.size == 0:
        return 0.5
    # Subsample if too large to avoid O(n^2)
    if pos.size * neg.size > _MAX_AUC_PAIRS:
        rng = np.random.RandomState(42)
        pos_limit = min(pos.size, max(1, int(math.sqrt(_MAX_AUC_PAIRS))))
        neg_limit = min(neg.size, max(1, _MAX_AUC_PAIRS // pos_limit))
        pos = rng.choice(pos, pos_limit, replace=False)
        neg = rng.choice(neg, neg_limit, replace=False)
    margins = pos[:, None] - neg[None, :]
    auc = float(np.mean(margins > 0) + 0.5 * np.mean(margins == 0))
    return auc  # preserve raw AUC so inverse correlations remain visible


# ---------------------------------------------------------------------------
# 9. Dense feature distribution analysis
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class DenseFeatureStats:
    """Distribution summary for dense (continuous) features."""

    # col → summary dict
    distributions: dict[str, dict[str, Any]]

    @staticmethod
    def summarize(values: list[float]) -> dict[str, Any]:
        if not values:
            return {"count": 0}
        arr = np.array(values, dtype=np.float64)
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            return {"count": len(values), "finite_count": 0}
        return {
            "count": len(values),
            "finite_count": int(finite.size),
            "mean": float(finite.mean()),
            "std": float(finite.std()),
            "min": float(finite.min()),
            "p25": float(np.percentile(finite, 25)),
            "median": float(np.median(finite)),
            "p75": float(np.percentile(finite, 75)),
            "max": float(finite.max()),
            "skewness": float(_skewness(finite)),
            "zero_rate": float((finite == 0.0).mean()),
        }


def _skewness(arr: np.ndarray) -> float:
    """Compute skewness (Fisher's definition)."""
    n = arr.size
    if n < 3:
        return 0.0
    mean = arr.mean()
    std = arr.std()
    if std == 0:
        return 0.0
    return float(np.mean(((arr - mean) / std) ** 3))


# ---------------------------------------------------------------------------
# 10. Missing value pattern analysis
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class MissingPatternStats:
    """Co-missing pattern analysis for high-null features."""

    # pairs of features that are frequently co-missing
    co_missing_pairs: list[dict[str, Any]]
    # label-conditioned missing: is missing correlated with label?
    label_missing_correlation: dict[str, float]  # col → correlation strength


# ---------------------------------------------------------------------------
# 11. Sequence behaviour patterns (beyond length)
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class SequencePatternStats:
    """Patterns within sequences: repeat rate, diversity."""

    # domain → summary dict
    patterns: dict[str, dict[str, Any]]


# ---------------------------------------------------------------------------
# 12. Cardinality bin distribution
# ---------------------------------------------------------------------------


def compute_cardinality_bins(ranking: list[dict[str, Any]]) -> dict[str, int]:
    """Bin features by cardinality ranges."""
    bins = {"1-10": 0, "11-100": 0, "101-1K": 0, "1K-10K": 0, "10K-100K": 0, "100K+": 0}
    for r in ranking:
        c = r.get("n_unique", 0)
        if c <= 0:
            continue
        if c <= 10:
            bins["1-10"] += 1
        elif c <= 100:
            bins["11-100"] += 1
        elif c <= 1000:
            bins["101-1K"] += 1
        elif c <= 10000:
            bins["1K-10K"] += 1
        elif c <= 100000:
            bins["10K-100K"] += 1
        else:
            bins["100K+"] += 1
    return bins


@dataclass(slots=True)
class DatasetScanResult:
    """Aggregated result from a single streaming pass over the dataset."""

    groups: ColumnGroups
    col_stats: dict[str, ColumnStats]
    label_dist: LabelDistribution
    seq_stats: dict[str, SequenceLengthStats]
    row_count: int
    user_stats: UserStats | None = None
    label_cond_stats: LabelConditionalStats | None = None
    dense_stats: DenseFeatureStats | None = None
    missing_patterns: MissingPatternStats | None = None
    seq_patterns: SequencePatternStats | None = None


def scan_dataset(
    rows: Iterable[dict[str, Any]],
    *,
    max_rows: int = 0,
    max_unique_track: int = 200_000,
    positive_label: int = 2,
    label_col: str = _DEFAULT_LABEL_COL,
) -> DatasetScanResult | None:
    """Compute column stats, label distribution, and sequence lengths in one pass.

    Also collects user-level aggregation, label-conditional feature stats,
    dense feature distributions, missing patterns, and sequence behaviour
    patterns.

    Args:
        positive_label: The label value considered as positive/conversion
            (default ``2`` for CVR).
        label_col: Column name holding the label (default ``"label_type"``).

    Returns ``None`` if no rows were consumed.
    """
    _domain_prefixes = _DOMAIN_SEQ_PREFIXES

    groups: ColumnGroups | None = None
    col_accumulators: dict[str, ColumnStats] = {}
    unique_sets: dict[str, set[Any] | None] = {}
    label_dist = LabelDistribution()
    seq_stats: dict[str, SequenceLengthStats] = {
        d: SequenceLengthStats(domain=d) for d in _domain_prefixes
    }
    seq_probe: dict[str, str | None] = {d: None for d in _domain_prefixes}
    row_count = 0

    # -- New accumulators --
    # User-level
    user_behavior_counts: Counter[str] = Counter()
    user_label_counts: dict[str, Counter[int]] = defaultdict(Counter)
    domain_user_activity: dict[str, set[str]] = {d: set() for d in _domain_prefixes}

    # Label-conditional null tracking
    cond_nulls: dict[str, dict[int, list[int]]] = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    # col → {label: [null_count, total_count]}

    # For single-feature AUC: collect (value, label) for sparse int features
    feature_auc_values: dict[str, list[float]] = defaultdict(list)
    feature_auc_labels: dict[str, list[int]] = defaultdict(list)

    # Dense feature value collection (capped to avoid OOM)
    dense_values: dict[str, list[float]] = defaultdict(list)

    # Missing pattern: track per-row missing sets for high-null features (capped)
    _missing_row_vectors: list[frozenset[str]] = []

    # Sequence patterns: track per-domain item repeat/diversity
    seq_item_repeat_sums: dict[str, list[float]] = {d: [] for d in _domain_prefixes}

    for i, row in enumerate(rows):
        if max_rows and i >= max_rows:
            break
        if not isinstance(row, dict):
            row = dict(row)
        row_count += 1

        # Classify columns on first row
        if groups is None:
            groups = classify_columns(list(row.keys()))
            sparse_cols = groups.user_int + groups.item_int

        # Column stats (delegates to shared helper)
        for col, value in row.items():
            if col not in col_accumulators:
                col_accumulators[col] = ColumnStats(name=col)
                unique_sets[col] = None if _should_skip_unique(col) else set()
            _update_col_stat(col, value, col_accumulators[col], unique_sets[col], max_unique_track)

        # Label distribution
        label_val = row.get(label_col)
        label_dist.add(label_val)

        # Binary label for conditional analysis (1 = positive/conversion, 0 = negative)
        binary_label = 1 if label_val == positive_label else 0

        # --- User-level aggregation ---
        uid = row.get("user_id")
        if uid is not None:
            uid_str = str(uid)
            user_behavior_counts[uid_str] += 1
            if label_val is not None:
                user_label_counts[uid_str][int(label_val)] += 1

        # --- Label-conditional null tracking ---
        if label_val is not None and groups is not None:
            missing_this_row: list[str] = []
            for col in sparse_cols:
                v = row.get(col)
                bucket = cond_nulls[col][binary_label]
                bucket[1] += 1
                if v is None:
                    bucket[0] += 1
                    missing_this_row.append(col)
                # Single-feature AUC (for numeric sparse features)
                elif isinstance(v, (int, float)) and not isinstance(v, bool) and len(feature_auc_values[col]) < _MAX_AUC_PER_COL:
                    feature_auc_values[col].append(float(v))
                    feature_auc_labels[col].append(binary_label)

            if missing_this_row and len(_missing_row_vectors) < _MAX_MISSING_VECTORS:
                _missing_row_vectors.append(frozenset(missing_this_row))

        # --- Dense feature values ---
        if groups is not None:
            for col in groups.user_dense:
                v = row.get(col)
                if v is None:
                    continue
                remaining = _MAX_DENSE_VALUES - len(dense_values[col])
                if remaining <= 0:
                    continue
                if isinstance(v, (list, tuple)):
                    for x in v:
                        if isinstance(x, (int, float)) and not isinstance(x, bool):
                            dense_values[col].append(float(x))
                            remaining -= 1
                            if remaining <= 0:
                                break
                elif isinstance(v, (int, float)) and not isinstance(v, bool):
                    dense_values[col].append(float(v))

        # --- Domain activity tracking + sequence patterns + lengths ---
        for domain, prefix in _domain_prefixes.items():
            probe = seq_probe.get(domain)
            if probe is None:
                for c in row:
                    if c.startswith(prefix) and not c.endswith(_TIMESTAMP_FEATURE_SUFFIX):
                        seq_probe[domain] = c
                        probe = c
                        break
            seq_val = row.get(probe) if probe else None
            if seq_val is not None and isinstance(seq_val, (list, tuple)) and len(seq_val) > 0:
                if uid is not None:
                    domain_user_activity[domain].add(str(uid))
                # Sequence item repeat rate (cap to avoid O(L) set on very long seqs)
                total_items = len(seq_val)
                sample = seq_val[:_REPEAT_RATE_CAP] if total_items > _REPEAT_RATE_CAP else seq_val
                unique_items = len(set(sample))
                repeat_rate = 1.0 - unique_items / len(sample) if len(sample) > 0 else 0.0
                seq_item_repeat_sums[domain].append(repeat_rate)
                # Sequence length
                seq_stats[domain].lengths.append(total_items)
            else:
                seq_stats[domain].lengths.append(0)

    if groups is None:
        return None

    # Finalise unique counts
    for col, uniques in unique_sets.items():
        if uniques is not None:
            col_accumulators[col].n_unique = len(uniques)
            if len(uniques) >= max_unique_track:
                col_accumulators[col].unique_capped = True
    user_stats = UserStats(
        user_behavior_counts=user_behavior_counts,
        user_label_counts=dict(user_label_counts),
        domain_activity=domain_user_activity,
    )

    # --- Build LabelConditionalStats ---
    # Finalise conditional nulls
    final_cond_nulls: dict[str, dict[int, tuple[int, int]]] = {}
    for col, label_map in cond_nulls.items():
        final_cond_nulls[col] = {label: tuple(counts) for label, counts in label_map.items()}  # type: ignore[misc]

    # Compute single-feature AUC
    feature_auc: dict[str, float] = {}
    for col in feature_auc_values:
        vals = feature_auc_values[col]
        labs = feature_auc_labels[col]
        if len(vals) >= 10:
            feature_auc[col] = round(_compute_single_feature_auc(vals, labs), 4)

    label_cond_stats = LabelConditionalStats(
        conditional_nulls=final_cond_nulls,
        feature_auc=feature_auc,
    )

    # --- Build DenseFeatureStats ---
    dense_distributions = {}
    for col, vals in dense_values.items():
        dense_distributions[col] = DenseFeatureStats.summarize(vals)
    dense_stats = DenseFeatureStats(distributions=dense_distributions)

    # --- Build MissingPatternStats ---
    # Co-missing analysis for top null features
    co_missing_pairs: list[dict[str, Any]] = []
    if _missing_row_vectors:
        # Find features that appear in missing vectors most often
        feat_missing_freq: Counter[str] = Counter()
        for mv in _missing_row_vectors:
            for f in mv:
                feat_missing_freq[f] += 1
        top_missing = [f for f, _ in feat_missing_freq.most_common(20)]
        # Count co-missing pairs
        pair_counts: Counter[tuple[str, str]] = Counter()
        for mv in _missing_row_vectors:
            top_in_row = [f for f in top_missing if f in mv]
            for i_idx in range(len(top_in_row)):
                for j_idx in range(i_idx + 1, len(top_in_row)):
                    pair_counts[(top_in_row[i_idx], top_in_row[j_idx])] += 1
        total_rows_with_missing = len(_missing_row_vectors)
        for (f1, f2), count in pair_counts.most_common(15):
            co_missing_pairs.append({
                "feature_a": f1,
                "feature_b": f2,
                "co_missing_count": count,
                "co_missing_rate": round(count / total_rows_with_missing, 4) if total_rows_with_missing else 0.0,
            })

    # Label-missing correlation: check if missing rate differs significantly by label
    label_missing_corr: dict[str, float] = {}
    for col, label_map in final_cond_nulls.items():
        rates = {}
        for label, (null_count, total) in label_map.items():
            rates[label] = null_count / total if total else 0.0
        if len(rates) >= 2:
            sorted_labels = sorted(rates)
            label_missing_corr[col] = round(
                abs(rates[sorted_labels[-1]] - rates[sorted_labels[0]]), 4
            )

    missing_patterns = MissingPatternStats(
        co_missing_pairs=co_missing_pairs,
        label_missing_correlation=label_missing_corr,
    )

    # --- Build SequencePatternStats ---
    seq_patterns_dict: dict[str, dict[str, Any]] = {}
    for domain, repeat_rates in seq_item_repeat_sums.items():
        if repeat_rates:
            arr = np.array(repeat_rates)
            seq_patterns_dict[domain] = {
                "mean_repeat_rate": round(float(arr.mean()), 4),
                "median_repeat_rate": round(float(np.median(arr)), 4),
                "max_repeat_rate": round(float(arr.max()), 4),
                "non_empty_sequences": len(repeat_rates),
            }
    seq_patterns = SequencePatternStats(patterns=seq_patterns_dict)

    return DatasetScanResult(
        groups=groups,
        col_stats=col_accumulators,
        label_dist=label_dist,
        seq_stats=seq_stats,
        row_count=row_count,
        user_stats=user_stats,
        label_cond_stats=label_cond_stats,
        dense_stats=dense_stats,
        missing_patterns=missing_patterns,
        seq_patterns=seq_patterns,
    )


def echarts_user_activity(user_stats: UserStats) -> dict[str, Any]:
    """ECharts option for user activity distribution bar chart."""
    dist = user_stats.activity_distribution()
    bins = dist.get("bins", {})
    return {
        "title": {"text": f"用户活跃度分布 (共 {dist.get('total_users', 0)} 用户)"},
        "tooltip": {"trigger": "axis", "axisPointer": {"type": "shadow"}},
        "grid": {"left": 60, "right": 30, "top": 60, "bottom": 40},
        "xAxis": {"type": "category", "data": list(bins.keys()), "name": "行为次数"},
        "yAxis": {"type": "value", "name": "用户数"},
        "series": [{
            "type": "bar",
            "data": list(bins.values()),
            "itemStyle": {"color": _EC_COLORS[0]},
            "label": {"show": True, "position": "top"},
        }],
    }


def echarts_cross_domain_overlap(user_stats: UserStats) -> dict[str, Any]:
    """ECharts option for cross-domain user overlap heatmap."""
    overlap = user_stats.cross_domain_overlap()
    domains = overlap["domains"]
    matrix = overlap["overlap_matrix"]
    data = []
    for i, d1 in enumerate(domains):
        for j, d2 in enumerate(domains):
            data.append([i, j, matrix[i][j]])
    max_val = max((d[2] for d in data), default=1)
    return {
        "_height": "350px",
        "title": {"text": "跨域用户重叠矩阵"},
        "tooltip": {"formatter": "{c}"},
        "grid": {"left": 80, "right": 40, "top": 60, "bottom": 60},
        "xAxis": {"type": "category", "data": domains, "splitArea": {"show": True}},
        "yAxis": {"type": "category", "data": domains, "splitArea": {"show": True}},
        "visualMap": {
            "min": 0, "max": int(max_val), "show": True,
            "orient": "horizontal", "left": "center", "bottom": 0,
            "inRange": {"color": ["#f9e2af", "#fab387", "#f38ba8"]},
        },
        "series": [{
            "type": "heatmap",
            "data": data,
            "label": {"show": True, "fontSize": 10},
        }],
    }


def echarts_feature_auc(label_cond_stats: LabelConditionalStats, *, top_n: int = 25) -> dict[str, Any]:
    """ECharts option for single-feature AUC ranking."""
    items = sorted(label_cond_stats.feature_auc.items(), key=lambda x: abs(x[1] - 0.5), reverse=True)[:top_n]
    if not items:
        return {"tooltip": {}, "series": []}
    items_rev = list(reversed(items))
    names = [c.replace("user_int_feats_", "u_").replace("item_int_feats_", "i_") for c, _ in items_rev]
    values = [v for _, v in items_rev]
    return {
        "_height": f"{max(300, len(items) * 22)}px",
        "title": {"text": "单特征 AUC 排名 (top features)"},
        "tooltip": {"trigger": "axis", "axisPointer": {"type": "shadow"}},
        "grid": {"left": 90, "right": 40, "top": 50, "bottom": 30},
        "xAxis": {"type": "value", "name": "AUC", "min": max(0.0, min(values) - 0.05), "max": min(1.0, max(0.55, max(values) + 0.05))},
        "yAxis": {"type": "category", "data": names, "axisLabel": {"fontSize": 9}},
        "series": [{
            "type": "bar",
            "data": values,
            "itemStyle": {"color": _EC_COLORS[2]},
            "label": {"show": True, "position": "right", "formatter": "{c}"},
            "markLine": {
                "silent": True,
                "data": [{"xAxis": 0.5, "lineStyle": {"color": _EC_COLORS[4], "type": "dashed"},
                          "label": {"formatter": "随机基线"}}],
            },
        }],
    }


def echarts_null_rate_by_label(label_cond_stats: LabelConditionalStats, *, top_n: int = 20) -> dict[str, Any]:
    """ECharts option for null rate difference between positive/negative labels."""
    diffs = label_cond_stats.null_rate_diff(top_n=top_n)
    if not diffs:
        return {"tooltip": {}, "series": []}
    diffs_rev = list(reversed(diffs))
    names = [d["column"].replace("user_int_feats_", "u_").replace("item_int_feats_", "i_") for d in diffs_rev]
    pos_rates = [round(d["null_rate_positive"], 4) for d in diffs_rev]
    neg_rates = [round(d["null_rate_negative"], 4) for d in diffs_rev]
    return {
        "_height": f"{max(300, len(diffs) * 22)}px",
        "title": {"text": "正负样本缺失率对比 (差异最大的特征)"},
        "tooltip": {"trigger": "axis", "axisPointer": {"type": "shadow"}},
        "legend": {"data": ["正样本 (转化)", "负样本 (非转化)"]},
        "grid": {"left": 100, "right": 40, "top": 60, "bottom": 30},
        "xAxis": {"type": "value", "name": "缺失率", "max": 1.0},
        "yAxis": {"type": "category", "data": names, "axisLabel": {"fontSize": 8}},
        "series": [
            {"name": "正样本 (转化)", "type": "bar", "data": pos_rates, "itemStyle": {"color": _EC_COLORS[2]}},
            {"name": "负样本 (非转化)", "type": "bar", "data": neg_rates, "itemStyle": {"color": _EC_COLORS[1]}},
        ],
    }


def echarts_dense_distributions(dense_stats: DenseFeatureStats) -> dict[str, Any]:
    """ECharts option for dense feature distribution bar chart (mean \u00b1 std)."""
    cols = sorted(dense_stats.distributions.keys())
    if not cols:
        return {"tooltip": {}, "series": []}
    summaries = [dense_stats.distributions[c] for c in cols]
    valid = [(c, s) for c, s in zip(cols, summaries) if s.get("finite_count", 0) > 0]
    if not valid:
        return {"tooltip": {}, "series": []}

    # Bar chart of mean ± std
    names = [c.replace("user_dense_feats_", "d_") for c, _ in valid]
    means = [round(s["mean"], 4) for _, s in valid]
    stds = [round(s.get("std", 0), 4) for _, s in valid]
    return {
        "title": {"text": "稠密特征分布概览 (均值 ± 标准差)"},
        "tooltip": {"trigger": "axis"},
        "grid": {"left": 60, "right": 30, "top": 60, "bottom": 60},
        "xAxis": {"type": "category", "data": names, "axisLabel": {"rotate": 45}},
        "yAxis": {"type": "value", "name": "值"},
        "series": [
            {
                "name": "均值",
                "type": "bar",
                "data": means,
                "itemStyle": {"color": _EC_COLORS[0]},
            },
            {
                "name": "标准差",
                "type": "bar",
                "data": stds,
                "itemStyle": {"color": _EC_COLORS[3], "opacity": 0.6},
            },
        ],
        "legend": {"data": ["均值", "标准差"]},
    }


def echarts_cardinality_bins(bins: dict[str, int]) -> dict[str, Any]:
    """ECharts option for cardinality bin distribution pie chart."""
    data = [{"name": k, "value": v} for k, v in bins.items() if v > 0]
    return {
        "title": {"text": "特征基数区间分布"},
        "tooltip": {"trigger": "item", "formatter": "{b}: {c} 个特征 ({d}%)"},
        "legend": {"bottom": 0},
        "series": [{
            "type": "pie",
            "radius": ["30%", "60%"],
            "avoidLabelOverlap": True,
            "itemStyle": {"borderRadius": 6, "borderWidth": 2},
            "label": {"formatter": "{b}\n{c} ({d}%)"},
            "data": data,
            "color": _EC_COLORS,
        }],
    }


def echarts_seq_repeat_rate(seq_patterns: SequencePatternStats) -> dict[str, Any]:
    """ECharts option for per-domain sequence item repeat rate."""
    domains = sorted(seq_patterns.patterns.keys())
    if not domains:
        return {"tooltip": {}, "series": []}
    means = [seq_patterns.patterns[d].get("mean_repeat_rate", 0) for d in domains]
    return {
        "title": {"text": "序列内物品重复率 (按域)"},
        "tooltip": {"trigger": "axis"},
        "grid": {"left": 60, "right": 30, "top": 60, "bottom": 40},
        "xAxis": {"type": "category", "data": domains},
        "yAxis": {"type": "value", "name": "平均重复率", "max": 1.0},
        "series": [{
            "type": "bar",
            "data": means,
            "itemStyle": {"color": _EC_COLORS[4]},
            "label": {"show": True, "position": "top", "formatter": "{c}"},
        }],
    }


def echarts_co_missing(missing_patterns: MissingPatternStats, *, top_n: int = 10) -> dict[str, Any]:
    """ECharts option for co-missing feature pairs."""
    pairs = missing_patterns.co_missing_pairs[:top_n]
    if not pairs:
        return {"tooltip": {}, "series": []}
    pairs_rev = list(reversed(pairs))
    names = [
        f"{p['feature_a'].replace('user_int_feats_', 'u_').replace('item_int_feats_', 'i_')} × "
        f"{p['feature_b'].replace('user_int_feats_', 'u_').replace('item_int_feats_', 'i_')}"
        for p in pairs_rev
    ]
    rates = [p["co_missing_rate"] for p in pairs_rev]
    return {
        "_height": f"{max(250, len(pairs) * 24)}px",
        "title": {"text": "特征共缺失率 TOP 对"},
        "tooltip": {"trigger": "axis"},
        "grid": {"left": 150, "right": 40, "top": 50, "bottom": 30},
        "xAxis": {"type": "value", "name": "共缺失率", "max": 1.0},
        "yAxis": {"type": "category", "data": names, "axisLabel": {"fontSize": 8}},
        "series": [{
            "type": "bar",
            "data": rates,
            "itemStyle": {"color": _EC_COLORS[5]},
            "label": {"show": True, "position": "right", "formatter": "{c}"},
        }],
    }


def serialize_echarts(option: dict[str, Any]) -> str:
    """Serialize ECharts option dict to JSON string."""
    return json.dumps(option, indent=2, ensure_ascii=False)
