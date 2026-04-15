from __future__ import annotations

from tests.support import build_row

from taac2026.reporting.dataset_eda import (
    _serialize_echarts,
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


def _sample_rows() -> list[dict]:
    base_ts = 1_770_000_000
    return [
        build_row(0, base_ts + 100, True, "u1", 101),
        build_row(1, base_ts + 200, False, "u1", 102),
        build_row(2, base_ts + 300, True, "u2", 103),
        build_row(3, base_ts + 400, False, "u3", 101),
        build_row(4, base_ts + 500, True, "u2", 104),
    ]


class TestClassifyColumns:
    def test_groups_all_columns(self) -> None:
        rows = _sample_rows()
        groups = classify_columns(list(rows[0].keys()))
        assert groups.total == len(rows[0].keys())
        assert "user_id" in groups.scalar
        assert len(groups.user_int) >= 1
        assert len(groups.item_int) >= 1
        assert len(groups.domain_seq) == 4


class TestColumnStats:
    def test_counts_and_nulls(self) -> None:
        rows = _sample_rows()
        stats = compute_column_stats(iter(rows))
        assert stats["user_id"].count == 5
        assert stats["user_id"].non_null == 5
        assert stats["user_id"].null_rate == 0.0

    def test_list_column_lengths(self) -> None:
        rows = _sample_rows()
        stats = compute_column_stats(iter(rows), columns=["domain_a_seq_11"])
        st = stats["domain_a_seq_11"]
        assert st.is_list
        assert st.mean_list_len is not None
        assert st.mean_list_len == 3.0  # each sample row has 3 events


class TestLabelDistribution:
    def test_counts_match(self) -> None:
        rows = _sample_rows()
        dist = compute_label_distribution(iter(rows))
        table = dist.as_table()
        total_from_table = sum(r["count"] for r in table)
        assert total_from_table == len(rows)


class TestSequenceLengths:
    def test_all_domains_present(self) -> None:
        rows = _sample_rows()
        seq_stats = compute_sequence_lengths(iter(rows))
        assert set(seq_stats.keys()) == {"domain_a", "domain_b", "domain_c", "domain_d"}
        for domain, st in seq_stats.items():
            assert st.count == len(rows)


class TestCardinalityRanking:
    def test_ranked_descending(self) -> None:
        rows = _sample_rows()
        groups = classify_columns(list(rows[0].keys()))
        stats = compute_column_stats(iter(rows))
        ranking = compute_cardinality_ranking(stats, groups)
        values = [r["n_unique"] for r in ranking]
        assert values == sorted(values, reverse=True)


class TestECharts:
    """Smoke-test that ECharts generators return serializable dicts."""

    def test_echarts_label_distribution(self) -> None:
        dist = compute_label_distribution(iter(_sample_rows()))
        opt = echarts_label_distribution(dist)
        assert "series" in opt
        j = _serialize_echarts(opt)
        assert '"type": "pie"' in j

    def test_echarts_cardinality(self) -> None:
        rows = _sample_rows()
        groups = classify_columns(list(rows[0].keys()))
        stats = compute_column_stats(iter(rows))
        ranking = compute_cardinality_ranking(stats, groups)
        opt = echarts_cardinality(ranking)
        assert "series" in opt
        _serialize_echarts(opt)

    def test_echarts_sequence_lengths(self) -> None:
        seq_stats = compute_sequence_lengths(iter(_sample_rows()))
        opt = echarts_sequence_lengths(seq_stats)
        assert "series" in opt
        _serialize_echarts(opt)

    def test_echarts_coverage_heatmap(self) -> None:
        rows = _sample_rows()
        groups = classify_columns(list(rows[0].keys()))
        stats = compute_column_stats(iter(rows))
        opt = echarts_coverage_heatmap(stats, groups)
        assert "series" in opt
        _serialize_echarts(opt)

    def test_echarts_ndcg_decay(self) -> None:
        opt = echarts_ndcg_decay()
        assert "series" in opt
        j = _serialize_echarts(opt)
        assert '"type": "line"' in j

    def test_echarts_cross_edition(self) -> None:
        opt = echarts_cross_edition()
        assert "series" in opt
        _serialize_echarts(opt)

    def test_echarts_column_layout(self) -> None:
        rows = _sample_rows()
        groups = classify_columns(list(rows[0].keys()))
        opt = echarts_column_layout(groups)
        assert "series" in opt
        j = _serialize_echarts(opt)
        assert '"type": "pie"' in j

    def test_echarts_null_rates(self) -> None:
        stats = compute_column_stats(iter(_sample_rows()))
        opt = echarts_null_rates(stats)
        assert "series" in opt
        _serialize_echarts(opt)

    def test_echarts_edition_comparison(self) -> None:
        opt = echarts_edition_comparison()
        assert "series" in opt
        j = _serialize_echarts(opt)
        assert "TAAC 2025" in j

    def test_echarts_seq_length_summary(self) -> None:
        seq_stats = compute_sequence_lengths(iter(_sample_rows()))
        opt = echarts_seq_length_summary(seq_stats)
        assert "series" in opt
        _serialize_echarts(opt)
