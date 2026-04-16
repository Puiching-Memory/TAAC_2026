# Docs Pipeline Reference

## Generated Asset Inventory

| Source | Generator Command | Output Path | Gitignored |
|--------|------------------|-------------|------------|
| EDA ECharts JSON | `uv run taac-dataset-eda` | `docs/assets/figures/eda/*.echarts.json` | Yes |
| Model performance plots | `uv run taac-plot-model-performance` | `figures/*.png`, `figures/*.svg` | No |
| Static site | `uv run --no-project --isolated --with zensical zensical build --clean` | `site/` | Yes |

## Current ECharts Chart Files (18 total)

### Base charts (10)

- `label_distribution.echarts.json` — Label-type pie chart
- `null_rates.echarts.json` — Top-30 highest null rate features
- `cardinality.echarts.json` — Sparse feature cardinality bar chart
- `sequence_lengths.echarts.json` — Per-domain sequence length line chart
- `coverage_heatmap.echarts.json` — Sparse feature coverage heatmap
- `column_layout.echarts.json` — Column group donut chart
- `ndcg_decay.echarts.json` — NDCG@K discount curve
- `label_cross_edition.echarts.json` — Cross-edition label distribution
- `edition_comparison.echarts.json` — Dataset dimension comparison
- `seq_length_summary.echarts.json` — Per-domain radar chart

### Deep analysis charts (8)

- `user_activity.echarts.json` — User activity distribution
- `cross_domain_overlap.echarts.json` — Cross-domain user overlap heatmap
- `feature_auc.echarts.json` — Single-feature AUC ranking
- `null_rate_by_label.echarts.json` — Positive/negative null rate comparison
- `dense_distributions.echarts.json` — Dense feature mean ± std
- `cardinality_bins.echarts.json` — Cardinality bin distribution pie
- `seq_repeat_rate.echarts.json` — Sequence item repeat rate per domain
- `co_missing.echarts.json` — Co-missing feature pairs

## CI Integration

The GitHub Actions deploy-docs workflow runs both stages:

```yaml
- run: uv run taac-dataset-eda          # Stage 1: generate
- run: uv run --no-project --isolated --with zensical zensical build --clean  # Stage 2: build
```

See `.github/workflows/deploy-docs.yml` for the full pipeline.

## Markdown Chart Embedding Convention

```html
<div class="echarts" data-src="assets/figures/eda/FILENAME.echarts.json"></div>
```

- `data-src` paths are relative to the **site root**, resolved by `echarts-fence.js` via `__md_scope`
- The JS fetches `getBase() + "/" + data-src`, where `getBase()` is the site root URL
- If the JSON file is missing, the browser receives a 404 HTML page → JSON parse error
