---
name: taac-docs-workflow
description: Enforce the docs build pipeline for this TAAC 2026 codebase. Use when any agent edits files under src/taac2026/reporting/, docs/analysis/, docs/assets/, the EDA CLI, ECharts generators, or any code that produces gitignored artifacts consumed by the documentation site. Ensures generated assets are regenerated and the site is rebuilt before previewing.
---

# Taac Docs Workflow

## Overview

Use this skill whenever a change touches the documentation pipeline — EDA analysis code, ECharts chart generators, reporting CLIs, doc Markdown pages, or JS/CSS assets.
The docs site depends on **generated artifacts that are not committed to version control**.
Forgetting to regenerate them after code changes causes broken charts, missing pages, or silent 404s that serve HTML instead of JSON.

## The Pipeline

The documentation build has three mandatory stages that must run **in order**:

```
1. Generate artifacts   →   2. Build site   →   3. Preview
```

### Stage 1 — Generate artifacts

ECharts JSON files under `docs/assets/figures/eda/` are **gitignored**.
They are produced by the EDA CLI and must be regenerated whenever:

- Any function in `src/taac2026/reporting/dataset_eda.py` is added or changed
- The EDA CLI (`src/taac2026/application/reporting/eda_cli.py`) is modified
- New `<div class="echarts" data-src="...">` references are added to Markdown
- Chart naming, output paths, or serialization logic changes

Command:

```bash
uv run taac-dataset-eda
```

This writes all `*.echarts.json` files to `docs/assets/figures/eda/`.

### Stage 2 — Build site

The static site generator copies `docs/` (including `docs/assets/figures/eda/`) into `site/`.
If Stage 1 was skipped, the build will succeed but the site will be missing chart data.

Command:

```bash
uv run --no-project --isolated --with zensical zensical build --clean
```

### Stage 3 — Preview

```bash
uv run --no-project --isolated --with zensical zensical serve
```

The dev server serves from `site/`.
**Do not skip Stage 2** — `zensical serve` may hot-reload Markdown changes but will not pick up new or changed assets without a rebuild.

## Rules

### Always regenerate after editing reporting code

When you edit any of these files, you **must** run `uv run taac-dataset-eda` before building docs:

| File | Impact |
|------|--------|
| `src/taac2026/reporting/dataset_eda.py` | All ECharts JSON files |
| `src/taac2026/application/reporting/eda_cli.py` | Which charts get written and their filenames |
| `src/taac2026/domain/metrics.py` | Metric computations used in EDA charts |
| `docs/analysis/dataset-eda.md` | May reference new `data-src` chart files |

### Always rebuild site after generating artifacts

After `uv run taac-dataset-eda`, run `zensical build --clean` so the fresh JSON files are copied into `site/`.

### Verify chart file existence

Before marking a docs task complete, verify that all `data-src` references in Markdown have corresponding JSON files:

```powershell
# List all referenced chart files in docs
Select-String -Path docs/analysis/*.md -Pattern 'data-src="assets/figures/eda/([^"]+)"' -AllMatches |
  ForEach-Object { $_.Matches.Groups[1].Value } | Sort-Object -Unique

# List all generated chart files
Get-ChildItem docs/assets/figures/eda/*.echarts.json -Name | Sort-Object
```

Every referenced file must exist.
A missing file will silently serve a 404 HTML page, causing `ECharts load error: Unexpected token '<'` in the browser.

### New chart checklist

When adding a new ECharts chart:

1. Add the generator function in `dataset_eda.py` (e.g. `echarts_my_chart()`)
2. Add the `_write_ec("my_chart", echarts_my_chart(...))` call in `eda_cli.py`
3. Add `<div class="echarts" data-src="assets/figures/eda/my_chart.echarts.json"></div>` in the Markdown page
4. Add a smoke test in `tests/test_dataset_eda.py`
5. Run `uv run taac-dataset-eda` to generate the JSON
6. Run `zensical build --clean` to include it in the site
7. Verify in browser that the chart renders without errors

### Common failure mode

**Symptom**: `ECharts load error: Unexpected token '<', "<!doctype"... is not valid JSON`

**Cause**: The browser fetched a URL that returned an HTML 404 page instead of a JSON file.
This means the `.echarts.json` file does not exist in `site/assets/figures/eda/`.

**Fix**: Run Stage 1 + Stage 2 (generate + rebuild).

## Gitignored artifacts

These paths are gitignored and must be regenerated, never committed:

```
docs/assets/figures/eda/*.echarts.json
site/
```

## Reference

Read `references/docs-pipeline.md` for the full asset inventory and CI integration details.
