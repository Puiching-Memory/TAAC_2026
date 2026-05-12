---
name: taac-platform-api-inspection
description: "Use when: inspecting TAAC/Taiji platform pages on taiji.algo.qq.com, especially model training instances, checkpoint pages, metrics plots, TensorBoard-like scalars, browser MCP sessions, authenticated cookies, or direct platform API calls. Prefer API data over screenshots."
---

# TAAC Platform API Inspection

Use when a user asks to inspect TAAC/Taiji training pages or analyze metrics.
Get structured data from the authenticated browser session. Never estimate
values from screenshots.

## Core Principles

- **Always use `page.evaluate()` + `fetch({ credentials: "include" })` for API
  calls.** `page.context().request.get()` fails with `Storage.getCookies:
  Method not found` in the MCP browser environment.
- **Always summarize inside `page.evaluate()`.** Raw `tf_events` JSON is
  50–200 KB and will be truncated into ephemeral resource files. Return only
  compact summaries — one object per metric.
- **Response interception + `page.reload()` is the fastest discovery method.**
  One reload captures both `get_ckpt` and `tf_events` endpoints.

## Two-Step Pipeline

### Step 1: Navigate and discover endpoints

Navigate to the checkpoint URL, then intercept XHR responses during reload:

```javascript
// run_playwright_code — intercept + reload
const captured = [];
const keywords = ["api", "training", "ckpt", "metric", "scalar", "plot", "instance", "tf_events"];
const handler = async (response) => {
  const url = response.url();
  if (!keywords.some(kw => url.toLowerCase().includes(kw))) return;
  const ct = response.headers()["content-type"] || "";
  const item = { url, status: response.status(), contentType: ct };
  if (ct.includes("json")) {
    try { item.preview = (await response.text()).slice(0, 4000); } catch (e) { item.error = String(e); }
  }
  captured.push(item);
};
page.on("response", handler);
await page.reload({ waitUntil: "networkidle" });
page.off("response", handler);
return captured;
```

This surfaces:
- `.../external/<instance_id>/get_ckpt` — checkpoint list, sizes, publish status
- `.../external/<instance_id>/tf_events` — all scalar metrics

Extract `<instance_id>` from the URL: it's the last path segment
(`95d1b0469e0fba1e019e1811651b16d5` in a typical checkpoint URL).

If the browser redirects to `https://algo.qq.com/?type=login`, stop and ask the
user to log in first.

### Step 2: Fetch and summarize metrics in-browser

```javascript
// run_playwright_code — fetch + summarize
return page.evaluate(async () => {
  const url = 'https://taiji.algo.qq.com/taskmanagement/api/v1/instances/external/<INSTANCE_ID>/tf_events';
  const json = await (await fetch(url, { credentials: 'include' })).json();
  const groups = json.data?.data ?? {};
  const summaries = [];

  for (const [group, charts] of Object.entries(groups)) {
    if (!Array.isArray(charts)) continue;
    for (const chart of charts) {
      const dates = chart.date || [];
      const titles = chart.title || [];
      const values = chart.value || [];
      for (let i = 0; i < titles.length; i++) {
        const name = titles[i];
        const vals = values[i] || [];
        const points = dates
          .map((step, j) => ({ step: +step, value: +vals[j] }))
          .filter(p => Number.isFinite(p.step) && Number.isFinite(p.value));
        if (!points.length) continue;

        let min = points[0], max = points[0];
        for (const p of points) {
          if (p.value < min.value) min = p;
          if (p.value > max.value) max = p;
        }
        const n = points.length;
        const earlyN = Math.max(1, Math.floor(n * 0.2));
        const lateN = Math.max(1, Math.floor(n * 0.2));
        const earlyAvg = points.slice(0, earlyN).reduce((s, p) => s + p.value, 0) / earlyN;
        const lateAvg = points.slice(n - lateN).reduce((s, p) => s + p.value, 0) / lateN;
        const trend = lateAvg > earlyAvg * 1.005 ? '↑' : lateAvg < earlyAvg * 0.995 ? '↓' : '→';

        summaries.push({
          group, name, count: n,
          first: [points[0].step, +points[0].value.toFixed(6)],
          last:  [points[n-1].step, +points[n-1].value.toFixed(6)],
          min:   [min.step, +min.value.toFixed(6)],
          max:   [max.step, +max.value.toFixed(6)],
          trend
        });
      }
    }
  }
  return summaries;
});
```

Replace `<INSTANCE_ID>` with the value from Step 1. The result is a compact
array, e.g. `{group:"AUC", name:"AUC/valid", first:[500,0.7929],
last:[5000,0.8296], min:[500,0.7929], max:[2500,0.8346], trend:"↑"}`.

## Analysis Framework

When reporting results, structure the analysis around these signals:

### Primary signals (always check)

| Signal | What to look for |
|--------|-----------------|
| **Overfitting** | Train Loss ↓ while valid AUC peaks then ↓ or LogLoss ↑ |
| **Best checkpoint** | The step where valid AUC is maximized (not the final step) |
| **Convergence** | Is valid AUC still rising at the end, or plateaued? |
| **Score separation** | `score_margin_mean` trend; positive/negative score divergence |

### Symbiosis-specific signals (when present)

| Signal | What to look for |
|--------|-----------------|
| **Token collapse** | `effective_rank` ~1, `mean_pairwise_cosine` ~1, `max_pairwise_cosine` ~1 |
| **Token health** | `latent_valid_ratio` across seq_a/b/c/d — are tokens meaningful? |
| **Diversity trend** | `effective_rank` ↑, `mean_pairwise_cosine` ↓ = good; opposite = bad |
| **Span usage** | `active_token_ratio` for candidate/cross/context/item/sequence spans |

### Probe metrics (when present)

Compare `Probe/auc`, `Probe/logloss`, `Probe/auc_retention` against
`AUC/valid` and `LogLoss/valid`. Probe degradation while full metrics improve
indicates fragile representations.

## Pitfalls

- **Don't return raw JSON.** If you see "Large tool result written to file",
  you've already failed — re-run with in-browser summarization.
- **`json.data?.data` not `json.data`.** The `tf_events` endpoint nests metric
  groups one level deeper than expected.
- **Guard with `Array.isArray(charts)`.** Not all entries under `json.data?.data`
  are arrays.
- **Console noise is normal.** WeChat login probes (`localhost.weixin.qq.com`,
  `ERR_CONNECTION_CLOSED`), preload warnings, and telemetry errors are page
  noise, not training failures.
- **Don't click Publish/Delete/Cancel/Submit** unless the user explicitly asks.

## Safety

- Never expose cookies, tokens, or authorization headers.
- Prefer browser-context requests over terminal `curl`.
- Don't commit downloaded platform payloads or screenshots.