---
name: taac-platform-api-inspection
description: "Use when: inspecting TAAC/Taiji platform pages on taiji.algo.qq.com, especially model training instances, checkpoint pages, metrics plots, TensorBoard-like scalars, browser MCP sessions, authenticated cookies, or direct platform API calls. Prefer API data over screenshots."
---

# TAAC Platform API Inspection

Use this skill when a user asks to inspect TAAC/Taiji platform training pages,
checkpoint pages, output details, metrics, plots, or model evaluation pages. The
goal is to get structured data from the authenticated browser session instead of
estimating values from screenshots.

## Core Rule

Prefer authenticated API calls through the shared browser context over visual
inspection.

Screenshots are useful for layout confirmation, but chart values should come
from API JSON, page state, or DOM text whenever possible. Do not infer precise
metric values from pixels when the logged-in browser can call the platform API.

## Tool Setup

Before using deferred browser tools, load them with `tool_search`. The tools most
often needed are:

- `read_page`: confirm the current URL, login state, visible tables, and page refs.
- `run_playwright_code`: inspect network requests and call authenticated APIs.
- `screenshot_page`: fallback only, or to confirm the page being analyzed.
- `click_element`: navigate tabs only when direct API discovery is not enough.

If the browser lands on `https://algo.qq.com/?type=login` or shows a public TAAC
landing page, ask the user to finish login in the shared browser. Do not request
or handle passwords, QR codes, tokens, or raw cookies in chat.

## Workflow

1. Confirm the page and login state with `read_page`.
   - Record the actual URL.
   - Extract job id, instance id, project/team id, checkpoint id, and visible
     checkpoint names from the page or URL.
   - Verify whether the page is a training detail page, checkpoint page, or a
     login/landing fallback.

2. Discover API endpoints from the browser context.
   - Use `performance.getEntriesByType("resource")` to list loaded XHR/fetch
     resources.
   - Filter for likely API URLs containing terms such as `api`, `training`,
     `ckpt`, `checkpoint`, `metric`, `scalar`, `plot`, `tensorboard`, `job`,
     `instance`, `output`, or the current instance id.
   - If resource entries are stale or incomplete, attach a temporary
     `page.on("response")` listener and reload the current page.

3. Call APIs with the browser context's authenticated cookie jar.
   - Prefer `page.context().request.get(url)` for same-origin JSON endpoints.
   - `page.evaluate(() => fetch(url, { credentials: "include" }))` is also
     acceptable for same-origin APIs.
   - Do not print cookie values. Do not copy cookies into terminal commands
     unless there is no browser-context alternative.

4. Parse structured data.
   - For checkpoint tables, use JSON fields or DOM table text, not screenshots.
   - For metrics, extract scalar series as `{name, step, value}` arrays or the
     closest equivalent returned by the platform.
   - Summarize exact best steps, last steps, minima/maxima, and trend direction.
   - For `Probe/*` metrics, compare `Probe/auc`, `Probe/logloss`, and
     `Probe/auc_retention` against full `AUC/valid` and `LogLoss/valid`.

5. Use screenshots only as a fallback.
   - Take screenshots when API access fails, the chart library hides data, or the
     user explicitly asks for visual confirmation.
   - Label screenshot-derived values as approximate.

## Useful Playwright Snippets

Discover loaded API-like resources:

```javascript
return page.evaluate(() => {
  const keywords = [
    "api",
    "training",
    "ckpt",
    "checkpoint",
    "metric",
    "scalar",
    "plot",
    "tensorboard",
    "job",
    "instance",
    "output",
  ];
  return performance
    .getEntriesByType("resource")
    .map((entry) => ({ name: entry.name, type: entry.initiatorType }))
    .filter((entry) => keywords.some((keyword) => entry.name.toLowerCase().includes(keyword)))
    .slice(-100);
});
```

Capture JSON responses during reload:

```javascript
const captured = [];
const keywords = ["api", "training", "ckpt", "metric", "scalar", "plot", "instance"];
const handler = async (response) => {
  const url = response.url();
  if (!keywords.some((keyword) => url.toLowerCase().includes(keyword))) return;
  const contentType = response.headers()["content-type"] || "";
  const item = { url, status: response.status(), contentType };
  if (contentType.includes("json")) {
    try {
      const text = await response.text();
      item.preview = text.slice(0, 4000);
    } catch (error) {
      item.error = String(error);
    }
  }
  captured.push(item);
};
page.on("response", handler);
await page.reload({ waitUntil: "networkidle" });
page.off("response", handler);
return captured;
```

Call a discovered JSON endpoint with browser cookies:

```javascript
const response = await page.context().request.get(endpointUrl, {
  headers: { accept: "application/json,text/plain,*/*" },
});
const text = await response.text();
return {
  url: endpointUrl,
  status: response.status(),
  ok: response.ok(),
  contentType: response.headers()["content-type"] || "",
  bodyPreview: text.slice(0, 12000),
};
```

Extract visible table text without screenshots:

```javascript
return page.locator("table").evaluateAll((tables) =>
  tables.map((table) => table.innerText)
);
```

## Safety And Hygiene

- Never expose raw cookies, authorization headers, CSRF tokens, QR codes, or
  session identifiers in chat output.
- Do not click destructive or state-changing buttons such as `Publish`,
  `Delete`, `Cancel`, or `Submit` unless the user explicitly asks.
- Prefer browser-context requests over terminal `curl`; the browser context keeps
  cookies in memory and avoids leaking them into shell history or logs.
- If a terminal API call is unavoidable, use a temporary file outside the repo,
  avoid echoing secrets, and delete the temporary file before finishing.
- Do not commit downloaded platform payloads, screenshots, cookies, or temporary
  API captures.

## Reporting Pattern

When reporting training analysis, include:

- checkpoint selected and publish status;
- exact or structured metric values when available;
- trend comparison between full metrics and hard probe metrics;
- whether values came from API JSON, DOM text, or screenshot fallback;
- any access limitation, such as login redirect or 401.

For TAAC training pages, a strong summary usually compares:

- `AUC/valid` vs `Probe/auc`;
- `LogLoss/valid` vs `Probe/logloss`;
- `Probe/auc_retention` trend;
- score distribution diagnostics such as `score_std`, positive/negative means,
  and sample counts.
