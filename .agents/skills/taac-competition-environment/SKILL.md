---
name: taac-competition-environment
description: Use when working on TAAC local setup, uv/CUDA dependency flow, run.sh behavior, online training or inference bundles, package contents, platform Python execution, or environment/debug docs.
---

# TAAC Competition Environment

Use this skill to find the live environment contract quickly. Do not copy behavior from this file when the source has changed.

## Read First

For local vs online runner behavior:

- `run.sh`
- `src/taac2026/application/bootstrap/run_sh.py`
- `src/taac2026/infrastructure/platform/{deps,env,imports}.py`
- `docs/guide/competition-online-server.md`
- `docs/guide/online-training-bundle.md`
- `docs/getting-started.md`

For package layout:

- `src/taac2026/application/packaging/{cli,service}.py`
- `src/taac2026/infrastructure/bundles/`
- `src/taac2026/application/bootstrap/inference_bundle*.py`
- `tests/unit/application/packaging/`
- `tests/unit/application/bootstrap/`

For dependencies:

- `pyproject.toml`
- `uv.lock`
- docs that mention setup or bundle execution

## Non-Obvious Context

- Local repository commands are expected to use `uv`; generated online bundles must not require `uv`.
- Bundle validation must inspect what gets zipped, not just whether local imports work from the repository root.
- The online platform may provide its own Python/CUDA stack. Avoid changes that make bundle mode depend on dev extras, repo-local files, or `uv.lock`.
- `docs/archive/files/...` are historical competition references, not active runtime code.

## Validation

For environment/bootstrap changes:

```bash
uv run pytest tests/unit/application/bootstrap -q
uv run pytest tests/unit/application/packaging -q
```

For bundle-impacting changes, build and inspect at least one tiny bundle:

```bash
uv run taac-package-train --experiment experiments/baseline --output-dir /tmp/taac-training-bundle --json
uv run taac-package-infer --experiment experiments/baseline --output-dir /tmp/taac-inference-bundle --json
python -m zipfile -l /tmp/taac-training-bundle/code_package.zip | sed -n '1,80p'
```

For docs-only environment edits, validate docs with the docs skill instead of running the full unit suite by default.
