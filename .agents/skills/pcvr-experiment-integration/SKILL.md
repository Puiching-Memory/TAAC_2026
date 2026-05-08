---
name: pcvr-experiment-integration
description: Use when changing TAAC PCVR experiment packages or shared PCVR runtime code under experiments/, src/taac2026/domain, src/taac2026/application, or src/taac2026/infrastructure, including model contracts, experiment discovery, packaging, training/evaluation/inference hooks, and accelerator-backed modeling behavior.
---

# PCVR Experiment Integration

Use this skill as a reading map, not as a replacement for source or docs. Prefer the live code and tests as the contract.

## Read First

For experiment package changes:

- `experiments/<name>/__init__.py`
- `experiments/<name>/model.py`
- `src/taac2026/api.py`
- `src/taac2026/application/experiments/{experiment,factory,discovery,registry,runtime}.py`
- `src/taac2026/domain/{config,experiment,model_contract,sidecar}.py`
- `tests/unit/experiments/test_packages.py`
- `tests/unit/experiments/test_runtime_contract_matrix.py`

For shared runtime or model-input changes:

- `src/taac2026/infrastructure/modeling/`
- `src/taac2026/infrastructure/runtime/`
- `src/taac2026/application/{training,evaluation}/`
- `tests/unit/domain/test_model_contract.py`
- focused tests under `tests/unit/application/` and `tests/unit/infrastructure/`

For accelerator-backed model behavior:

- `src/taac2026/infrastructure/accelerators/`
- `src/taac2026/infrastructure/modeling/`
- `tests/unit/infrastructure/accelerators/`

For bundle behavior, also use `$taac-competition-environment`.

## Non-Obvious Context

- Experiment packages should stay thin: package-local model code and truly experiment-specific helpers only. Shared training/evaluation/inference/packaging behavior belongs under `src/taac2026`.
- Checkpoint sidecars are a runtime contract. When model construction, schema conversion, or config defaults change, inspect the training-to-evaluation/inference path rather than only the training path.
- Do not infer current contracts from archived official snapshots. Treat `docs/archive/files/...` as historical reference and verify against active source/tests.
- Archived schema fixtures are stored at `docs/archive/files/schema/`. They are useful for explicit local `--schema-path` smoke runs when repository sample data is absent, but they remain reference fixtures rather than the live model/input contract.
- When refactoring shared infrastructure, assume generated online bundles import the packaged `src/taac2026` tree without repo-only helpers.

## Validation

Run the smallest tests that cover the touched contract first:

```bash
uv run pytest tests/unit/experiments/test_packages.py -q
uv run pytest tests/unit/experiments/test_runtime_contract_matrix.py -q
uv run pytest tests/unit/domain/test_model_contract.py -q
uv run pytest tests/unit/application/experiments -q
```

For shared runtime, packaging, or accelerator changes, add the focused `tests/unit/application/...` or `tests/unit/infrastructure/...` files that import the touched modules. Before finishing broad changes, prefer:

```bash
uv run pytest tests/unit -q
uv run --with ruff ruff check .
```
