# TAAC 2026 2.0.0 Refactor TODO

Last updated: 2026-05-04

## Working Rules

- Keep this file aligned with the implementation after each refactor slice.
- Re-check the relevant docs before changing behavior, especially `docs/architecture.md`, `docs/guide/contributing.md`, and `docs/guide/online-training-bundle.md`.
- Prefer compatibility-preserving changes first; add migration or validation before removing old behavior.
- Keep PCVR experiment packages model-focused. Shared training, evaluation, inference, bundle, and platform logic should stay in `src/taac2026`.

## Current Documentation Baseline

- `docs/architecture.md` defines `experiments/` as the plugin layer and `src/taac2026` as the framework layer.
- `docs/guide/online-training-bundle.md` says training bundles contain `run.sh` plus `code_package.zip`, and inference bundles contain `infer.py` plus `code_package.zip`.
- `docs/guide/online-training-bundle.md` says bundle manifests live under `project/.taac_*_manifest.json` and should point at `experiments/<group>/<experiment>`.
- `docs/guide/online-training-bundle.md` now documents versioned bundle manifest fields: `manifest_version`, `bundle_kind`, `bundle_format_version`, `framework.version`, `runtime_env`, and `compatibility`.
- `docs/architecture.md` now documents versioned PCVR `train_config.json` sidecars while preserving legacy flat-file compatibility.
- Boundary contracts now use Pydantic 2.12.5 internally for validation while continuing to expose plain JSON / dict payloads.
- Runtime platform behavior now lives in `src/taac2026/infrastructure/platform`; `run.sh` and generated `infer.py` are thin bootstrap entrypoints.
- Shared PCVR model primitives now live in `taac2026.infrastructure.pcvr.modeling`; package-local `layers.py` files are compatibility re-exports.
- `docs/guide/contributing.md` now teaches new PCVR packages to use `create_pcvr_experiment()` and reserve hook overrides for real runtime differences.

## P0 - Foundation

- [x] Fix stale experiment path wiring in Docker/dev entrypoints so they use `experiments/pcvr/...` instead of the removed `config/...` layout.
- [x] Add a small PCVR experiment factory that supplies default train, prediction, and runtime hooks.
- [x] Migrate standard PCVR experiments to the factory without changing model behavior.
- [x] Keep Symbiosis on explicit overrides, but express those overrides through the same factory.
- [x] Update docs to make the factory the preferred experiment package contract.
- [x] Run focused tests for experiment discovery, experiment packages, bundle packaging, and run.sh command behavior.

## P1 - Versioned Contracts

- [x] Add explicit version fields to train config sidecars while continuing to read existing flat `train_config.json` files.
- [x] Add a bundle manifest object with format version, framework version, experiment path, entrypoint, runtime variables, and compatibility metadata.
- [x] Add a bundle verifier used by both training and inference packagers.
- [x] Add tests for old manifest/config compatibility and new manifest/config validation.

## P1.5 - Pydantic Contract Models

- [x] Pin `pydantic==2.12.5` as a runtime dependency so online bundles can import contract validators without dev extras.
- [x] Convert PCVR `train_config.json` sidecar helpers to Pydantic-backed validation while preserving flat payload output and legacy reads.
- [x] Convert bundle manifest models and validation to Pydantic-backed objects while preserving manifest JSON shape.
- [x] Keep Pydantic scoped to boundary contracts rather than model/training hot paths.

## P2 - Runtime Platform Layer

- [x] Move platform environment parsing out of `run.sh` and generated `infer.py` into Python runtime context objects.
- [x] Keep `run.sh` as a thin shell bootstrapper for local and training-bundle execution.
- [x] Add platform adapters for local uv, online training bundle, online inference bundle, and Docker GPU execution.
- [x] Expand CI-covered unit coverage with tiny bundle round-trip smoke tests for training and inference entrypoints.

## P3 - Shared Model Primitives

- [x] Centralize duplicated PCVR tokenizers, embedding banks, RMSNorm configuration, and parameter grouping mixins.
- [x] Migrate non-baseline experiments to shared primitives.
- [x] Keep paper-specific architecture blocks inside each experiment package.
- [x] Reconcile `docs/guide/contributing.md` imports with the actual shared modeling API.

## Progress Log

- 2026-05-04: Created this TODO after checking architecture, contributing, online bundle docs, and Docker path wiring.
- 2026-05-04: Updated `docker-compose.yml` train/evaluate services to pass `TAAC_EXPERIMENT=experiments/pcvr/<name>` and `TRAIN_DATA_PATH`, matching `docs/architecture.md`, `docs/guide/online-training-bundle.md`, and `run.sh` behavior.
- 2026-05-04: Added `create_pcvr_experiment` and migrated all PCVR experiment packages to use it. Standard packages now rely on default hooks from the factory; Symbiosis declares only build/config overrides.
- 2026-05-04: Updated architecture and contributing docs so new PCVR packages use `create_pcvr_experiment` instead of copying full hook objects.
- 2026-05-04: Validation passed with focused Ruff, experiment discovery/package tests, bundle packaging tests, run.sh command tests, and the PCVR runtime contract matrix.
- 2026-05-04: Added versioned PCVR train_config sidecars with legacy flat-file reads, introduced a shared bundle manifest object/verifier for training and inference packagers, updated docs, and validated P1 with focused Ruff plus 201 unit tests.
- 2026-05-04: Synced the PCVR agent skill with the factory-first experiment contract and validated docs with `uv run zensical build --strict`.
- 2026-05-04: Pinned Pydantic 2.12.5 and moved PCVR train_config sidecar plus bundle manifest validation onto Pydantic-backed contract models without changing emitted JSON shape.
- 2026-05-04: Added the shared platform runtime layer, thinned `run.sh` and generated `infer.py`, centralized PCVR model primitives, migrated non-baseline models to shared imports, and covered both changes with focused unit tests.
