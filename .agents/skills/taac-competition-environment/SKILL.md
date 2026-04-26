---
name: taac-competition-environment
description: 'Use when setting up, documenting, debugging, or reviewing the TAAC 2026 competition environment flow: local uv development, CUDA/CPU dependency profiles, run.sh commands, two-file online training bundles, code_package.zip, and online Conda+pip/Python execution without uv.'
argument-hint: 'local setup, package, online run, or env debug'
user-invocable: true
---

# TAAC Competition Environment

## When to Use

- Set up or debug the local TAAC development environment.
- Decide whether a command should run through `uv` or plain Python.
- Build, inspect, or explain online training bundles.
- Prepare online platform runs where the platform executes only `run.sh`.
- Review docs or scripts that mention package shape, dependency installation, Conda, pip, CUDA profiles, or competition workflow.

## Environment Model

This repository deliberately uses two different environment modes:

- Local repository mode uses `uv` for dependency resolution, lockfile fidelity, and command execution.
- Online bundle mode uses the platform's already activated Python or Conda environment and runs with plain `python`; do not require `uv` online.

The same top-level `run.sh` supports both modes:

- If `code_package.zip` exists beside `run.sh`, bundle mode is enabled and the default runner is `python`.
- If running from the repository root without `code_package.zip`, local mode is enabled and the default runner is `uv`.
- `TAAC_RUNNER=python|uv` can override the default when debugging.

## Local Development With uv

Use `uv` locally because `pyproject.toml` and `uv.lock` are the source of truth for development dependencies.

Recommended bootstrap:

```bash
git lfs install
git lfs pull
uv python install 3.10.20
uv sync --locked --extra cuda126
```

For local training or validation, use the only CUDA profile currently supported by the project:

```bash
uv sync --locked --extra cuda126
```

Use the top-level entrypoint instead of calling console scripts directly:

```bash
bash run.sh train --experiment config/baseline \
    --dataset-path /path/to/parquet_or_dir \
    --schema-path /path/to/schema.json

uv run pytest tests/unit -q
bash run.sh package --experiment config/interformer --output-dir /tmp/interformer-online --force
```

Local defaults:

- All local commands use CUDA profile `cuda126`; setting `TAAC_CUDA_PROFILE` or `--cuda-profile` to any other value is treated as an error.
- Training, evaluation, inference, and packaging commands all reuse the same `cuda126` environment; pytest, hypothesis, and benchmark tooling are part of the default dependencies and should be run with `uv run pytest`.
- `TAAC_SKIP_UV_SYNC=1` skips automatic `uv sync` when the environment is already prepared.
- `TAAC_INSTALL_UV=0` prevents `run.sh` from trying to install `uv` if it is missing.

## Dependency Profiles

The project requires Python `>=3.10,<3.14`.

Important extras:

- `cuda126`: CUDA 12.6 PyTorch, FBGEMM, and TorchRec runtime for the repository.

Default dependencies already include pytest, hypothesis, and benchmark tooling.

Do not point `uv` at an alternate package index unless you are intentionally updating dependency resolution. The lockfile is expected to resolve against the indexes declared in `pyproject.toml`.

## Online Bundle Shape

The online platform is expected to recognize and execute only top-level `run.sh`. The upload directory must contain exactly the runnable entry script and code package:

```text
<training_bundle>/
├── run.sh
└── code_package.zip
```

Build it locally with:

```bash
bash run.sh package --experiment config/baseline --output-dir outputs/training_bundles/baseline_training_bundle --force
bash run.sh package --experiment config/interformer --output-dir outputs/training_bundles/interformer_training_bundle --force
bash run.sh package --experiment config/onetrans --output-dir outputs/training_bundles/onetrans_training_bundle --force
```

The package command calls `taac-package-train`, which writes:

- `run.sh`: copied from the repository root and marked executable.
- `code_package.zip`: minimal runtime source tree.

The zip contains `project/.taac_training_manifest.json`, `pyproject.toml`, `uv.lock`, `README.md` when present, `src/taac2026`, and only the selected experiment package under `config/<experiment>`. It must not include tests, docs, or unrelated experiment packages.

## Online Conda + pip / Python Runtime

Online bundle mode must not depend on `uv`.

Use the platform-provided Conda environment when available:

```bash
conda activate <platform-env>
export TAAC_PYTHON="$(command -v python)"
export TAAC_RUNNER=python
```

Dependency responsibility online:

- Prefer the platform or image-provided CUDA, PyTorch, FBGEMM, and TorchRec stack.
- Use Conda for the base Python/CUDA/PyTorch environment if the platform allows custom images or startup commands.
- Use pip inside that Conda environment only for missing pure-Python packages that are not already available.
- Do not call `uv sync` or require `uv.lock` online; `uv.lock` is packaged for provenance and local reproducibility, not online installation.

If the platform image allows a pre-run dependency step, use the active Conda Python:

```bash
python -m pip install --upgrade pip
python -m pip install numpy pyarrow scikit-learn rich tensorboard tqdm optuna tomli
```

Install the CUDA PyTorch stack only through the platform-approved channel. Avoid pip-installing a second incompatible Torch stack over the platform environment.

## Online Run Procedure

After uploading `run.sh` and `code_package.zip` to the same directory, configure paths and run the script:

```bash
export TAAC_DATASET_PATH=/path/to/train.parquet_or_dataset_dir
export TAAC_SCHEMA_PATH=/path/to/schema.json
export TAAC_OUTPUT_DIR=/path/to/output
export TAAC_RUNNER=python
bash run.sh --compile --amp --amp-dtype bfloat16
```

Important runtime variables:

- `TAAC_DATASET_PATH` or `TRAIN_DATA_PATH`: parquet file or parquet directory; usually required for training.
- `TAAC_SCHEMA_PATH` or `TRAIN_SCHEMA_PATH`: official `schema.json` when it is not colocated with the parquet data.
- `TAAC_OUTPUT_DIR` or `TRAIN_CKPT_PATH`: training output directory.
- `TAAC_EXPERIMENT`: override the bundled experiment path; normally leave unset so the manifest decides.
- `TAAC_BUNDLE_WORKDIR`: directory where `code_package.zip` is extracted.
- `TAAC_CODE_PACKAGE`: non-default path to `code_package.zip`.
- `TAAC_FORCE_EXTRACT=1`: force re-extraction of the zip.
- `TAAC_PYTHON`: explicit Python interpreter, often the active Conda interpreter.
- `TAAC_RUNNER=python`: force online no-uv execution.

In bundle mode, `run.sh` extracts the zip to `.taac_bundle/project`, sets:

```bash
PYTHONPATH="<bundle-workdir>/project/src:<bundle-workdir>/project:${PYTHONPATH}"
```

Then it invokes:

```bash
python -m taac2026.application.training.cli --experiment <manifest experiment> ...
```

## Competition Workflow

Use this lifecycle for competition work:

1. Develop locally with `uv sync --locked --extra cuda126`.
2. Add or modify an experiment package under `config/<name>`.
3. Run focused unit tests locally with `uv run pytest tests/unit -q`.
4. For training experiments, keep the same `cuda126` environment and train through `bash run.sh train --experiment config/<name>`.
5. Build the online bundle with `bash run.sh package --experiment config/<name> --force`.
6. Inspect `code_package.zip` when changing packaging logic; confirm it contains the selected experiment package and required assets such as `ns_groups.json`.
7. Upload only `run.sh` and `code_package.zip` to the online platform.
8. Run online in the platform Conda/Python environment with `TAAC_RUNNER=python` and dataset/output environment variables.
9. Collect checkpoints, logs, tensorboard events, predictions, and sidecars from the platform output directory.

## Validation Commands

Local validation:

```bash
uv run pytest tests/unit -q
uv run --with ruff ruff check src/taac2026 tests/unit
```

Bundle validation:

```bash
bash run.sh package --experiment config/interformer --output-dir /tmp/interformer-bundle --force
python -m zipfile -l /tmp/interformer-bundle/code_package.zip | head
```

Online-style local smoke without `uv`:

```bash
export TAAC_RUNNER=python
export TAAC_DATASET_PATH=/path/to/train.parquet_or_dataset_dir
export TAAC_SCHEMA_PATH=/path/to/schema.json
bash /tmp/interformer-bundle/run.sh --device cpu --num_epochs 1 --batch_size 8
```

Use tiny or sample data for smoke tests; full training should use the platform GPU environment.

## Troubleshooting

If `pyproject.toml not found` appears, upload `run.sh` beside `code_package.zip`, or run from the repository root.

If imports fail online, confirm `run.sh` extracted `code_package.zip` and that `PYTHONPATH` includes both `project/src` and `project`.

If a dependency is missing online, install it into the active Conda environment with `python -m pip install ...` before running `run.sh`, or rebuild the platform image. Do not switch the bundle runner to `uv` unless the platform explicitly provides `uv` and network access.

If Torch, CUDA, FBGEMM, or TorchRec versions conflict online, fix the Conda/platform image rather than pip-overwriting core GPU packages inside the job.

If the wrong experiment runs online, inspect `project/.taac_training_manifest.json` inside `code_package.zip` and check whether `TAAC_EXPERIMENT` is overriding the manifest.

If stale extracted code is reused, set:

```bash
export TAAC_FORCE_EXTRACT=1
```

Then rerun `bash run.sh`.