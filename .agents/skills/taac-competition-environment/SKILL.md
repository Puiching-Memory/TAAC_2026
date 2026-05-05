---
name: taac-competition-environment
description: 'Use when setting up, documenting, debugging, or reviewing the current TAAC 2026 environment flow: local uv development, CUDA dependency profile cuda126, run.sh train/val/eval/infer behavior, taac-package-train / taac-package-infer, Bundle layouts under experiments/*, and online Conda+pip/Python execution without uv.'
argument-hint: 'local setup, package, online run, or env debug'
user-invocable: true
---

# TAAC Competition Environment

## When to Use

- Set up or debug the local TAAC development environment.
- Decide whether a command should run through `uv` or plain Python.
- Build, inspect, or explain online training or inference bundles.
- Prepare online platform runs where the platform executes `run.sh` or `infer.py`.
- Review docs or scripts that mention package shape, dependency installation, Conda, pip, CUDA profiles, or competition workflow.

## Environment Model

This repository deliberately uses two different environment modes:

- Local repository mode uses `uv` for dependency resolution, lockfile fidelity, and command execution.
- Online bundle mode uses the platform's already activated Python or Conda environment and runs with plain `python`; do not require `uv` online.

The same top-level `run.sh` supports both modes:

- If `code_package.zip` exists beside `run.sh`, bundle mode is enabled and the default runner is `python`.
- If running from the repository root without `code_package.zip`, local mode is enabled and the default runner is `uv`.
- `TAAC_RUNNER=python|uv` can override the default when debugging.
- `run.sh` is a thin bootstrapper; command parsing, manifest defaults, pip install behavior, and runner dispatch live in `taac2026.application.bootstrap.run_sh`.

Important current behavior:

- `run.sh` only supports `train`, `val`, `eval`, and `infer`.
- `run.sh` no longer supports `package`, `package-infer`, or `test`.
- Packaging is handled by `taac-package-train` and `taac-package-infer`.
- Experiment paths live under `experiments/...`, not `config/...`.

## Local Development With uv

Use `uv` locally because `pyproject.toml` and `uv.lock` are the source of truth for development dependencies.

Recommended bootstrap:

```bash
git lfs install
git lfs pull
uv python install 3.10.20
uv sync --locked --extra cuda126
uv sync --locked --extra dev --extra cuda126  # when running tests, lint, or local docs
```

For local training or validation, use the only CUDA profile currently supported by the project:

```bash
uv sync --locked --extra cuda126
```

Use the top-level entrypoint instead of calling console scripts directly:

```bash
bash run.sh train --experiment experiments/baseline \
    --dataset-path /path/to/parquet_or_dir \
    --schema-path /path/to/schema.json

uv run pytest tests/unit -q
uv run taac-package-train --experiment experiments/interformer --output-dir /tmp/interformer-training
uv run taac-package-infer --experiment experiments/interformer --output-dir /tmp/interformer-inference
```

Local defaults:

- All local `run.sh` commands use CUDA profile `cuda126`; setting `TAAC_CUDA_PROFILE` or `--cuda-profile` to any other value is treated as an error.
- Training, evaluation, inference, and local CLI tooling all reuse the same `cuda126` environment; pytest, coverage, Ruff, Vulture, benchmark tooling, and Zensical live in the `dev` extra and should be installed with `uv sync --locked --extra dev --extra cuda126` when needed.
- `TAAC_SKIP_UV_SYNC=1` skips automatic `uv sync` when the environment is already prepared.

## Dependency Profiles

The project requires Python `>=3.10,<3.14`.

Important extras:

- `dev`: Ruff, Vulture, pytest, hypothesis, benchmark tooling, coverage helpers, and Zensical for local testing and docs work.
- `cuda126`: CUDA 12.6 PyTorch, FBGEMM, and TorchRec runtime for the repository.

Do not point `uv` at an alternate package index unless you are intentionally updating dependency resolution. The lockfile is expected to resolve against the indexes declared in `pyproject.toml`.

## Online Bundle Shapes

### Training Bundle

The training upload directory contains exactly:

```text
<training_bundle>/
├── run.sh
└── code_package.zip
```

Build it locally with:

```bash
uv run taac-package-train --experiment experiments/baseline --output-dir outputs/training_bundles/baseline_training_bundle
uv run taac-package-train --experiment experiments/interformer --output-dir outputs/training_bundles/interformer_training_bundle
uv run taac-package-train --experiment experiments/onetrans --output-dir outputs/training_bundles/onetrans_training_bundle
```

`taac-package-train` writes:

- `run.sh`: copied from the repository root and marked executable.
- `code_package.zip`: minimal runtime source tree.

The zip contains `project/.taac_training_manifest.json`, `project/pyproject.toml`, `project/src/taac2026`, and only the selected experiment package under `project/experiments/<experiment>`. It must not include tests, docs, unrelated experiment packages, or local provenance files such as `uv.lock` and `README.md`.

### Inference Bundle

The inference upload directory contains exactly:

```text
<inference_bundle>/
├── infer.py
└── code_package.zip
```

Build it locally with:

```bash
uv run taac-package-infer --experiment experiments/baseline --output-dir outputs/inference_bundles/baseline_inference_bundle
uv run taac-package-infer --experiment experiments/interformer --output-dir outputs/inference_bundles/interformer_inference_bundle
```

`taac-package-infer` writes:

- `infer.py`: self-extracting inference entrypoint.
- `code_package.zip`: minimal runtime source tree with `project/.taac_inference_manifest.json`.

The generated `infer.py` imports `taac2026.application.bootstrap.inference_bundle` from the extracted code package, so keep bootstrap and platform runtime modules included in every code package.

## Official Baseline Snapshots

The competition reference snapshots may appear as top-level source drops:

- self-contained training baseline with `run.sh`, a training entrypoint, a trainer, utilities, dataset code, model code, and `ns_groups.json`.
- self-contained final scoring baseline with `infer.py`, dataset code, and model code.

Treat these sources as disposable references. Extract the contracts, update repository docs/skills without naming source-drop paths, and delete the source drops when finished. Do not include source drops in generated bundles; build from `experiments/<experiment>` with the packaging CLIs instead.

The official training snapshot reads `TRAIN_DATA_PATH`, `TRAIN_CKPT_PATH`, `TRAIN_LOG_PATH`, and `TRAIN_TF_EVENTS_PATH`. The official scoring snapshot reads `MODEL_OUTPUT_PATH`, `EVAL_DATA_PATH`, and `EVAL_RESULT_PATH`, then writes `predictions.json` under `EVAL_RESULT_PATH` with the shape `{"predictions": {user_id: probability}}`.

The important portable contract between training and scoring is the checkpoint directory. In current repository code, evaluation and inference rebuild the model primarily from `model.safetensors`, `schema.json`, and `train_config.json`; do not assume a package-local `ns_groups.json` file is part of the current workspace contract.

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
- Packaged `run.sh` and `infer.py` default to `pip install .` and therefore skip the repository `dev` extra; use `TAAC_BUNDLE_PIP_EXTRAS` only when you intentionally need an extra inside bundle mode.
- Do not call `uv sync` or require `uv.lock` online; `uv.lock` remains a local repository artifact for provenance and reproducibility and is not packaged into online bundles.

If the platform image allows a pre-run dependency step, use the active Conda Python:

```bash
python -m pip install --upgrade pip
python -m pip install numpy pyarrow scikit-learn rich tensorboard tqdm optuna tomli
```

Install the CUDA PyTorch stack only through the platform-approved channel. Avoid pip-installing a second incompatible Torch stack over the platform environment.

## Online Run Procedure

### Training Bundle

After uploading `run.sh` and `code_package.zip` to the same directory, configure paths and run the script:

```bash
export TRAIN_DATA_PATH=/path/to/train.parquet_or_dataset_dir
export TAAC_SCHEMA_PATH=/path/to/schema.json
export TRAIN_CKPT_PATH=/path/to/output
export TAAC_RUNNER=python
bash run.sh --compile --amp --amp-dtype bfloat16
```

### Inference Bundle

After uploading `infer.py` and `code_package.zip` to the same directory, configure paths and run the script:

```bash
export EVAL_DATA_PATH=/path/to/eval.parquet_or_dir
export EVAL_RESULT_PATH=/path/to/result_dir
export MODEL_OUTPUT_PATH=/path/to/model.safetensors
export TAAC_SCHEMA_PATH=/path/to/schema.json
python infer.py
```

Important runtime variables:

- `TRAIN_DATA_PATH`: parquet file or parquet directory; usually required for training.
- `TAAC_SCHEMA_PATH`: official `schema.json` when it is not colocated with the parquet data.
- `TRAIN_CKPT_PATH`: training output directory.
- `TAAC_EXPERIMENT`: override the bundled experiment path; normally leave unset so the manifest decides.
- `TAAC_BUNDLE_WORKDIR`: directory where `code_package.zip` is extracted.
- `TAAC_CODE_PACKAGE`: non-default path to `code_package.zip`.
- `TAAC_FORCE_EXTRACT=1`: force re-extraction of the zip.
- `TAAC_BUNDLE_PIP_EXTRAS`: optional extras to install in bundle mode; default is empty so packaged train/infer skip `dev`.
- `TAAC_PYTHON`: explicit Python interpreter, often the active Conda interpreter.
- `TAAC_RUNNER=python`: force online no-uv execution.
- `EVAL_DATA_PATH`: inference dataset path.
- `EVAL_RESULT_PATH`: inference output directory.
- `MODEL_OUTPUT_PATH`: checkpoint path used by `infer.py`.

In training bundle mode, `run.sh` extracts the zip to `.taac_bundle/project`, sets:

```bash
PYTHONPATH="<bundle-workdir>/project/src:<bundle-workdir>/project:${PYTHONPATH}"
```

Then it invokes:

```bash
python -m taac2026.application.training.cli --experiment <manifest experiment> ...
```

The Python module used by `run.sh` is `taac2026.application.bootstrap.run_sh`; it owns manifest reading, `TAAC_BUNDLE_PIP_EXTRAS`, `TAAC_SKIP_PIP_INSTALL`, and the `train` / `val` / `eval` / `infer` argument mapping.

In inference bundle mode, `infer.py` performs the same `project/` extraction and then invokes:

```bash
python -m taac2026.application.evaluation.infer
```

The generated script delegates manifest reading, pip installation, default experiment selection, and final import setup to `taac2026.application.bootstrap.inference_bundle`.

## Competition Workflow

Use this lifecycle for competition work:

1. Develop locally with `uv sync --locked --extra cuda126`.
2. Install local dev tooling with `uv sync --locked --extra dev --extra cuda126` when you need pytest, Ruff, or docs preview.
3. Add or modify an experiment package under `experiments/<name>`.
4. Run focused unit tests locally with `uv run pytest tests/unit -q`.
5. For training experiments, keep the same `cuda126` environment and train through `bash run.sh train --experiment experiments/<name>`.
6. Build the online bundle with `uv run taac-package-train` or `uv run taac-package-infer`.
7. Inspect `code_package.zip` when changing packaging logic; confirm it contains the selected experiment package under `project/experiments/...` and the expected manifest under `project/`.
8. Upload only the two generated top-level files for the bundle type you need.
9. Run online in the platform Conda/Python environment with `TAAC_RUNNER=python` for training bundles, or direct `python infer.py` for inference bundles.
10. Collect checkpoints, logs, tensorboard events, predictions, and sidecars from the platform output directory.

## Validation Commands

Local validation:

```bash
uv run pytest tests/unit -q
uv run --with ruff ruff check .
```

Bundle validation:

```bash
uv run taac-package-train --experiment experiments/interformer --output-dir /tmp/interformer-training --json
uv run taac-package-infer --experiment experiments/interformer --output-dir /tmp/interformer-inference --json

python -m zipfile -l /tmp/interformer-training/code_package.zip | head
python -m zipfile -l /tmp/interformer-inference/code_package.zip | head
```

Online-style local smoke without `uv`:

```bash
export TAAC_RUNNER=python
export TRAIN_DATA_PATH=/path/to/train.parquet_or_dataset_dir
export TAAC_SCHEMA_PATH=/path/to/schema.json
bash /tmp/interformer-training/run.sh --device cpu --max_steps 1 --batch_size 8 --num_workers 0
```

Inference-style local smoke:

```bash
export TAAC_BUNDLE_WORKDIR=/tmp/interformer-inference-workdir
export EVAL_DATA_PATH=/path/to/eval.parquet_or_dir
export EVAL_RESULT_PATH=/tmp/interformer-infer-results
export MODEL_OUTPUT_PATH=/path/to/model.safetensors
export TAAC_SCHEMA_PATH=/path/to/schema.json
python /tmp/interformer-inference/infer.py
```

Use tiny or sample data for smoke tests; full training should use the platform GPU environment.

## Troubleshooting

If `pyproject.toml not found` appears, upload `run.sh` beside `code_package.zip`, or run from the repository root.

If imports fail online, confirm `run.sh` extracted `code_package.zip` and that `PYTHONPATH` includes both `project/src` and `project`.

If a dependency is missing online, install it into the active Conda environment with `python -m pip install ...` before running `run.sh`, or rebuild the platform image. Do not switch the bundle runner to `uv` unless the platform explicitly provides `uv` and network access.

If Torch, CUDA, FBGEMM, or TorchRec versions conflict online, fix the Conda/platform image rather than pip-overwriting core GPU packages inside the job.

If the wrong experiment runs online, inspect `project/.taac_training_manifest.json` inside `code_package.zip` and check whether `TAAC_EXPERIMENT` is overriding the manifest.

If `infer.py` fails before extraction, set `TAAC_BUNDLE_WORKDIR` explicitly; some platforms may not provide `USER_CACHE_PATH`.

If stale extracted code is reused, set:

```bash
export TAAC_FORCE_EXTRACT=1
```

Then rerun `bash run.sh`.