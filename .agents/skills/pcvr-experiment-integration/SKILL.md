---
name: pcvr-experiment-integration
description: 'Use when creating, reviewing, refactoring, or packaging TAAC PCVR experiment packages under experiments/<name>, including baseline, InterFormer, OneTrans, Symbiosis, and new paper implementations. Covers the current PCVRExperiment constructor, model_class_name, ModelInput, explicit PCVRNSConfig groups in __init__.py, shared application/domain/infrastructure runtime hooks, Bundle packaging, and focused tests under tests/unit/experiments, tests/unit/application, tests/unit/domain, and tests/unit/infrastructure.'
argument-hint: 'experiment path or paper model name'
user-invocable: true
---

# PCVR Experiment Integration

## When to Use

- Add a new PCVR experiment package under `experiments/<name>`.
- Review or refactor an existing PCVR package such as `experiments/baseline`, `experiments/interformer`, or `experiments/onetrans`.
- Change shared PCVR runtime code across `src/taac2026/domain`, `src/taac2026/application`, or `src/taac2026/infrastructure`.
- Build or validate training or inference bundles for PCVR experiments.

## Package Contract

Each PCVR experiment package should keep only experiment-specific assets and model code:

- `__init__.py` should define `EXPERIMENT = create_pcvr_experiment(...)`.
- `package_dir` should be `Path(__file__).resolve().parent`.
- `model_class_name` must explicitly name the model class exported by `model.py`.
- Full experiment-specific model classes must live in package-local `model.py`; do not centralize paper or experiment architecture classes in shared domain/application/infrastructure modules.
- Non-HyFormer papers must use paper-specific names such as `PCVRInterFormer` or `PCVROneTrans`; do not expose `PCVRHyFormer` outside the official HyFormer baseline.
- The official baseline may keep `PCVRHyFormer` because it is the HyFormer implementation.
- Do not add package-local `run.sh`, `train.py`, or `trainer.py`; shared PCVR runtime owns training, evaluation, inference, and packaging.

Current minimum package shape:

```text
experiments/<name>/
├── __init__.py
└── model.py
```

Optional package-local helpers such as `layers.py` are fine when the model body would otherwise become too large.

Example:

```python
from pathlib import Path

from taac2026.api import PCVRModelConfig, PCVRNSConfig, PCVRTrainConfig, create_pcvr_experiment


EXPERIMENT = create_pcvr_experiment(
    name="pcvr_example",
    package_dir=Path(__file__).resolve().parent,
    model_class_name="PCVRExampleModel",
    train_defaults=PCVRTrainConfig(
        model=PCVRModelConfig(num_blocks=2),
        ns=PCVRNSConfig(
            grouping_strategy="explicit",
            user_groups={"U1": [1, 15]},
            item_groups={"I1": [11, 13]},
            tokenizer_type="rankmixer",
            user_tokens=5,
            item_tokens=2,
        ),
    ),
)

TRAIN_HOOKS = EXPERIMENT.train_hooks
PREDICTION_HOOKS = EXPERIMENT.prediction_hooks
RUNTIME_HOOKS = EXPERIMENT.runtime_hooks
```

For new model-only experiments, use `create_pcvr_experiment(...)` and replace the experiment name, model class, and defaults. Avoid inventing new hooks unless the behavior really differs; when it does, pass only the differing functions through `train_hook_overrides`, `prediction_hook_overrides`, or `runtime_hook_overrides`.

## Model Contract

`model.py` must expose the `ModelInput` type and the model class named by `model_class_name`.

Prefer importing currently shared building blocks from `taac2026.infrastructure.modeling`:

- `ModelInput`
- mask and pooling helpers such as `make_padding_mask`, `masked_mean`, `masked_last`, and `safe_key_padding_mask`
- tokenizer and embedding helpers such as `FeatureEmbeddingBank`, `NonSequentialTokenizer`, `DenseTokenProjector`, and `SequenceTokenizer`
- `RMSNorm` plus `configure_rms_norm_runtime` for runtime-selectable Torch / TileLang RMSNorm
- `EmbeddingParameterMixin` for sparse/dense parameter grouping and high-cardinality reinitialization

Package-local `layers.py` files in the built-in non-baseline experiments are compatibility re-exports. New experiments should import these shared primitives directly from `taac2026.infrastructure.modeling` unless they are implementing truly paper-specific layers.

Reusable modeling primitives live under `src/taac2026/infrastructure/modeling/`. Keep model bodies, paper-specific blocks, and experiment naming inside the owning `experiments/<experiment>/model.py` package.

The model constructor must accept the arguments currently passed by `build_pcvr_model`:

```python
user_int_feature_specs: list[tuple[int, int, int]]
item_int_feature_specs: list[tuple[int, int, int]]
user_dense_dim: int
item_dense_dim: int
seq_vocab_sizes: dict[str, list[int]]
user_ns_groups: list[list[int]]
item_ns_groups: list[list[int]]
d_model: int = 64
emb_dim: int = 64
num_queries: int = 1
num_blocks: int = 2
num_heads: int = 4
seq_encoder_type: str = "transformer"
hidden_mult: int = 4
dropout_rate: float = 0.01
seq_top_k: int = 50
seq_causal: bool = False
action_num: int = 1
num_time_buckets: int = 65
rank_mixer_mode: str = "full"
use_rope: bool = False
rope_base: float = 10000.0
emb_skip_threshold: int = 0
seq_id_threshold: int = 10000
gradient_checkpointing: bool = False
ns_tokenizer_type: str = "rankmixer"
user_ns_tokens: int = 0
item_ns_tokens: int = 0
```

If the package needs RMSNorm runtime switching, expose an optional module-level `configure_rms_norm_runtime(...)` hook; `rms_norm_backend` and `rms_norm_block_rows` are configured there, not passed directly to the model constructor.

Behavioral requirements:

- `forward(inputs: ModelInput)` returns logits.
- `predict(inputs: ModelInput)` returns `(logits, embeddings)`.
- The model exposes `num_ns` for logging and checkpoint metadata.
- Use the package-local `EmbeddingParameterMixin` pattern unless the model has a deliberate custom sparse/dense parameter split.
- Use `num_blocks`; do not introduce `num_hyformer_blocks` in new non-baseline code.

## NS Groups

Current repository code keeps NS grouping in `PCVRNSConfig` inside `__init__.py`; do not require a package-local `ns_groups.json` file.

Rules:

- `train_defaults` should include an explicit `PCVRNSConfig(...)`.
- Evaluation and inference must read a complete checkpoint-side `train_config.json`; missing config keys should fail instead of falling back to experiment defaults.
- Explicit grouping should remain part of `train_config.json`, so evaluation and inference reuse the training grouping.
- Keep metadata keys prefixed with `_`; the runtime only consumes `user_ns_groups` and `item_ns_groups`.

Expected config shape:

```python
PCVRNSConfig(
    grouping_strategy="explicit",
    metadata={"_purpose": "Shared PCVR non-sequential feature grouping."},
    user_groups={"U1": [1, 15]},
    item_groups={"I1": [11, 13]},
    tokenizer_type="rankmixer",
    user_tokens=5,
    item_tokens=2,
)
```

## Shared Runtime

The shared PCVR runtime is split by ownership:

- `src/taac2026/application/experiments/` owns `PCVRExperiment`, package discovery/registry, and experiment factory wiring.
- `src/taac2026/domain/model_contract.py` owns `ModelInput`, schema-to-model conversion, and batch-to-input conversion.
- `src/taac2026/domain/config.py` and `src/taac2026/domain/sidecar.py` own PCVR config and checkpoint-side contract data.
- `src/taac2026/application/training/` and `src/taac2026/application/evaluation/` own CLI workflows and use cases.
- `src/taac2026/infrastructure/data/`, `runtime/`, `optimization/`, and `accelerators/` own dataset loading, trainer mechanics, optimizers, and operator implementations.

Online bundles currently use two top-level files, but there are two different shapes:

- training: `run.sh` + `code_package.zip`
- inference: `infer.py` + `code_package.zip`

The code package should include `project/.taac_*_manifest.json`, `project/pyproject.toml`, `project/src/taac2026`, and the selected experiment package under `project/experiments/...`. It should not include `tests/`, `uv.lock`, `README.md`, or unrelated experiment packages.
Runtime platform behavior for `run.sh` and generated `infer.py` lives in `src/taac2026/application/bootstrap` and `src/taac2026/infrastructure/platform`; keep shell/bootstrap code thin and put manifest, pip, and env parsing in those owners.

## Official Baseline Snapshot Notes

Official baseline source snapshots are disposable references only. Read them for contracts, summarize durable lessons in docs/skills without naming source-drop paths, and remove the source drops after migration; do not package or depend on them as long-lived repository code.

The official self-contained training baseline contains `run.sh`, a training entrypoint, a trainer, utilities, dataset code, model code, and `ns_groups.json`. Its active `run.sh` used `ns_tokenizer_type=rankmixer`, `user_ns_tokens=5`, `item_ns_tokens=2`, `num_queries=2`, `ns_groups_json=""`, `emb_skip_threshold=1000000`, and `num_workers=8`. The commented group-tokenizer alternative uses `ns_groups.json` with `num_queries=1` because `rank_mixer_mode=full` requires `d_model % (num_queries * num_sequences + num_ns) == 0`.

The official scoring baseline shares the same dataset and model definitions as the training snapshot. Its `infer.py` rebuilds `PCVRHyFormer` from checkpoint-side `schema.json`, optional `ns_groups.json`, and `train_config.json`, then strictly loads `model.safetensors` and writes `predictions.json` as `{"predictions": {user_id: probability}}`.

When adapting this snapshot into `experiments/baseline`, keep the model body local but adapt the public constructor to the shared contract. In particular, the official `num_hyformer_blocks` argument maps to the shared `num_blocks` argument.

Checkpoint sidecars are part of the model contract, not optional convenience files. Training must persist `model.safetensors`, `schema.json`, and `train_config.json`, because evaluation/inference use those files as the source of truth for model reconstruction.

## Tests to Update

When adding or changing a PCVR experiment package, update focused tests instead of relying only on a smoke run:

- `tests/unit/experiments/test_packages.py`: experiment loading, `model_class_name`, no leaked `PCVRHyFormer` for non-baseline packages, forward, backward, and predict.
- `tests/unit/application/experiments/test_discovery.py`: package discovery under `experiments/<name>`.
- `tests/unit/experiments/test_runtime_contract_matrix.py`: schema-to-feature-spec conversion and runtime config contract.
- `tests/unit/domain/test_model_contract.py`: batch conversion and schema protocol behavior.
- `tests/unit/application/packaging/test_training.py`: training bundle contains the expected runtime sources and manifests.
- `tests/unit/application/packaging/test_inference.py`: inference bundle contains the expected runtime sources and manifests.
- `tests/unit/application/experiments/test_registry.py`: baseline package loading behavior.
- `tests/unit/infrastructure/test_checkpoints.py`: checkpoint sidecar behavior.

## Verification Commands

Run narrow checks first:

```bash
uv run --with ruff ruff check src/taac2026/domain src/taac2026/application src/taac2026/infrastructure experiments/<experiment> tests/unit/experiments/test_packages.py
uv run pytest tests/unit/experiments/test_packages.py -q
uv run pytest tests/unit/application/experiments/test_discovery.py -q
uv run pytest tests/unit/experiments/test_runtime_contract_matrix.py -q
```

For a smoke run, prefer a tiny CPU command over a long GPU training job:

```bash
uv run taac-train --experiment experiments/<experiment> \
  --dataset-path data/sample_1000_raw/demo_1000.parquet \
  --schema-path data/sample_1000_raw/schema.json \
  --run-dir outputs/<experiment>_smoke \
  --device cpu --num_workers 0 --batch_size 8 --max_steps 1
```

Then run the current unit suite before finishing:

```bash
uv run pytest tests/unit -q
```

For bundle changes, verify the package output:

```bash
uv run taac-package-train --experiment experiments/<experiment> --output-dir outputs/training_bundles/<experiment>_training_bundle --json
uv run taac-package-infer --experiment experiments/<experiment> --output-dir outputs/inference_bundles/<experiment>_inference_bundle --json
```

## Review Checklist

- Package uses `PCVRExperiment`, not a bespoke experiment adapter.
- `model_class_name` matches the class exported by `model.py`.
- Non-baseline packages do not expose `PCVRHyFormer` or use HyFormer-specific CLI names.
- NS grouping is declared explicitly in `PCVRNSConfig` and survives into `train_config.json`.
- The model accepts the shared constructor contract and uses `num_blocks`.
- `forward`, `loss.backward`, and `predict` pass on a synthetic `ModelInput`.
- Training and inference bundles include the selected experiment package and the correct `project/.taac_*_manifest.json` file.