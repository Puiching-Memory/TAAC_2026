---
name: pcvr-experiment-integration
description: 'Use when creating, reviewing, refactoring, or packaging TAAC PCVR experiment packages under config/*, including baseline, InterFormer, OneTrans, and new paper implementations. Covers PCVRExperiment, model_class_name, ModelInput, ns_groups.json, shared training/runtime, online bundle packaging, and tests.'
argument-hint: 'experiment path or paper model name'
user-invocable: true
---

# PCVR Experiment Integration

## When to Use

- Add a new PCVR experiment package under `config/<name>`.
- Review or refactor an existing PCVR package such as `config/baseline`, `config/interformer`, or `config/onetrans`.
- Change shared PCVR runtime code in `src/taac2026/infrastructure/pcvr`.
- Build or validate online training bundles for PCVR experiments.

## Package Contract

Each PCVR experiment package should keep only experiment-specific assets and model code:

- `__init__.py` must define `EXPERIMENT = PCVRExperiment(...)`.
- `package_dir` should be `Path(__file__).resolve().parent`.
- `model_class_name` must explicitly name the model class exported by `model.py`.
- Full experiment-specific model classes must live in package-local `model.py`; do not centralize paper or experiment architecture classes in `src/taac2026/infrastructure/pcvr`.
- Non-HyFormer papers must use paper-specific names such as `PCVRInterFormer` or `PCVROneTrans`; do not expose `PCVRHyFormer` outside the official HyFormer baseline.
- The official baseline may keep `PCVRHyFormer` because it is the HyFormer implementation.
- Do not add package-local `run.sh`, `train.py`, or `trainer.py`; shared PCVR runtime owns training, evaluation, inference, and packaging.

Example:

```python
from pathlib import Path

from taac2026.infrastructure.pcvr.experiment import PCVRExperiment


EXPERIMENT = PCVRExperiment(
    name="pcvr_example",
    package_dir=Path(__file__).resolve().parent,
    model_class_name="PCVRExampleModel",
    default_train_args=(
        "--ns_groups_json",
        "ns_groups.json",
        "--num_blocks",
        "2",
    ),
)
```

## Model Contract

`model.py` must expose the `ModelInput` type and the model class named by `model_class_name`.

Prefer importing shared building blocks from `taac2026.infrastructure.pcvr.modeling`:

- `ModelInput`
- `EmbeddingParameterMixin`
- `NonSequentialTokenizer`
- `DenseTokenProjector`
- `SequenceTokenizer`
- mask and pooling helpers such as `make_padding_mask`, `masked_mean`, `masked_last`, and `safe_key_padding_mask`

`src/taac2026/infrastructure/pcvr/modeling.py` is for reusable primitives only. Keep model bodies, paper-specific blocks, and experiment naming inside the owning `config/<experiment>/model.py` package.

The model constructor must accept the arguments passed by `build_pcvr_model`:

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
ns_tokenizer_type: str = "rankmixer"
user_ns_tokens: int = 0
item_ns_tokens: int = 0
```

Behavioral requirements:

- `forward(inputs: ModelInput)` returns logits.
- `predict(inputs: ModelInput)` returns `(logits, embeddings)`.
- The model exposes `num_ns` for logging and checkpoint metadata.
- Use `EmbeddingParameterMixin` unless the model has a deliberate custom sparse/dense parameter split.
- Use `num_blocks`; do not introduce `num_hyformer_blocks` in new non-baseline code.

## NS Groups

Every PCVR experiment package must include a package-local `ns_groups.json`.

Rules:

- Default train args should include `--ns_groups_json ns_groups.json`.
- Shared config in `DEFAULT_PCVR_MODEL_CONFIG` defaults to `ns_groups.json`.
- If an explicitly configured NS groups file is missing, fail fast with `FileNotFoundError`; do not silently fall back to singleton groups.
- Checkpoints copy the resolved NS groups file as `ns_groups.json`, so evaluation and inference reuse the training grouping.
- Keep metadata keys prefixed with `_`; the runtime only consumes `user_ns_groups` and `item_ns_groups`.

Expected JSON shape:

```json
{
  "_purpose": "Shared PCVR non-sequential feature grouping for this experiment package.",
  "user_ns_groups": {
    "U1": [1, 15]
  },
  "item_ns_groups": {
    "I1": [11, 13]
  }
}
```

## Shared Runtime

The shared runtime lives under `src/taac2026/infrastructure/pcvr`:

- `PCVRExperiment` loads package modules and delegates train/evaluate/infer.
- `build_pcvr_model` converts dataset schema and config into model constructor arguments.
- `train_pcvr_model` owns CLI parsing, data loader setup, model construction, logging, and trainer wiring.
- `PCVRPointwiseTrainer` owns optimization, evaluation steps, checkpoints, and sidecars.

Online training bundles should remain the platform two-file shape:

- `run.sh`
- `code_package.zip`

The code package should include `src/taac2026`, `pyproject.toml`, `uv.lock` when present, and the selected experiment package including `model.py` and `ns_groups.json`.

## Tests to Update

When adding or changing a PCVR experiment package, update focused tests instead of relying only on a smoke run:

- `tests/unit/test_experiment_packages.py`: experiment loading, `model_class_name`, no leaked `PCVRHyFormer` for non-baseline packages, `ns_groups.json`, forward, backward, and predict.
- `tests/unit/test_pcvr_protocol.py`: schema-to-feature-spec conversion, NS groups mapping, and missing explicit NS groups failure.
- `tests/unit/test_package_training.py`: bundle contains runtime sources and package-local `ns_groups.json`.
- `tests/unit/test_checkpoint_and_loader.py`: baseline package loading and checkpoint sidecar behavior.

## Verification Commands

Run narrow checks first:

```bash
python -m json.tool config/<experiment>/ns_groups.json >/dev/null
uv run --with ruff ruff check src/taac2026/infrastructure/pcvr config/<experiment> tests/unit/test_experiment_packages.py
uv run pytest tests/unit/test_experiment_packages.py -q
```

Then run the current unit suite before finishing:

```bash
uv run pytest tests/unit -q
```

For bundle changes, verify the package output:

```bash
uv run taac-package-train --experiment config/<experiment> --output-dir outputs/training_bundles/<experiment>_training_bundle --force --json
```

## Review Checklist

- Package uses `PCVRExperiment`, not a bespoke experiment adapter.
- `model_class_name` matches the class exported by `model.py`.
- Non-baseline packages do not expose `PCVRHyFormer` or use HyFormer-specific CLI names.
- `ns_groups.json` exists in the package and is enabled by default.
- Missing explicit NS groups paths fail instead of falling back.
- The model accepts the shared constructor contract and uses `num_blocks`.
- `forward`, `loss.backward`, and `predict` pass on a synthetic `ModelInput`.
- Training bundle includes the selected experiment package and `ns_groups.json`.