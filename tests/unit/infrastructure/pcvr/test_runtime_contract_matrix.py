from __future__ import annotations

import zipfile
from pathlib import Path
from types import SimpleNamespace
from typing import NamedTuple

import pytest
import torch

from taac2026.application.maintenance.bundle_packaging import build_training_bundle
from taac2026.infrastructure.checkpoints import (
    build_checkpoint_dir_name,
    checkpoint_step,
    resolve_checkpoint_path,
    validate_checkpoint_dir_name,
    write_checkpoint_sidecars,
)
from taac2026.infrastructure.experiments.loader import load_experiment_package
from taac2026.infrastructure.pcvr.config import PCVRTrainConfig
from taac2026.infrastructure.pcvr.config_sidecar import (
    PCVR_TRAIN_CONFIG_FORMAT,
    PCVR_TRAIN_CONFIG_VERSION,
    build_pcvr_train_config_sidecar,
)
from taac2026.infrastructure.pcvr.protocol import (
    batch_to_model_input,
    build_feature_specs,
    build_pcvr_model,
    load_ns_groups,
    num_time_buckets,
    parse_seq_max_lens,
    resolve_schema_path,
)
from taac2026.infrastructure.pcvr.runtime_stack import default_load_train_config
from taac2026.infrastructure.io.json_utils import dumps, loads
from tests.unit.infrastructure.pcvr._pcvr_experiment_matrix import ExperimentCase, REPO_ROOT, discover_pcvr_experiment_cases, load_model_module


EXPERIMENT_CASES = discover_pcvr_experiment_cases()


class _ModelInput(NamedTuple):
    user_int_feats: torch.Tensor
    item_int_feats: torch.Tensor
    user_dense_feats: torch.Tensor
    item_dense_feats: torch.Tensor
    seq_data: dict[str, torch.Tensor]
    seq_lens: dict[str, torch.Tensor]
    seq_time_buckets: dict[str, torch.Tensor]


class _RecordingModel(torch.nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.kwargs = kwargs


def _schema(entries: list[tuple[int, int, int]]) -> SimpleNamespace:
    return SimpleNamespace(entries=entries)


def _dataset(user_count: int, item_count: int) -> SimpleNamespace:
    user_entries = [(100 + index, index, 1) for index in range(user_count)]
    item_entries = [(200 + index, index, 1) for index in range(item_count)]
    return SimpleNamespace(
        user_int_schema=_schema(user_entries),
        item_int_schema=_schema(item_entries),
        user_int_vocab_sizes=[10 + index for index in range(user_count)],
        item_int_vocab_sizes=[20 + index for index in range(item_count)],
        user_dense_schema=SimpleNamespace(total_dim=3),
        item_dense_schema=SimpleNamespace(total_dim=2),
        seq_domain_vocab_sizes={"seq_a": [7, 9], "seq_b": [5]},
    )

def _code_package_names(code_package_path: Path) -> set[str]:
    with zipfile.ZipFile(code_package_path) as archive:
        return set(archive.namelist())


def _code_package_manifest(code_package_path: Path) -> dict[str, object]:
    with zipfile.ZipFile(code_package_path) as archive:
        return loads(archive.read("project/.taac_training_manifest.json"))


@pytest.fixture(scope="module", params=EXPERIMENT_CASES, ids=lambda case: case.path)
def experiment_case(request) -> ExperimentCase:
    return request.param


@pytest.fixture(scope="module")
def loaded_experiment(experiment_case: ExperimentCase):
    return load_experiment_package(experiment_case.path)


@pytest.fixture(scope="module")
def loaded_model_module(experiment_case: ExperimentCase):
    return load_model_module(experiment_case)


@pytest.fixture(scope="module")
def built_bundle(experiment_case: ExperimentCase, tmp_path_factory: pytest.TempPathFactory):
    output_dir = tmp_path_factory.mktemp(f"bundle_{Path(experiment_case.path).name}")
    result = build_training_bundle(experiment_case.path, output_dir=output_dir, root=REPO_ROOT)
    return result, _code_package_manifest(result.code_package_path), _code_package_names(result.code_package_path)


@pytest.mark.parametrize(
    ("raw_value", "expected"),
    [
        ("", {}),
        (" , ", {}),
        ("seq_a:4", {"seq_a": 4}),
        ("seq_a:4,seq_b:8", {"seq_a": 4, "seq_b": 8}),
        (" seq_a : 4 ", {"seq_a": 4}),
        ("seq_a:4,,seq_b:8,", {"seq_a": 4, "seq_b": 8}),
        ("seq_a:004", {"seq_a": 4}),
        ("seq_a:4, seq_a:8", {"seq_a": 8}),
        ("seq_a:0", {"seq_a": 0}),
        ("seq_a:3,\nseq_b:5", {"seq_a": 3, "seq_b": 5}),
    ],
)
def test_parse_seq_max_lens_cases(raw_value: str, expected: dict[str, int]) -> None:
    assert parse_seq_max_lens(raw_value) == expected


@pytest.mark.parametrize(
    ("entries", "vocab_sizes", "expected"),
    [
        ([(10, 0, 1)], [11], [(11, 0, 1)]),
        ([(10, 0, 2)], [11, 21], [(21, 0, 2)]),
        ([(10, 0, 2), (20, 2, 1)], [11, 21, 7], [(21, 0, 2), (7, 2, 1)]),
        ([(10, 1, 2)], [3, 5, 8], [(8, 1, 2)]),
        ([(10, 0, 3)], [4, 4, 4], [(4, 0, 3)]),
        ([(10, 2, 2)], [1, 1, 9, 8], [(9, 2, 2)]),
        ([(10, 0, 1), (20, 1, 2)], [2, 6, 5], [(2, 0, 1), (6, 1, 2)]),
        ([(10, 3, 1)], [1, 2, 3, 10], [(10, 3, 1)]),
    ],
)
def test_build_feature_specs_cases(
    entries: list[tuple[int, int, int]],
    vocab_sizes: list[int],
    expected: list[tuple[int, int, int]],
) -> None:
    assert build_feature_specs(_schema(entries), vocab_sizes) == expected


@pytest.mark.parametrize(
    "scenario",
    [
        "explicit",
        "checkpoint",
        "dataset_dir",
        "dataset_parent",
        "explicit_missing_falls_back_to_checkpoint",
        "missing",
    ],
)
def test_resolve_schema_path_cases(tmp_path: Path, scenario: str) -> None:
    dataset_dir = tmp_path / "data_dir"
    dataset_dir.mkdir()
    dataset_file = dataset_dir / "train.parquet"
    dataset_file.write_text("", encoding="utf-8")
    checkpoint_dir = tmp_path / "checkpoints" / "global_step1"
    checkpoint_dir.mkdir(parents=True)
    explicit_path = tmp_path / "explicit_schema.json"

    if scenario == "explicit":
        explicit_path.write_text("{}", encoding="utf-8")
        actual = resolve_schema_path(dataset_dir, explicit_path, checkpoint_dir)
        assert actual == explicit_path.resolve()
        return

    if scenario == "checkpoint":
        checkpoint_schema = checkpoint_dir / "schema.json"
        checkpoint_schema.write_text("{}", encoding="utf-8")
        actual = resolve_schema_path(dataset_dir, None, checkpoint_dir)
        assert actual == checkpoint_schema.resolve()
        return

    if scenario == "dataset_dir":
        dataset_schema = dataset_dir / "schema.json"
        dataset_schema.write_text("{}", encoding="utf-8")
        actual = resolve_schema_path(dataset_dir, None, checkpoint_dir)
        assert actual == dataset_schema.resolve()
        return

    if scenario == "dataset_parent":
        dataset_schema = dataset_dir / "schema.json"
        dataset_schema.write_text("{}", encoding="utf-8")
        actual = resolve_schema_path(dataset_file, None, checkpoint_dir)
        assert actual == dataset_schema.resolve()
        return

    if scenario == "explicit_missing_falls_back_to_checkpoint":
        checkpoint_schema = checkpoint_dir / "schema.json"
        checkpoint_schema.write_text("{}", encoding="utf-8")
        actual = resolve_schema_path(dataset_dir, explicit_path, checkpoint_dir)
        assert actual == checkpoint_schema.resolve()
        return

    with pytest.raises(FileNotFoundError, match=r"schema\.json"):
        resolve_schema_path(dataset_dir, None, checkpoint_dir)


@pytest.mark.parametrize(
    ("user_count", "item_count"),
    [(0, 0), (1, 0), (0, 1), (2, 3), (4, 2)],
)
def test_load_ns_groups_defaults_to_singletons_when_disabled(user_count: int, item_count: int, tmp_path: Path) -> None:
    dataset = _dataset(user_count, item_count)

    user_groups, item_groups = load_ns_groups(
        dataset,
        {
            "ns_grouping_strategy": "singleton",
            "user_ns_groups": {},
            "item_ns_groups": {},
        },
        tmp_path,
        tmp_path,
    )

    assert user_groups == [[index] for index in range(user_count)]
    assert item_groups == [[index] for index in range(item_count)]


@pytest.mark.parametrize(
    ("payload", "expected_user", "expected_item"),
    [
        (
            {"user_ns_groups": {"u": [20, 10]}, "item_ns_groups": {"i": [7]}},
            [[1, 0]],
            [[0]],
        ),
        (
            {"user_ns_groups": {"u1": [30], "u2": [20, 10]}, "item_ns_groups": {"i1": [8], "i2": [7]}},
            [[2], [1, 0]],
            [[1], [0]],
        ),
    ],
)
def test_load_ns_groups_maps_feature_ids_preserves_declared_order(
    tmp_path: Path,
    payload: dict[str, dict[str, list[int]]],
    expected_user: list[list[int]],
    expected_item: list[list[int]],
) -> None:
    dataset = SimpleNamespace(
        user_int_schema=_schema([(10, 0, 1), (20, 1, 1), (30, 2, 1)]),
        item_int_schema=_schema([(7, 0, 1), (8, 1, 1)]),
    )

    user_groups, item_groups = load_ns_groups(
        dataset,
        {
            "ns_grouping_strategy": "explicit",
            **payload,
        },
        tmp_path,
        tmp_path,
    )

    assert user_groups == expected_user
    assert item_groups == expected_item


@pytest.mark.parametrize(
    ("payload", "missing_name"),
    [
        ({"user_ns_groups": {"u": [999]}, "item_ns_groups": {"i": [7]}}, "999"),
        ({"user_ns_groups": {"u": [10]}, "item_ns_groups": {"i": [999]}}, "999"),
    ],
)
def test_load_ns_groups_raises_for_unknown_feature_ids(
    tmp_path: Path,
    payload: dict[str, dict[str, list[int]]],
    missing_name: str,
) -> None:
    dataset = SimpleNamespace(
        user_int_schema=_schema([(10, 0, 1)]),
        item_int_schema=_schema([(7, 0, 1)]),
    )

    with pytest.raises(KeyError, match=missing_name):
        load_ns_groups(
            dataset,
            {
                "ns_grouping_strategy": "explicit",
                **payload,
            },
            tmp_path,
            tmp_path,
        )


@pytest.mark.parametrize(
    ("config", "bucket_count", "expected"),
    [
        ({"use_time_buckets": True}, 7, 7),
        ({"use_time_buckets": False}, 9, 0),
        ({"use_time_buckets": 0}, 11, 0),
    ],
)
def test_num_time_buckets_cases(config: dict[str, object], bucket_count: int, expected: int) -> None:
    data_module = SimpleNamespace(NUM_TIME_BUCKETS=bucket_count)

    assert num_time_buckets(config, data_module) == expected


def test_num_time_buckets_requires_explicit_config() -> None:
    with pytest.raises(KeyError, match="use_time_buckets"):
        num_time_buckets({}, SimpleNamespace(NUM_TIME_BUCKETS=65))


@pytest.mark.parametrize(
    ("domains", "explicit_domains"),
    [
        ([], set()),
        (["seq_a"], set()),
        (["seq_a"], {"seq_a"}),
        (["seq_a", "seq_b"], set()),
        (["seq_a", "seq_b"], {"seq_a"}),
        (["seq_a", "seq_b", "seq_c"], {"seq_a", "seq_b", "seq_c"}),
    ],
)
def test_batch_to_model_input_cases(domains: list[str], explicit_domains: set[str]) -> None:
    batch: dict[str, object] = {
        "user_int_feats": torch.ones(2, 1, dtype=torch.long),
        "item_int_feats": torch.ones(2, 1, dtype=torch.long),
        "user_dense_feats": torch.ones(2, 2),
        "item_dense_feats": torch.zeros(2, 0),
        "_seq_domains": domains,
    }
    for index, domain in enumerate(domains, start=1):
        length = index + 2
        batch[domain] = torch.ones(2, 1, length, dtype=torch.long)
        batch[f"{domain}_len"] = torch.tensor([length, max(1, length - 1)], dtype=torch.long)
        if domain in explicit_domains:
            batch[f"{domain}_time_bucket"] = torch.full((2, length), index, dtype=torch.long)

    model_input = batch_to_model_input(batch, _ModelInput, torch.device("cpu"))

    assert set(model_input.seq_data) == set(domains)
    for index, domain in enumerate(domains, start=1):
        max_length = index + 2
        assert model_input.seq_data[domain].shape == (2, 1, max_length)
        assert model_input.seq_lens[domain].shape == (2,)
        assert model_input.seq_time_buckets[domain].shape == (2, max_length)
        if domain in explicit_domains:
            assert torch.equal(model_input.seq_time_buckets[domain], torch.full((2, max_length), index, dtype=torch.long))
        else:
            assert torch.equal(model_input.seq_time_buckets[domain], torch.zeros(2, max_length, dtype=torch.long))


@pytest.mark.parametrize(
    ("config", "use_explicit_groups", "expected_user_groups", "expected_item_groups", "expected_time_buckets"),
    [
        ({"use_time_buckets": True}, False, [[0], [1]], [[0]], 13),
        ({"use_time_buckets": False}, False, [[0], [1]], [[0]], 0),
        ({"use_time_buckets": True}, True, [[1, 0]], [[0]], 13),
        ({"use_time_buckets": 0}, True, [[1, 0]], [[0]], 0),
    ],
)
def test_build_pcvr_model_forwards_constructor_contract(
    tmp_path: Path,
    config: dict[str, object],
    use_explicit_groups: bool,
    expected_user_groups: list[list[int]],
    expected_item_groups: list[list[int]],
    expected_time_buckets: int,
) -> None:
    runtime_config: dict[str, object] = {}

    def configure_rms_norm_runtime(*, rms_norm_backend: str, rms_norm_block_rows: int) -> None:
        runtime_config["backend"] = rms_norm_backend
        runtime_config["block_rows"] = rms_norm_block_rows

    dataset = SimpleNamespace(
        user_int_schema=_schema([(10, 0, 1), (20, 1, 1)]),
        item_int_schema=_schema([(7, 0, 1)]),
        user_int_vocab_sizes=[11, 17],
        item_int_vocab_sizes=[19],
        user_dense_schema=SimpleNamespace(total_dim=3),
        item_dense_schema=SimpleNamespace(total_dim=2),
        seq_domain_vocab_sizes={"seq_a": [5, 6], "seq_b": [7]},
    )
    package_dir = tmp_path / "package"
    checkpoint_dir = tmp_path / "checkpoint"
    package_dir.mkdir()
    checkpoint_dir.mkdir()
    resolved_config = {
        "d_model": "16",
        "emb_dim": "8",
        "num_queries": "2",
        "num_blocks": "3",
        "num_heads": "4",
        "seq_encoder_type": "transformer",
        "hidden_mult": "2",
        "dropout_rate": "0.0",
        "seq_top_k": "50",
        "seq_causal": False,
        "action_num": "1",
        "rank_mixer_mode": "full",
        "use_rope": False,
        "rope_base": "10000.0",
        "emb_skip_threshold": "1000000",
        "seq_id_threshold": "10000",
        "gradient_checkpointing": False,
        "rms_norm_backend": "tilelang",
        "rms_norm_block_rows": "8",
        "ns_tokenizer_type": "rankmixer",
        "user_ns_tokens": "2",
        "item_ns_tokens": "1",
        "ns_grouping_strategy": "singleton",
        "user_ns_groups": {},
        "item_ns_groups": {},
        **config,
    }
    if use_explicit_groups:
        resolved_config["ns_grouping_strategy"] = "explicit"
        resolved_config["user_ns_groups"] = {"u": [20, 10]}
        resolved_config["item_ns_groups"] = {"i": [7]}

    model = build_pcvr_model(
        model_module=SimpleNamespace(
            RecordedModel=_RecordingModel,
            configure_rms_norm_runtime=configure_rms_norm_runtime,
        ),
        model_class_name="RecordedModel",
        data_module=SimpleNamespace(NUM_TIME_BUCKETS=13),
        dataset=dataset,
        config=resolved_config,
        package_dir=package_dir,
        checkpoint_dir=checkpoint_dir,
    )

    assert isinstance(model, _RecordingModel)
    assert model.kwargs["user_int_feature_specs"] == [(11, 0, 1), (17, 1, 1)]
    assert model.kwargs["item_int_feature_specs"] == [(19, 0, 1)]
    assert model.kwargs["user_dense_dim"] == 3
    assert model.kwargs["item_dense_dim"] == 2
    assert model.kwargs["seq_vocab_sizes"] == {"seq_a": [5, 6], "seq_b": [7]}
    assert model.kwargs["user_ns_groups"] == expected_user_groups
    assert model.kwargs["item_ns_groups"] == expected_item_groups
    assert model.kwargs["num_time_buckets"] == expected_time_buckets
    assert model.kwargs["d_model"] == 16
    assert model.kwargs["emb_dim"] == 8
    assert model.kwargs["num_blocks"] == 3
    assert model.kwargs["num_heads"] == 4
    assert runtime_config == {"backend": "tilelang", "block_rows": 8}


@pytest.mark.parametrize(
    ("path_value", "expected"),
    [
        (Path("global_step0"), 0),
        (Path("global_step12.layer=2"), 12),
        (Path("global_step12.layer=2.best_model"), 12),
        (Path("global_step3.head=4") / "model.safetensors", 3),
        (Path("global_step0007"), 7),
        (Path("global_step9.hidden=64.extra_token"), 9),
        (Path("best_model"), -1),
        (Path("invalid_parent") / "model.safetensors", -1),
    ],
)
def test_checkpoint_step_cases(path_value: Path, expected: int) -> None:
    assert checkpoint_step(path_value) == expected


@pytest.mark.parametrize(
    "name",
    [
        "",
        "best",
        "globalstep1",
        "global_step",
        "global_step1 space",
        "global_step1!",
        "global_step1/child",
        "g" * 301,
    ],
)
def test_validate_checkpoint_dir_name_rejects_invalid_names(name: str) -> None:
    with pytest.raises(ValueError):
        validate_checkpoint_dir_name(name)


@pytest.mark.parametrize(
    ("global_step", "params", "is_best", "expected"),
    [
        (0, None, False, "global_step0"),
        (1, {"layer": 2}, False, "global_step1.layer=2"),
        (1, {"head": 4}, False, "global_step1.head=4"),
        (7, {"hidden": 128, "unused": 9}, False, "global_step7.hidden=128"),
        (9, {"layer": "02", "head": "04"}, True, "global_step9.layer=02.head=04.best_model"),
        (12, {"layer": 2, "head": 4, "hidden": 64}, True, "global_step12.layer=2.head=4.hidden=64.best_model"),
    ],
)
def test_build_checkpoint_dir_name_cases(
    global_step: int,
    params: dict[str, object] | None,
    is_best: bool,
    expected: str,
) -> None:
    assert build_checkpoint_dir_name(global_step, params, is_best=is_best) == expected


def test_build_checkpoint_dir_name_rejects_negative_global_step() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        build_checkpoint_dir_name(-1)


@pytest.mark.parametrize(
    "scenario",
    [
        "explicit_file",
        "explicit_dir",
        "legacy_pt_file",
        "best_model",
        "latest_step",
        "direct_checkpoint",
        "missing",
    ],
)
def test_resolve_checkpoint_path_cases(tmp_path: Path, scenario: str) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    if scenario == "explicit_file":
        candidate = tmp_path / "manual.safetensors"
        candidate.write_text("manual", encoding="utf-8")
        assert resolve_checkpoint_path(run_dir, candidate) == candidate.resolve()
        return

    if scenario == "explicit_dir":
        candidate_dir = tmp_path / "manual_dir"
        candidate_dir.mkdir()
        model_path = candidate_dir / "model.safetensors"
        model_path.write_text("manual", encoding="utf-8")
        assert resolve_checkpoint_path(run_dir, candidate_dir) == model_path.resolve()
        return

    if scenario == "legacy_pt_file":
        legacy_path = tmp_path / "manual.pt"
        legacy_path.write_text("legacy", encoding="utf-8")
        with pytest.raises(ValueError, match=r"unsupported checkpoint format"):
            resolve_checkpoint_path(run_dir, legacy_path)
        return

    if scenario == "best_model":
        older = run_dir / "global_step1.best_model"
        newer = run_dir / "global_step2.best_model"
        older.mkdir()
        newer.mkdir()
        (older / "model.safetensors").write_text("old", encoding="utf-8")
        (newer / "model.safetensors").write_text("new", encoding="utf-8")
        assert resolve_checkpoint_path(run_dir) == (newer / "model.safetensors").resolve()
        return

    if scenario == "latest_step":
        older = run_dir / "global_step1"
        newer = run_dir / "global_step3.layer=2"
        older.mkdir()
        newer.mkdir()
        (older / "model.safetensors").write_text("old", encoding="utf-8")
        (newer / "model.safetensors").write_text("new", encoding="utf-8")
        assert resolve_checkpoint_path(run_dir) == (newer / "model.safetensors").resolve()
        return

    if scenario == "direct_checkpoint":
        direct_model = run_dir / "model.safetensors"
        direct_model.write_text("direct", encoding="utf-8")
        assert resolve_checkpoint_path(run_dir) == direct_model.resolve()
        return

    with pytest.raises(FileNotFoundError, match=r"no model\.safetensors checkpoint"):
        resolve_checkpoint_path(run_dir)


@pytest.mark.parametrize(
    ("include_schema", "include_train_config", "expected_keys"),
    [
        (False, False, set()),
        (True, False, {"schema"}),
        (False, True, {"train_config"}),
        (True, True, {"schema", "train_config"}),
    ],
)
def test_write_checkpoint_sidecars_cases(
    tmp_path: Path,
    include_schema: bool,
    include_train_config: bool,
    expected_keys: set[str],
) -> None:
    checkpoint_dir = tmp_path / "global_step1.best_model"
    schema_path = tmp_path / "schema.json"
    if include_schema:
        schema_path.write_text('{"schema": true}\n', encoding="utf-8")

    train_config = (
        {
            "ns_grouping_strategy": "explicit",
            "user_ns_groups": {"u": [10, 20]},
            "item_ns_groups": {"i": [7]},
            "d_model": 64,
        }
        if include_train_config
        else None
    )
    written = write_checkpoint_sidecars(
        checkpoint_dir,
        schema_path=schema_path if include_schema else None,
        train_config=train_config,
    )

    assert set(written) == expected_keys
    if "schema" in expected_keys:
        assert (checkpoint_dir / "schema.json").exists()
    if "train_config" in expected_keys:
        payload = loads((checkpoint_dir / "train_config.json").read_bytes())
        assert payload["ns_grouping_strategy"] == "explicit"
        assert payload["user_ns_groups"] == {"u": [10, 20]}
        assert payload["item_ns_groups"] == {"i": [7]}


def test_build_pcvr_train_config_sidecar_adds_version_metadata() -> None:
    flat_config = PCVRTrainConfig().to_flat_dict()

    payload = build_pcvr_train_config_sidecar(flat_config)

    assert payload["train_config_format"] == PCVR_TRAIN_CONFIG_FORMAT
    assert payload["train_config_version"] == PCVR_TRAIN_CONFIG_VERSION
    assert payload["framework_name"] == "taac2026"
    assert payload["framework_version"]
    assert payload["d_model"] == flat_config["d_model"]
    assert payload["user_ns_groups"] == flat_config["user_ns_groups"]


@pytest.mark.parametrize("versioned", [False, True])
def test_default_load_train_config_accepts_legacy_and_versioned_payloads(tmp_path: Path, versioned: bool) -> None:
    checkpoint_dir = tmp_path / "global_step1"
    checkpoint_dir.mkdir()
    flat_config = PCVRTrainConfig().to_flat_dict()
    payload = build_pcvr_train_config_sidecar(flat_config) if versioned else flat_config
    (checkpoint_dir / "train_config.json").write_text(dumps(payload), encoding="utf-8")

    loaded = default_load_train_config(None, checkpoint_dir)

    assert loaded["d_model"] == flat_config["d_model"]
    assert loaded["ns_grouping_strategy"] == flat_config["ns_grouping_strategy"]
    if versioned:
        assert loaded["train_config_version"] == PCVR_TRAIN_CONFIG_VERSION


@pytest.mark.parametrize("identifier_kind", ["path", "path_object"])
def test_load_experiment_package_accepts_path_and_module_identifiers(
    experiment_case: ExperimentCase,
    identifier_kind: str,
) -> None:
    identifier = experiment_case.path if identifier_kind == "path" else (REPO_ROOT / experiment_case.path)
    experiment = load_experiment_package(identifier)

    assert experiment.name == experiment_case.name
    assert experiment.package_dir == (REPO_ROOT / experiment_case.path).resolve()


def test_experiment_package_contracts(loaded_experiment, experiment_case: ExperimentCase) -> None:
    train_defaults = loaded_experiment.train_defaults.to_flat_dict()

    assert loaded_experiment.name == experiment_case.name
    assert loaded_experiment.package_dir == (REPO_ROOT / experiment_case.path).resolve()
    assert loaded_experiment.train_defaults is not None
    assert loaded_experiment.metadata["kind"] == "pcvr"
    assert loaded_experiment.metadata["model_class"] == experiment_case.model_class
    assert train_defaults["ns_grouping_strategy"] == "explicit"
    assert train_defaults["user_ns_groups"]
    assert train_defaults["item_ns_groups"]
    assert "num_hyformer_blocks" not in train_defaults


def test_model_module_contracts(loaded_model_module, experiment_case: ExperimentCase) -> None:
    assert hasattr(loaded_model_module, "ModelInput")
    assert hasattr(loaded_model_module, experiment_case.model_class)
    if experiment_case.path != "experiments/pcvr/baseline":
        assert not hasattr(loaded_model_module, "PCVRHyFormer")


def test_ns_group_config_has_required_keys(experiment_case: ExperimentCase) -> None:
    experiment = load_experiment_package(experiment_case.path)
    payload = experiment.train_defaults.to_flat_dict()

    assert payload["ns_grouping_strategy"] == "explicit"
    assert isinstance(payload["user_ns_groups"], dict)
    assert isinstance(payload["item_ns_groups"], dict)
    assert all(isinstance(group, list) for group in payload["user_ns_groups"].values())
    assert all(isinstance(group, list) for group in payload["item_ns_groups"].values())
    assert all(all(isinstance(feature_id, int) for feature_id in group) for group in payload["user_ns_groups"].values())
    assert all(all(isinstance(feature_id, int) for feature_id in group) for group in payload["item_ns_groups"].values())


def test_bundle_manifest_points_to_selected_experiment(
    built_bundle: tuple[object, dict[str, object], set[str]],
    experiment_case: ExperimentCase,
) -> None:
    result, manifest, _names = built_bundle

    assert result.output_dir.is_dir()
    assert result.run_script_path.exists()
    assert result.code_package_path.exists()
    assert manifest["bundle_format"] == "taac2026-training-v2"
    assert manifest["bundled_experiment_path"] == experiment_case.path
    assert manifest["entrypoint"] == "run.sh"
    assert manifest["code_package"] == "code_package.zip"


def test_bundle_contains_model_and_explicit_ns_config(
    built_bundle: tuple[object, dict[str, object], set[str]],
    experiment_case: ExperimentCase,
) -> None:
    _result, _manifest, names = built_bundle

    assert f"project/{experiment_case.path}/__init__.py" in names
    assert f"project/{experiment_case.path}/model.py" in names
    assert f"project/{experiment_case.path}/ns_groups.json" not in names
    assert "project/src/taac2026/application/training/cli.py" in names
    assert "project/src/taac2026/infrastructure/pcvr/training.py" in names


def test_bundle_excludes_package_local_runtime_wrappers(
    built_bundle: tuple[object, dict[str, object], set[str]],
    experiment_case: ExperimentCase,
) -> None:
    _result, _manifest, names = built_bundle

    assert "project/run.sh" not in names
    assert f"project/{experiment_case.path}/run.sh" not in names
    assert f"project/{experiment_case.path}/train.py" not in names
    assert f"project/{experiment_case.path}/trainer.py" not in names
