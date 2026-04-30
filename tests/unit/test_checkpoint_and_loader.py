from __future__ import annotations

from pathlib import Path

import pytest
import torch

from taac2026.infrastructure.checkpoints import (
    PRIMARY_CHECKPOINT_FILENAME,
    build_checkpoint_dir_name,
    load_checkpoint_state_dict,
    resolve_checkpoint_path,
    save_checkpoint_state_dict,
    validate_checkpoint_dir_name,
    write_checkpoint_sidecars,
)
from taac2026.infrastructure.experiments.loader import load_experiment_package


def test_resolve_checkpoint_prefers_best_model(tmp_path: Path) -> None:
    old_dir = tmp_path / "global_step1.layer=2"
    best_dir = tmp_path / "global_step2.layer=2.best_model"
    old_dir.mkdir()
    best_dir.mkdir()
    (old_dir / PRIMARY_CHECKPOINT_FILENAME).write_text("old", encoding="utf-8")
    (best_dir / PRIMARY_CHECKPOINT_FILENAME).write_text("best", encoding="utf-8")

    assert resolve_checkpoint_path(tmp_path) == best_dir / PRIMARY_CHECKPOINT_FILENAME


def test_resolve_checkpoint_rejects_legacy_pt_files(tmp_path: Path) -> None:
    legacy_path = tmp_path / "global_step2.layer=2.best_model" / "legacy.pt"
    legacy_path.parent.mkdir()
    legacy_path.write_text("legacy", encoding="utf-8")

    with pytest.raises(ValueError, match=r"unsupported checkpoint format"):
        resolve_checkpoint_path(tmp_path, legacy_path)


def test_validate_checkpoint_name_rejects_non_global_step_prefix() -> None:
    with pytest.raises(ValueError, match="global_step"):
        validate_checkpoint_dir_name("best")


def test_build_checkpoint_dir_name_uses_global_step_prefix() -> None:
    assert build_checkpoint_dir_name(12, {"layer": 2, "head": 4, "hidden": 64}, is_best=True) == "global_step12.layer=2.head=4.hidden=64.best_model"


def test_write_checkpoint_sidecars_rewrites_ns_groups_path(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "global_step1.best_model"
    schema_path = tmp_path / "schema.json"
    ns_groups_path = tmp_path / "ns_groups.json"
    schema_path.write_text('{"schema": true}\n', encoding="utf-8")
    ns_groups_path.write_text('{"groups": true}\n', encoding="utf-8")

    written = write_checkpoint_sidecars(
        checkpoint_dir,
        schema_path=schema_path,
        ns_groups_path=ns_groups_path,
        train_config={"ns_groups_json": str(ns_groups_path), "d_model": 64},
    )

    assert set(written) == {"schema", "ns_groups", "train_config"}
    assert (checkpoint_dir / "schema.json").exists()
    assert '"ns_groups_json": "ns_groups.json"' in (checkpoint_dir / "train_config.json").read_text(encoding="utf-8")


def test_save_and_load_checkpoint_state_dict_round_trip(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "global_step3.best_model"
    state_dict = {
        "weight": torch.arange(6, dtype=torch.float32).reshape(2, 3),
        "bias": torch.tensor([1.5], dtype=torch.float32),
    }

    checkpoint_path = save_checkpoint_state_dict(state_dict, checkpoint_dir)

    assert checkpoint_path == checkpoint_dir / PRIMARY_CHECKPOINT_FILENAME
    assert not any(checkpoint_dir.glob("*.pt"))

    loaded_state_dict = load_checkpoint_state_dict(checkpoint_path, map_location="cpu")
    assert torch.equal(loaded_state_dict["weight"], state_dict["weight"])
    assert torch.equal(loaded_state_dict["bias"], state_dict["bias"])


def test_save_checkpoint_rejects_legacy_pt_path(tmp_path: Path) -> None:
    state_dict = {"weight": torch.arange(2, dtype=torch.float32)}

    with pytest.raises(ValueError, match=r"unsupported checkpoint format"):
        save_checkpoint_state_dict(state_dict, tmp_path / "legacy.pt")


def test_load_baseline_experiment_from_path() -> None:
    experiment = load_experiment_package("config/baseline")

    assert experiment.name == "pcvr_hyformer"
    assert experiment.package_dir is not None
