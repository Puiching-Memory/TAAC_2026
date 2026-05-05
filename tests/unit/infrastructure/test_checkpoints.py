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


def test_write_checkpoint_sidecars_persists_explicit_ns_group_config(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "global_step1.best_model"
    schema_path = tmp_path / "schema.json"
    schema_path.write_text('{"schema": true}\n', encoding="utf-8")

    written = write_checkpoint_sidecars(
        checkpoint_dir,
        schema_path=schema_path,
        train_config={
            "ns_grouping_strategy": "explicit",
            "user_ns_groups": {"u": [10, 20]},
            "item_ns_groups": {"i": [7]},
            "d_model": 64,
        },
    )

    assert set(written) == {"schema", "train_config"}
    assert (checkpoint_dir / "schema.json").exists()
    payload = (checkpoint_dir / "train_config.json").read_text(encoding="utf-8")
    assert '"ns_grouping_strategy": "explicit"' in payload
    assert '"user_ns_groups": {' in payload
    assert '"item_ns_groups": {' in payload


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

