from __future__ import annotations

from pathlib import Path

from taac2026.infrastructure.experiments.discovery import discover_experiment_paths
from taac2026.infrastructure.io.json_utils import dumps
from tests.unit._pcvr_experiment_matrix import build_pcvr_experiment_cases


def _write_minimal_pcvr_experiment(package_dir: Path, *, experiment_name: str, model_class_name: str) -> None:
    package_dir.mkdir(parents=True)
    (package_dir / "__init__.py").write_text(
        "from pathlib import Path\n"
        "\n"
        "from taac2026.infrastructure.pcvr.config import PCVRNSConfig, PCVRTrainConfig\n"
        "from taac2026.infrastructure.pcvr.experiment import PCVRExperiment\n"
        "\n"
        f"EXPERIMENT = PCVRExperiment(name={experiment_name!r}, package_dir=Path(__file__).resolve().parent, model_class_name={model_class_name!r}, train_defaults=PCVRTrainConfig(ns=PCVRNSConfig(groups_json=\"ns_groups.json\")))\n",
        encoding="utf-8",
    )
    (package_dir / "model.py").write_text(
        "from taac2026.infrastructure.pcvr.modeling import ModelInput\n"
        "\n"
        f"class {model_class_name}:\n"
        "    pass\n"
        "\n"
        f"__all__ = [\"ModelInput\", \"{model_class_name}\"]\n",
        encoding="utf-8",
    )
    (package_dir / "ns_groups.json").write_text(
        dumps({"user_ns_groups": {"U1": [0]}, "item_ns_groups": {"I1": [0]}}, indent=2, trailing_newline=True),
        encoding="utf-8",
    )


def test_discover_experiment_paths_filters_to_valid_packages(tmp_path: Path) -> None:
    config_root = tmp_path / "config"
    config_root.mkdir()

    valid = config_root / "valid_exp"
    valid.mkdir()
    for name in ("__init__.py", "model.py", "ns_groups.json"):
        (valid / name).write_text("", encoding="utf-8")

    missing_model = config_root / "missing_model"
    missing_model.mkdir()
    (missing_model / "__init__.py").write_text("", encoding="utf-8")
    (missing_model / "ns_groups.json").write_text("{}", encoding="utf-8")

    hidden = config_root / "__pycache__"
    hidden.mkdir()
    for name in ("__init__.py", "model.py", "ns_groups.json"):
        (hidden / name).write_text("", encoding="utf-8")

    assert discover_experiment_paths(config_root) == ["config/valid_exp"]


def test_build_pcvr_experiment_cases_discovers_minimal_new_package(tmp_path: Path) -> None:
    config_root = tmp_path / "config"
    _write_minimal_pcvr_experiment(
        config_root / "minimal_exp",
        experiment_name="pcvr_minimal_exp",
        model_class_name="PCVRMinimalExp",
    )

    cases = build_pcvr_experiment_cases(config_root)

    assert len(cases) == 1
    assert cases[0].path == "config/minimal_exp"
    assert cases[0].module == "config.minimal_exp"
    assert cases[0].name == "pcvr_minimal_exp"
    assert cases[0].model_class == "PCVRMinimalExp"
    assert cases[0].package_dir == (config_root / "minimal_exp").resolve()