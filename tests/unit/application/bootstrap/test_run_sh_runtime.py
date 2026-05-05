from __future__ import annotations

from pathlib import Path

import pytest

from taac2026.infrastructure.platform import deps as bundle_runtime
from taac2026.infrastructure.platform.env import (
    DOCKER_GPU_PLATFORM,
    ONLINE_TRAINING_BUNDLE_PLATFORM,
    resolve_run_sh_platform,
    select_run_sh_platform,
)
from taac2026.application.bootstrap.run_sh import extract_cuda_profile, parse_run_command


def test_parse_run_command_defaults_to_train_for_flags() -> None:
    parsed = parse_run_command(["--device", "cpu"])

    assert parsed.command == "train"
    assert parsed.args == ["--device", "cpu"]


def test_parse_run_command_rejects_legacy_test_subcommand() -> None:
    with pytest.raises(ValueError, match=r"run\.sh no longer supports 'test'"):
        parse_run_command(["test", "tests/unit"])


def test_extract_cuda_profile_removes_supported_profile(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TAAC_CUDA_PROFILE", raising=False)

    parsed = extract_cuda_profile(["--cuda-profile", "cuda126", "--device", "cpu"])

    assert parsed.profile == "cuda126"
    assert parsed.remaining_args == ["--device", "cpu"]


def test_bundle_pip_install_uses_bundle_extras_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[list[str], Path]] = []
    monkeypatch.setenv("TAAC_PIP_EXTRAS", "dev")
    monkeypatch.setenv("TAAC_PIP_EXTRA_ARGS", "-q")
    monkeypatch.setenv("TAAC_PIP_INDEX_URL", "")
    monkeypatch.delenv("TAAC_BUNDLE_PIP_EXTRAS", raising=False)
    monkeypatch.setattr(bundle_runtime.subprocess, "check_call", lambda command, cwd: calls.append((command, cwd)))

    bundle_runtime.install_project_pip_dependencies(tmp_path, ONLINE_TRAINING_BUNDLE_PLATFORM)

    assert calls == [([bundle_runtime.resolve_python(), "-m", "pip", "install", "--disable-pip-version-check", "-q", "."], tmp_path)]


def test_select_run_sh_platform_uses_online_training_adapter_for_bundles() -> None:
    platform = select_run_sh_platform(bundle_mode=True)

    assert platform is ONLINE_TRAINING_BUNDLE_PLATFORM
    assert platform.default_runner == "python"
    assert platform.pip_extras_env == "TAAC_BUNDLE_PIP_EXTRAS"


def test_resolve_run_sh_platform_uses_registered_names() -> None:
    assert resolve_run_sh_platform(DOCKER_GPU_PLATFORM.name) is DOCKER_GPU_PLATFORM