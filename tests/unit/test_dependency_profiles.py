from __future__ import annotations

from pathlib import Path
import re

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback
    import tomli as tomllib


def test_python_floor_and_cuda_profiles_are_declared() -> None:
    pyproject_path = Path(__file__).resolve().parents[2] / "pyproject.toml"
    config = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))

    assert config["project"]["requires-python"] == ">=3.10,<3.14"
    assert config["tool"]["ruff"]["target-version"] == "py310"

    dependencies = config["project"]["dependencies"]
    optional_dependencies = config["project"]["optional-dependencies"]

    assert all(not dependency.startswith("torch>=") for dependency in dependencies)
    assert all(not dependency.startswith("torchao>=") for dependency in dependencies)
    assert all("torchrec" not in dependency for dependency in dependencies)
    assert all("fbgemm-gpu" not in dependency for dependency in dependencies)
    assert any(dependency.startswith("tomli>=") for dependency in dependencies)
    assert any(dependency.startswith("triton>=") for dependency in dependencies)

    cpu_dependencies = optional_dependencies["cpu"]
    assert any(dependency.startswith("torch>=") for dependency in cpu_dependencies)
    assert any(dependency.startswith("torchao>=") for dependency in cpu_dependencies)
    assert all("torchrec" not in dependency for dependency in cpu_dependencies)
    assert all("fbgemm-gpu" not in dependency for dependency in cpu_dependencies)

    for profile_name in ("cuda126", "cuda128", "cuda130"):
        profile_dependencies = optional_dependencies[profile_name]
        assert any(dependency.startswith("torch>=") for dependency in profile_dependencies)
        assert any(dependency.startswith("torchao>=") for dependency in profile_dependencies)
        assert any(dependency.startswith("torchrec>=") for dependency in profile_dependencies)
        assert any(dependency.startswith("fbgemm-gpu>=") for dependency in profile_dependencies)


def test_cuda_sources_are_scoped_to_matching_extras() -> None:
    pyproject_path = Path(__file__).resolve().parents[2] / "pyproject.toml"
    config = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))

    uv_config = config["tool"]["uv"]
    sources = uv_config["sources"]
    conflicts = uv_config["conflicts"]
    expected_entries = [
        {
            "index": "pytorch-cpu",
            "marker": "platform_system == 'Linux'",
            "extra": "cpu",
        },
        {
            "index": "pytorch-cu126",
            "marker": "platform_system == 'Linux'",
            "extra": "cuda126",
        },
        {
            "index": "pytorch-cu128",
            "marker": "platform_system == 'Linux'",
            "extra": "cuda128",
        },
        {
            "index": "pytorch-cu130",
            "marker": "platform_system == 'Linux'",
            "extra": "cuda130",
        },
    ]
    expected_cuda_entries = expected_entries[1:]
    expected_conflicts = [
        [{"extra": "cpu"}, {"extra": "cuda126"}],
        [{"extra": "cpu"}, {"extra": "cuda128"}],
        [{"extra": "cpu"}, {"extra": "cuda130"}],
        [{"extra": "cuda126"}, {"extra": "cuda128"}],
        [{"extra": "cuda126"}, {"extra": "cuda130"}],
        [{"extra": "cuda128"}, {"extra": "cuda130"}],
    ]

    for package_name in ("torch", "torchao"):
        assert sources[package_name] == expected_entries
    assert sources["fbgemm-gpu"] == expected_cuda_entries
    assert uv_config["environments"] == ["sys_platform == 'linux'"]
    assert conflicts == expected_conflicts


def test_ci_workflow_checks_supported_python_versions() -> None:
    ci_workflow_path = Path(__file__).resolve().parents[2] / ".github" / "workflows" / "ci.yml"
    ci_workflow = ci_workflow_path.read_text(encoding="utf-8")
    matrix_block = 'matrix:\n        python-version: ["3.10", "3.11", "3.12", "3.13"]'

    assert ci_workflow.count(matrix_block) == 3
    assert 'CI_CANONICAL_PYTHON_VERSION: "3.13"' in ci_workflow
    assert "Style & Policy (Python ${{ matrix.python-version }})" in ci_workflow
    assert "CPU Unit Tests (Python ${{ matrix.python-version }})" in ci_workflow
    assert "CPU Benchmarks (Python ${{ matrix.python-version }})" in ci_workflow
    assert "cpu-benchmark-results-py${{ matrix.python-version }}" in ci_workflow
    assert ci_workflow.count("coverage-cpu-data-py${{ env.CI_CANONICAL_PYTHON_VERSION }}") == 2
    assert "matrix.python-version == env.CI_CANONICAL_PYTHON_VERSION" in ci_workflow
    assert "python-version: ${{ env.CI_CANONICAL_PYTHON_VERSION }}" in ci_workflow
    assert "compatibility checks run on Python 3.10, 3.11, 3.12, and 3.13" in ci_workflow
    assert re.search(r"canonical Python \$\{CI_CANONICAL_PYTHON_VERSION\} unit lane", ci_workflow) is not None