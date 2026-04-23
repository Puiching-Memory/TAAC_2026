from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch


if TYPE_CHECKING:
    from tests.support import TestWorkspace


@pytest.fixture
def benchmark_device() -> torch.device:
    return torch.device("cpu")


@pytest.fixture
def benchmark_workspace(tmp_path_factory: pytest.TempPathFactory) -> TestWorkspace:
    # Keep TorchRec-backed workspace support out of the shared benchmark import path.
    from tests.support import create_test_workspace

    return create_test_workspace(tmp_path_factory.mktemp("bench_workspace"))