from __future__ import annotations

from pathlib import Path

import pytest


UNIT_TEST_FILES = {
    "test_clean_pycache.py",
    "test_experiment_packages.py",
    "test_metrics.py",
    "test_model_performance_plot.py",
    "test_payload.py",
    "test_profiling_unit.py",
    "test_runtime_optimization.py",
    "test_search_trial.py",
    "test_search_worker.py",
}

INTEGRATION_TEST_FILES = {
    "test_data_pipeline.py",
    "test_evaluate_cli.py",
    "test_profiling.py",
    "test_runtime_integration.py",
    "test_search.py",
    "test_search_worker_integration.py",
}


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    del config
    for item in items:
        filename = Path(str(item.fspath)).name
        if filename in UNIT_TEST_FILES:
            item.add_marker(pytest.mark.unit)
        if filename in INTEGRATION_TEST_FILES:
            item.add_marker(pytest.mark.integration)
