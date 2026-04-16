from __future__ import annotations

from pathlib import Path

import pytest


UNIT_TEST_FILES = {
    "test_clean_pycache.py",
    "test_dataset_eda.py",
    "test_experiment_packages.py",
    "test_metrics.py",
    "test_model_performance_plot.py",
    "test_model_robustness.py",
    "test_property_based.py",
    "test_payload.py",
    "test_profiling_unit.py",
    "test_runtime_optimization.py",
    "test_schema_contract.py",
    "test_search_trial.py",
    "test_search_worker.py",
    "test_tech_timeline.py",
    "test_test_collection.py",
}

GPU_TEST_FILES = {
    "test_gpu.py",
}

INTEGRATION_TEST_FILES = {
    "test_data_pipeline.py",
    "test_evaluate_cli.py",
    "test_profiling.py",
    "test_runtime_integration.py",
    "test_search.py",
    "test_search_worker_integration.py",
    "test_training_recovery.py",
}


def _build_test_file_classification() -> dict[str, str]:
    overlapping_files = UNIT_TEST_FILES & INTEGRATION_TEST_FILES
    if overlapping_files:
        overlap = ", ".join(sorted(overlapping_files))
        raise pytest.UsageError(
            f"Test files cannot be classified as both unit and integration: {overlap}"
        )

    classification = {filename: "unit" for filename in UNIT_TEST_FILES}
    classification.update({filename: "integration" for filename in INTEGRATION_TEST_FILES})
    classification.update({filename: "gpu" for filename in GPU_TEST_FILES})
    return classification


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    del config
    classification = _build_test_file_classification()
    unclassified_files: set[str] = set()
    for item in items:
        filename = Path(str(item.fspath)).name
        marker = classification.get(filename)
        if marker == "unit":
            item.add_marker(pytest.mark.unit)
        elif marker == "integration":
            item.add_marker(pytest.mark.integration)
        elif marker == "gpu":
            item.add_marker(pytest.mark.gpu)
        else:
            unclassified_files.add(filename)

    if unclassified_files:
        missing = ", ".join(sorted(unclassified_files))
        raise pytest.UsageError(
            "Collected test files are not classified in UNIT_TEST_FILES or "
            f"INTEGRATION_TEST_FILES: {missing}"
        )
