from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import json
from json import JSONDecodeError
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

from ...domain.experiment import ExperimentSpec
from ...infrastructure.compute.device_scheduler import launchable_devices, query_gpu_devices
from ...infrastructure.experiments.payload import serialize_experiment
from ...infrastructure.io.console import create_progress_bar, logger
from ...infrastructure.io.files import ensure_dir, write_json
from .space import build_default_search_experiment
from .trial import execute_search_trial


def _require_optuna():
    try:
        import optuna
    except ModuleNotFoundError as exc:
        raise RuntimeError("optuna is required for taac-search; run uv sync --locked") from exc
    return optuna


@dataclass(slots=True)
class SearchWorkerProcess:
    trial: Any
    process: subprocess.Popen[str]
    result_path: Path
    physical_gpu_index: int | None


def _default_study_dir(experiment: ExperimentSpec) -> Path:
    base_output_dir = Path(experiment.train.output_dir)
    return base_output_dir.parent / f"{base_output_dir.name}_optuna"


def _trial_state_counts(study) -> dict[str, int]:
    return dict(sorted(Counter(trial.state.name for trial in study.trials).items()))


def _update_progress_bar(progress_bar, study, *, last_state: str, running_count: int = 0) -> None:
    progress_bar.update(1)
    trial_state_counts = _trial_state_counts(study)
    postfix: dict[str, str | int] = {
        "complete": trial_state_counts.get("COMPLETE", 0),
        "pruned": trial_state_counts.get("PRUNED", 0),
        "last": last_state.lower(),
    }
    if running_count > 0:
        postfix["running"] = running_count
    completed_trials = [trial for trial in study.trials if trial.state.name == "COMPLETE"]
    if completed_trials:
        postfix["best"] = f"{study.best_value:.4f}"
    progress_bar.set_postfix(postfix, refresh=False)


def _make_progress_callback(progress_bar):
    def callback(study, frozen_trial) -> None:
        _update_progress_bar(progress_bar, study, last_state=frozen_trial.state.name)

    return callback


def _apply_trial_result_user_attrs(trial, result: dict[str, Any]) -> None:
    if result.get("budget_probe") is not None:
        trial.set_user_attr("budget_probe", result["budget_probe"])
    if result.get("summary_path") is not None:
        trial.set_user_attr("summary_path", result["summary_path"])
    if result.get("final_budget_status") is not None:
        trial.set_user_attr("final_budget_status", result["final_budget_status"])
    if result.get("objective_value") is not None:
        trial.set_user_attr("objective_value", float(result["objective_value"]))
    if result.get("prune_reason"):
        trial.set_user_attr("prune_reason", str(result["prune_reason"]))
    if result.get("trial_error"):
        trial.set_user_attr("trial_error", str(result["trial_error"]))


def _finalize_parallel_trial(study, trial, result: dict[str, Any], optuna) -> str:
    _apply_trial_result_user_attrs(trial, result)
    status = str(result.get("status", "fail")).lower()
    if status == "complete":
        objective_value = result.get("objective_value")
        if objective_value is None:
            trial.set_user_attr("trial_error", "worker completed without an objective value")
            study.tell(trial, state=optuna.trial.TrialState.FAIL)
            return "FAIL"
        study.tell(trial, float(objective_value), state=optuna.trial.TrialState.COMPLETE)
        return "COMPLETE"
    if status == "pruned":
        study.tell(trial, state=optuna.trial.TrialState.PRUNED)
        return "PRUNED"
    study.tell(trial, state=optuna.trial.TrialState.FAIL)
    return "FAIL"


def _normalize_experiment_path(experiment_path: str | Path) -> str:
    path = Path(experiment_path)
    if path.exists():
        return str(path.resolve())
    return str(experiment_path)


def _launch_search_worker(
    experiment_path: str | Path,
    serialized_experiment: dict[str, Any],
    trial_dir: Path,
    physical_gpu_index: int | None,
) -> SearchWorkerProcess:
    config_path = trial_dir / "worker_experiment.json"
    result_path = trial_dir / "worker_result.json"
    write_json(config_path, serialized_experiment)

    command = [
        sys.executable,
        "-m",
        "taac2026.application.search.worker",
        "--experiment",
        _normalize_experiment_path(experiment_path),
        "--config-path",
        str(config_path),
        "--result-path",
        str(result_path),
    ]
    env = dict(os.environ)
    if physical_gpu_index is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(physical_gpu_index)
        command.extend(["--device", "cuda:0"])

    process = subprocess.Popen(
        command,
        cwd=str(Path.cwd()),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return SearchWorkerProcess(
        trial=None,
        process=process,
        result_path=result_path,
        physical_gpu_index=physical_gpu_index,
    )


def _collect_worker_result(worker: SearchWorkerProcess) -> dict[str, Any]:
    stdout, stderr = worker.process.communicate()
    result: dict[str, Any] | None = None
    if worker.result_path.exists():
        try:
            with worker.result_path.open("r", encoding="utf-8") as handle:
                result = json.load(handle)
        except (OSError, JSONDecodeError) as exc:
            message = stderr.strip() or stdout.strip() or str(exc)
            return {"status": "fail", "trial_error": f"invalid worker result payload: {message}"}

    if result is None:
        message = stderr.strip() or stdout.strip() or f"worker exited with code {worker.process.returncode}"
        return {"status": "fail", "trial_error": message}

    if result.get("status") == "fail" and not result.get("trial_error"):
        result["trial_error"] = stderr.strip() or stdout.strip() or f"worker exited with code {worker.process.returncode}"
    return result


def _build_search_report(
    study,
    experiment: ExperimentSpec,
    *,
    experiment_path: str | Path | None,
    study_root: Path,
    scheduler_info: dict[str, Any] | None,
) -> dict[str, Any]:
    completed_trials = [trial for trial in study.trials if trial.state.name == "COMPLETE"]
    study_summary_path = study_root / "study_summary.json"
    best_experiment_path = study_root / "best_experiment.json"
    best_trial_payload: dict[str, Any] | None = None
    if completed_trials:
        best_trial = study.best_trial
        best_trial_payload = {
            "number": best_trial.number,
            "value": best_trial.value,
            "params": dict(best_trial.params),
            "trial_dir": best_trial.user_attrs.get("trial_dir"),
            "summary_path": best_trial.user_attrs.get("summary_path"),
            "budget_probe": best_trial.user_attrs.get("budget_probe"),
            "final_budget_status": best_trial.user_attrs.get("final_budget_status"),
            "experiment": best_trial.user_attrs.get("experiment"),
        }
        if best_trial_payload["experiment"] is not None:
            write_json(best_experiment_path, best_trial_payload["experiment"])

    report = {
        "experiment_name": experiment.name,
        "experiment_path": str(experiment_path) if experiment_path is not None else None,
        "study_dir": str(study_root),
        "study_summary_path": str(study_summary_path),
        "best_experiment_path": str(best_experiment_path) if best_experiment_path.exists() or best_trial_payload is not None else None,
        "search": asdict(experiment.search),
        "scheduler": scheduler_info,
        "trial_state_counts": _trial_state_counts(study),
        "trial_count": len(study.trials),
        "best_trial": best_trial_payload,
        "trials": [
            {
                "number": trial.number,
                "state": trial.state.name,
                "value": trial.value,
                "params": dict(trial.params),
                "user_attrs": dict(trial.user_attrs),
            }
            for trial in study.trials
        ],
    }
    write_json(study_summary_path, report)
    return report


def _run_search_sequential(
    *,
    study,
    experiment: ExperimentSpec,
    experiment_path: str | Path | None,
    study_root: Path,
    show_progress: bool,
    scheduler_info: dict[str, Any] | None,
) -> dict[str, Any]:
    optuna = _require_optuna()
    build_search_experiment = experiment.build_search_experiment or build_default_search_experiment

    def objective(trial) -> float:
        trial_dir = ensure_dir(study_root / f"trial_{trial.number:04d}")
        try:
            trial_experiment = build_search_experiment(experiment, trial)
        except (RuntimeError, ValueError) as exc:
            trial.set_user_attr("prune_reason", str(exc))
            trial.set_user_attr("trial_error", str(exc))
            raise optuna.TrialPruned(str(exc)) from exc

        trial_experiment.train.output_dir = str(trial_dir)
        trial.set_user_attr("experiment", serialize_experiment(trial_experiment))
        trial.set_user_attr("trial_dir", str(trial_dir))

        result = execute_search_trial(trial_experiment)
        _apply_trial_result_user_attrs(trial, result)
        if result["status"] == "pruned":
            reason = result.get("prune_reason") or "trial pruned"
            raise optuna.TrialPruned(str(reason))
        if result["status"] != "complete":
            raise RuntimeError(result.get("trial_error") or "trial failed")
        return float(result["objective_value"])

    callbacks = []
    progress_bar = None
    if show_progress:
        progress_bar = create_progress_bar(
            total=experiment.search.n_trials,
            description=f"taac-search[{experiment.name}]",
        )
        callbacks.append(_make_progress_callback(progress_bar))

    try:
        study.optimize(
            objective,
            n_trials=experiment.search.n_trials,
            timeout=experiment.search.timeout_seconds,
            callbacks=callbacks,
        )
    finally:
        if progress_bar is not None:
            progress_bar.close()

    return _build_search_report(
        study,
        experiment,
        experiment_path=experiment_path,
        study_root=study_root,
        scheduler_info=scheduler_info,
    )


def _run_search_auto(
    *,
    study,
    experiment: ExperimentSpec,
    experiment_path: str | Path,
    study_root: Path,
    show_progress: bool,
    gpu_indices: set[int] | None,
    min_free_memory_gb: float,
    max_jobs_per_gpu: int,
    poll_interval_seconds: float,
    scheduler_info: dict[str, Any],
) -> dict[str, Any]:
    optuna = _require_optuna()
    build_search_experiment = experiment.build_search_experiment or build_default_search_experiment
    min_free_memory_mb = int(min_free_memory_gb * 1024.0)
    launched_trials = 0
    active_workers: dict[int, SearchWorkerProcess] = {}
    progress_bar = None
    start_time = time.monotonic()

    if show_progress:
        progress_bar = create_progress_bar(
            total=experiment.search.n_trials,
            description=f"taac-search[{experiment.name}]",
        )

    try:
        while launched_trials < experiment.search.n_trials or active_workers:
            timeout_reached = (
                experiment.search.timeout_seconds is not None
                and (time.monotonic() - start_time) >= float(experiment.search.timeout_seconds)
            )

            if not timeout_reached and launched_trials < experiment.search.n_trials:
                running_jobs_by_gpu = Counter(
                    worker.physical_gpu_index
                    for worker in active_workers.values()
                    if worker.physical_gpu_index is not None
                )
                launch_slots = launchable_devices(
                    query_gpu_devices(gpu_indices),
                    running_jobs_by_gpu,
                    min_free_memory_mb=min_free_memory_mb,
                    max_jobs_per_gpu=max_jobs_per_gpu,
                )
                for device in launch_slots:
                    if launched_trials >= experiment.search.n_trials:
                        break
                    trial = study.ask()
                    launched_trials += 1
                    try:
                        trial_experiment = build_search_experiment(experiment, trial)
                    except (RuntimeError, ValueError) as exc:
                        trial.set_user_attr("prune_reason", str(exc))
                        trial.set_user_attr("trial_error", str(exc))
                        study.tell(trial, state=optuna.trial.TrialState.PRUNED)
                        if progress_bar is not None:
                            _update_progress_bar(progress_bar, study, last_state="PRUNED", running_count=len(active_workers))
                        continue

                    trial_dir = ensure_dir(study_root / f"trial_{trial.number:04d}")
                    trial_experiment.train.output_dir = str(trial_dir)
                    serialized_experiment = serialize_experiment(trial_experiment)
                    trial.set_user_attr("experiment", serialized_experiment)
                    trial.set_user_attr("trial_dir", str(trial_dir))
                    trial.set_user_attr("assigned_gpu_index", device.index)
                    logger.info("launch trial={} gpu={} dir={}", trial.number, device.index, trial_dir)

                    worker = _launch_search_worker(
                        experiment_path=experiment_path,
                        serialized_experiment=serialized_experiment,
                        trial_dir=trial_dir,
                        physical_gpu_index=device.index,
                    )
                    worker.trial = trial
                    active_workers[trial.number] = worker

            finished_trial_numbers: list[int] = []
            for trial_number, worker in active_workers.items():
                if worker.process.poll() is None:
                    continue
                result = _collect_worker_result(worker)
                final_state = _finalize_parallel_trial(study, worker.trial, result, optuna)
                logger.info("trial {} finished with state={}", worker.trial.number, final_state)
                finished_trial_numbers.append(trial_number)

            for trial_number in finished_trial_numbers:
                del active_workers[trial_number]
                if progress_bar is not None:
                    frozen_trial = study.trials[trial_number]
                    _update_progress_bar(
                        progress_bar,
                        study,
                        last_state=frozen_trial.state.name,
                        running_count=len(active_workers),
                    )

            if active_workers or (launched_trials < experiment.search.n_trials and not timeout_reached):
                if not finished_trial_numbers:
                    time.sleep(poll_interval_seconds)
            else:
                break
    finally:
        if progress_bar is not None:
            progress_bar.close()

    return _build_search_report(
        study,
        experiment,
        experiment_path=experiment_path,
        study_root=study_root,
        scheduler_info=scheduler_info,
    )


def run_search(
    experiment: ExperimentSpec,
    *,
    experiment_path: str | Path | None = None,
    study_dir: str | Path | None = None,
    show_progress: bool = False,
    scheduler: str = "sequential",
    gpu_indices: set[int] | None = None,
    min_free_memory_gb: float = 12.0,
    max_jobs_per_gpu: int = 4,
    poll_interval_seconds: float = 5.0,
) -> dict[str, Any]:
    if max_jobs_per_gpu <= 0:
        raise ValueError("max_jobs_per_gpu must be positive")
    if min_free_memory_gb < 0.0:
        raise ValueError("min_free_memory_gb must be non-negative")
    if poll_interval_seconds <= 0.0:
        raise ValueError("poll_interval_seconds must be positive")

    optuna = _require_optuna()
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study_root = ensure_dir(study_dir or _default_study_dir(experiment))
    sampler_seed = experiment.search.sampler_seed if experiment.search.sampler_seed is not None else experiment.train.seed
    sampler = optuna.samplers.TPESampler(seed=sampler_seed)
    study = optuna.create_study(direction=experiment.search.direction, sampler=sampler)

    scheduler_info: dict[str, Any] = {
        "requested_mode": scheduler,
        "used_mode": "sequential",
        "gpu_indices": sorted(gpu_indices) if gpu_indices is not None else None,
        "min_free_memory_gb": float(min_free_memory_gb),
        "max_jobs_per_gpu": int(max_jobs_per_gpu),
        "poll_interval_seconds": float(poll_interval_seconds),
        "fallback_reason": None,
    }

    if scheduler == "auto":
        if experiment_path is None:
            scheduler_info["fallback_reason"] = "auto scheduling requires experiment_path; using sequential"
            logger.warning("auto scheduling requested without experiment_path; falling back to sequential")
        else:
            visible_devices = query_gpu_devices(gpu_indices)
            if visible_devices:
                scheduler_info["used_mode"] = "auto"
                if scheduler_info["gpu_indices"] is None:
                    scheduler_info["gpu_indices"] = [device.index for device in visible_devices]
                logger.info(
                    "search start: experiment={} mode=auto gpus={} trials={} min_free_memory_gb={} max_jobs_per_gpu={}",
                    experiment.name,
                    scheduler_info["gpu_indices"],
                    experiment.search.n_trials,
                    min_free_memory_gb,
                    max_jobs_per_gpu,
                )
                return _run_search_auto(
                    study=study,
                    experiment=experiment,
                    experiment_path=experiment_path,
                    study_root=study_root,
                    show_progress=show_progress,
                    gpu_indices=gpu_indices,
                    min_free_memory_gb=min_free_memory_gb,
                    max_jobs_per_gpu=max_jobs_per_gpu,
                    poll_interval_seconds=poll_interval_seconds,
                    scheduler_info=scheduler_info,
                )
            scheduler_info["fallback_reason"] = "no visible GPUs detected for auto scheduling; using sequential"
            logger.warning("no visible GPUs detected for auto scheduling; falling back to sequential")

    logger.info(
        "search start: experiment={} mode=sequential trials={}",
        experiment.name,
        experiment.search.n_trials,
    )
    return _run_search_sequential(
        study=study,
        experiment=experiment,
        experiment_path=experiment_path,
        study_root=study_root,
        show_progress=show_progress,
        scheduler_info=scheduler_info,
    )


__all__ = ["run_search"]
