from __future__ import annotations

from taac2026.application.benchmarking.pcvr_optimizer_benchmark import parse_args, run_benchmark


def test_parse_pcvr_optimizer_benchmark_args_accepts_optimizer_subset() -> None:
    args = parse_args(
        [
            "--steps",
            "2",
            "--warmup-steps",
            "1",
            "--repeats",
            "3",
            "--optimizers",
            "adamw,fused_adamw,muon",
        ]
    )

    assert args.steps == 2
    assert args.warmup_steps == 1
    assert args.repeats == 3
    assert args.optimizers == ("adamw", "fused_adamw", "muon")


def test_run_pcvr_optimizer_benchmark_returns_results_for_requested_optimizers() -> None:
    args = parse_args(
        [
            "--device",
            "cpu",
            "--batch-size",
            "8",
            "--feature-dim",
            "16",
            "--hidden-dim",
            "16",
            "--depth",
            "2",
            "--steps",
            "2",
            "--warmup-steps",
            "0",
            "--repeats",
            "3",
            "--optimizers",
            "adamw,fused_adamw,orthogonal_adamw,muon",
        ]
    )

    summary = run_benchmark(args)

    assert summary["device"] == "cpu"
    assert summary["repeats"] == 3
    assert summary["optimizers"] == ["adamw", "fused_adamw", "orthogonal_adamw", "muon"]
    assert len(summary["results"]) == 4
    for row in summary["results"]:
        assert row["optimizer"] in summary["optimizers"]
        assert row["status"] in {"ok", "unsupported"}
        assert row["repeats"] == 3
        assert row["successful_repeats"] <= 3
        assert len(row["runs"]) == row["successful_repeats"] or row["status"] != "ok"
        if row["status"] == "ok":
            assert row["steps_per_sec"] > 0.0
            assert row["step_time_ms"] >= 0.0
            assert row["optimizer_class"]
            assert row["steps_per_sec_mean"] > 0.0
            assert row["steps_per_sec_median"] > 0.0
            assert row["steps_per_sec_p95"] > 0.0
            assert row["step_time_ms_mean"] >= 0.0
            assert row["step_time_ms_median"] >= 0.0
            assert row["step_time_ms_p95"] >= 0.0
            assert len(row["runs"]) == 3