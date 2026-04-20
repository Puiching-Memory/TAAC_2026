from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import importlib
import json
from pathlib import Path
import statistics
import sys
import time
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from taac2026.infrastructure.nn.te_backend import detect_transformer_engine_availability, is_transformer_engine_installed
from taac2026.infrastructure.nn.transformer import TaacCrossAttentionBlock, TaacTransformerBlock


DEFAULT_OUTPUT = REPO_ROOT / "outputs" / "performance" / "transformer_backends.json"
BACKEND_CHOICES = ("torch", "triton", "te")
SCENARIO_CHOICES = ("self-masked", "self-no-mask", "self-causal", "cross-masked", "cross-no-mask")


@dataclass(frozen=True, slots=True)
class ShapeProfile:
    name: str
    batch_size: int
    query_length: int
    context_length: int
    hidden_dim: int
    num_heads: int
    ffn_dim: int


def _load_experiment_profile(experiment_name: str) -> ShapeProfile:
    module = importlib.import_module(f"config.{experiment_name}")
    experiment = module.EXPERIMENT
    return ShapeProfile(
        name=f"{experiment_name}-default",
        batch_size=int(experiment.train.resolved_eval_batch_size),
        query_length=int(experiment.data.max_seq_len),
        context_length=max(int(experiment.data.max_seq_len), int(experiment.model.recent_seq_len or 0)),
        hidden_dim=int(experiment.model.hidden_dim),
        num_heads=int(experiment.model.num_heads),
        ffn_dim=int(experiment.model.hidden_dim * experiment.model.ffn_multiplier),
    )


def _build_profiles() -> dict[str, ShapeProfile]:
    return {
        "hyformer-default": _load_experiment_profile("hyformer"),
        "deepcontextnet-default": _load_experiment_profile("deepcontextnet"),
        "medium-reference": ShapeProfile(
            name="medium-reference",
            batch_size=32,
            query_length=128,
            context_length=128,
            hidden_dim=256,
            num_heads=8,
            ffn_dim=1024,
        ),
        "large-reference": ShapeProfile(
            name="large-reference",
            batch_size=16,
            query_length=256,
            context_length=256,
            hidden_dim=512,
            num_heads=8,
            ffn_dim=2048,
        ),
    }


def _build_prefix_mask(batch_size: int, sequence_length: int, device: torch.device) -> torch.Tensor:
    positions = torch.arange(sequence_length, device=device).unsqueeze(0)
    max_reduction = max(1, sequence_length // 4)
    reductions = torch.arange(batch_size, device=device) % (max_reduction + 1)
    minimum_length = max(1, sequence_length - max_reduction)
    lengths = torch.clamp(sequence_length - reductions, min=minimum_length)
    return positions < lengths.unsqueeze(1)


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _summarize_times(times_ms: list[float]) -> dict[str, float]:
    ordered = sorted(times_ms)
    median_ms = float(statistics.median(ordered))
    mean_ms = float(statistics.fmean(ordered))
    p95_index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * 0.95))))
    p99_index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * 0.99))))
    return {
        "median_ms": median_ms,
        "mean_ms": mean_ms,
        "min_ms": float(ordered[0]),
        "max_ms": float(ordered[-1]),
        "p95_ms": float(ordered[p95_index]),
        "p99_ms": float(ordered[p99_index]),
        "times_ms": [float(value) for value in times_ms],
    }


def _measure_callable(target, *, device: torch.device, warmup: int, steps: int) -> dict[str, float | list[float]]:
    with torch.inference_mode():
        for _ in range(max(0, warmup)):
            target()
        _synchronize(device)

        times_ms: list[float] = []
        if device.type == "cuda":
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            for _ in range(steps):
                start_event.record()
                target()
                end_event.record()
                torch.cuda.synchronize(device)
                times_ms.append(float(start_event.elapsed_time(end_event)))
        else:
            for _ in range(steps):
                start_time = time.perf_counter()
                target()
                times_ms.append((time.perf_counter() - start_time) * 1e3)
    return _summarize_times(times_ms)


def _build_case(
    *,
    scenario: str,
    backend: str,
    profile: ShapeProfile,
    device: torch.device,
    te_precision: str,
    te_recipe_mode: str,
):
    if scenario.startswith("self-"):
        block = TaacTransformerBlock(
            hidden_dim=profile.hidden_dim,
            num_heads=profile.num_heads,
            ffn_dim=profile.ffn_dim,
            dropout=0.0,
            attention_dropout=0.0,
            norm_type="rmsnorm",
            ffn_type="swiglu",
            attention_type="causal" if scenario == "self-causal" else "standard",
            attention_backend=backend,
            ffn_backend=backend,
            te_precision=te_precision,
            te_recipe_mode=te_recipe_mode,
        ).to(device)
        block.eval()
        hidden_states = torch.randn(
            profile.batch_size,
            profile.query_length,
            profile.hidden_dim,
            device=device,
        )
        token_mask = _build_prefix_mask(profile.batch_size, profile.query_length, device) if scenario == "self-masked" else None

        def run() -> torch.Tensor:
            return block(hidden_states, token_mask)

        return run

    block = TaacCrossAttentionBlock(
        hidden_dim=profile.hidden_dim,
        num_heads=profile.num_heads,
        ffn_dim=profile.ffn_dim,
        dropout=0.0,
        attention_dropout=0.0,
        norm_type="layernorm",
        ffn_type="gelu",
        attention_backend=backend,
        ffn_backend=backend,
        te_precision=te_precision,
        te_recipe_mode=te_recipe_mode,
    ).to(device)
    block.eval()
    query_states = torch.randn(
        profile.batch_size,
        profile.query_length,
        profile.hidden_dim,
        device=device,
    )
    context_states = torch.randn(
        profile.batch_size,
        profile.context_length,
        profile.hidden_dim,
        device=device,
    )
    if scenario == "cross-masked":
        query_mask = _build_prefix_mask(profile.batch_size, profile.query_length, device)
        context_mask = _build_prefix_mask(profile.batch_size, profile.context_length, device)
    else:
        query_mask = None
        context_mask = None

    def run() -> torch.Tensor:
        return block(query_states, context_states, query_mask=query_mask, context_mask=context_mask)

    return run


def _benchmark_case(
    *,
    scenario: str,
    backend: str,
    profile: ShapeProfile,
    device: torch.device,
    warmup: int,
    steps: int,
    te_precision: str,
    te_recipe_mode: str,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "profile": asdict(profile),
        "scenario": scenario,
        "backend": backend,
    }
    if backend == "te" and not is_transformer_engine_installed():
        record.update({
            "skipped": True,
            "reason": "Transformer Engine is not installed",
        })
        return record

    try:
        run = _build_case(
            scenario=scenario,
            backend=backend,
            profile=profile,
            device=device,
            te_precision=te_precision,
            te_recipe_mode=te_recipe_mode,
        )
        timings = _measure_callable(run, device=device, warmup=warmup, steps=steps)
    except Exception as exc:
        record.update({
            "skipped": True,
            "reason": str(exc),
        })
        return record

    query_tokens = profile.batch_size * profile.query_length
    attention_pairs = profile.batch_size * profile.query_length * profile.context_length
    median_ms = float(timings["median_ms"])
    record.update(timings)
    record.update(
        {
            "skipped": False,
            "query_tokens_per_second": float(query_tokens) / max(median_ms / 1e3, 1e-9),
            "attention_pairs_per_second": float(attention_pairs) / max(median_ms / 1e3, 1e-9),
        }
    )
    return record


def _parse_args() -> argparse.Namespace:
    profiles = _build_profiles()
    parser = argparse.ArgumentParser(description="Benchmark shared transformer block backends on CUDA.")
    parser.add_argument(
        "--profiles",
        nargs="+",
        default=["hyformer-default", "deepcontextnet-default"],
        choices=sorted(profiles),
        help="Shape profiles to benchmark.",
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=["self-masked", "self-no-mask", "self-causal", "cross-masked"],
        choices=SCENARIO_CHOICES,
        help="Block scenarios to benchmark.",
    )
    parser.add_argument(
        "--backends",
        nargs="+",
        default=list(BACKEND_CHOICES),
        choices=BACKEND_CHOICES,
        help="Backends to compare.",
    )
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations per case.")
    parser.add_argument("--steps", type=int, default=50, help="Measured iterations per case.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for input generation.")
    parser.add_argument("--te-precision", default="auto", help="Transformer Engine precision override.")
    parser.add_argument("--te-recipe-mode", default="auto", help="Transformer Engine recipe override.")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="JSON output path.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for transformer backend benchmarks")

    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    profiles = _build_profiles()
    results = []
    for profile_name in args.profiles:
        profile = profiles[profile_name]
        for scenario in args.scenarios:
            for backend in args.backends:
                torch.cuda.empty_cache()
                result = _benchmark_case(
                    scenario=scenario,
                    backend=backend,
                    profile=profile,
                    device=device,
                    warmup=args.warmup,
                    steps=args.steps,
                    te_precision=args.te_precision,
                    te_recipe_mode=args.te_recipe_mode,
                )
                results.append(result)
                if result.get("skipped"):
                    print(f"SKIP {profile.name} {scenario} {backend}: {result['reason']}")
                else:
                    print(
                        f"OK   {profile.name} {scenario} {backend}: "
                        f"median={result['median_ms']:.3f} ms "
                        f"tokens/s={result['query_tokens_per_second']:.0f}"
                    )

    payload = {
        "device": {
            "name": torch.cuda.get_device_name(device),
            "compute_capability": list(torch.cuda.get_device_capability(device)),
        },
        "seed": args.seed,
        "warmup": args.warmup,
        "steps": args.steps,
        "te_precision": args.te_precision,
        "te_recipe_mode": args.te_recipe_mode,
        "transformer_engine": detect_transformer_engine_availability(device),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved benchmark results to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())