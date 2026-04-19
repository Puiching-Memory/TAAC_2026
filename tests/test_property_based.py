from __future__ import annotations

import numpy as np
from hypothesis import given, settings, strategies as st

from taac2026.application.training.cli import parse_train_args
from taac2026.domain.metrics import compute_classification_metrics
from taac2026.infrastructure.io.files import stable_hash64


@st.composite
def _metric_vectors(draw: st.DrawFn) -> tuple[list[int], list[float], list[int]]:
    size = draw(st.integers(min_value=0, max_value=24))
    labels = draw(st.lists(st.integers(min_value=0, max_value=1), min_size=size, max_size=size))
    logits = draw(
        st.lists(
            st.floats(allow_nan=False, allow_infinity=False, width=32),
            min_size=size,
            max_size=size,
        )
    )
    groups = draw(st.lists(st.integers(min_value=0, max_value=8), min_size=size, max_size=size))
    return labels, logits, groups


@settings(max_examples=32, deadline=None)
@given(inputs=_metric_vectors())
def test_metrics_hypothesis_outputs_remain_bounded(inputs: tuple[list[int], list[float], list[int]]) -> None:
    labels, logits, groups = inputs
    metrics = compute_classification_metrics(
        np.asarray(labels, dtype=np.float32),
        np.asarray(logits, dtype=np.float32),
        np.asarray(groups, dtype=np.int64),
    )

    assert 0.0 <= float(metrics["auc"]) <= 1.0
    assert 0.0 <= float(metrics["pr_auc"]) <= 1.0
    assert float(metrics["brier"]) >= 0.0
    assert float(metrics["logloss"]) >= 0.0
    assert 0.0 <= float(metrics["gauc"]["value"]) <= 1.0
    assert 0.0 <= float(metrics["gauc"]["coverage"]) <= 1.0


@settings(max_examples=32, deadline=None)
@given(value=st.text(max_size=128))
def test_stable_hash64_hypothesis_is_positive_and_deterministic(value: str) -> None:
    hashed = stable_hash64(value)

    assert hashed > 0
    assert hashed == stable_hash64(value)


@settings(max_examples=24, deadline=None)
@given(
    compile_enabled=st.booleans(),
    compile_backend=st.sampled_from([None, "inductor", "eager"]),
    compile_mode=st.sampled_from([None, "default", "max-autotune"]),
    amp_enabled=st.booleans(),
    amp_dtype=st.sampled_from(["float16", "bfloat16"]),
)
def test_parse_train_args_hypothesis_preserves_runtime_flags(
    compile_enabled: bool,
    compile_backend: str | None,
    compile_mode: str | None,
    amp_enabled: bool,
    amp_dtype: str,
) -> None:
    argv = ["--experiment", "config/baseline"]
    if compile_enabled:
        argv.append("--compile")
    if compile_backend is not None:
        argv.extend(["--compile-backend", compile_backend])
    if compile_mode is not None:
        argv.extend(["--compile-mode", compile_mode])
    if amp_enabled:
        argv.extend(["--amp", "--amp-dtype", amp_dtype])

    args = parse_train_args(argv)

    assert args.compile is compile_enabled or compile_backend is not None or compile_mode is not None
    assert args.compile_backend == compile_backend
    assert args.compile_mode == compile_mode
    assert args.amp is amp_enabled
    assert args.amp_dtype == (amp_dtype if amp_enabled else "float16")