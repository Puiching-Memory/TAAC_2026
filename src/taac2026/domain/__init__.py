"""Core domain contracts shared by applications and experiment packages."""

from taac2026.domain.config import EvalRequest, InferRequest, TrainRequest
from taac2026.domain.experiment import ExperimentSpec

__all__ = ["EvalRequest", "ExperimentSpec", "InferRequest", "TrainRequest"]
