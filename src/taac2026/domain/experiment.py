from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Callable

from .config import DataConfig, ModelConfig, SearchConfig, TrainConfig
from .features import FeatureSchema, sync_feature_schema


def _missing_build_model_component(*args: Any, **kwargs: Any) -> Any:
    del args
    del kwargs
    raise RuntimeError("ExperimentSpec requires build_model_component to be defined")


@dataclass(slots=True)
class ExperimentSpec:
    name: str
    data: DataConfig
    model: ModelConfig
    train: TrainConfig
    feature_schema: FeatureSchema | None = None
    build_data_pipeline: Callable[..., Any] | None = None
    build_model_component: Callable[..., Any] = _missing_build_model_component
    build_loss_stack: Callable[..., Any] | None = None
    build_optimizer_component: Callable[..., Any] | None = None
    switches: dict[str, bool] = field(default_factory=dict)
    search: SearchConfig = field(default_factory=SearchConfig)
    build_search_experiment: Callable[["ExperimentSpec", Any], "ExperimentSpec"] | None = None

    def __post_init__(self) -> None:
        if not callable(self.build_model_component):
            raise TypeError("ExperimentSpec.build_model_component must be callable")
        self.refresh_feature_schema()

    def refresh_feature_schema(self) -> None:
        self.feature_schema = sync_feature_schema(self.feature_schema, self.data, self.model)

    def clone(self) -> "ExperimentSpec":
        return ExperimentSpec(
            name=self.name,
            data=deepcopy(self.data),
            model=deepcopy(self.model),
            train=deepcopy(self.train),
            feature_schema=deepcopy(self.feature_schema),
            build_data_pipeline=self.build_data_pipeline,
            build_model_component=self.build_model_component,
            build_loss_stack=self.build_loss_stack,
            build_optimizer_component=self.build_optimizer_component,
            switches=deepcopy(self.switches),
            search=deepcopy(self.search),
            build_search_experiment=self.build_search_experiment,
        )

    def derive(self, **updates: Any) -> "ExperimentSpec":
        experiment = self.clone()
        for key, value in updates.items():
            if not hasattr(experiment, key):
                raise AttributeError(f"ExperimentSpec has no field '{key}'")
            setattr(experiment, key, value)
        experiment.refresh_feature_schema()
        return experiment


__all__ = ["ExperimentSpec"]
