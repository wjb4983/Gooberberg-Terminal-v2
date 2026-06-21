"""Schemas for minimal training runs."""

from __future__ import annotations

from datetime import date
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from quant_platform.models.schemas import ModelType


class TaskType(StrEnum):
    """Supported prediction task types."""

    REGRESSION = "regression"
    BINARY_CLASSIFICATION = "binary_classification"


class OptimizerName(StrEnum):
    """Supported optimizer choices."""

    ADAM = "adam"
    SGD = "sgd"


class LossName(StrEnum):
    """Supported loss function choices."""

    MSE = "mse"
    MAE = "mae"
    BCE_WITH_LOGITS = "bce_with_logits"


class TargetDefinition(BaseModel):
    """Definition of the synthetic target produced by the data module."""

    model_config = ConfigDict(extra="forbid")

    name: str = "synthetic_target"
    horizon: int = Field(default=1, gt=0)
    expression: str = "weighted_feature_sum"


class DateSplitConfig(BaseModel):
    """Date boundaries for deterministic train/validation/test splits."""

    model_config = ConfigDict(extra="forbid")

    train_start: date
    train_end: date
    validation_start: date
    validation_end: date
    test_start: date | None = None
    test_end: date | None = None

    @model_validator(mode="after")
    def _validate_order(self) -> DateSplitConfig:
        if self.train_start > self.train_end:
            raise ValueError("train_start must be on or before train_end")
        if self.validation_start > self.validation_end:
            raise ValueError("validation_start must be on or before validation_end")
        if self.train_end >= self.validation_start:
            raise ValueError("train_end must be before validation_start")
        if self.test_start is not None and self.test_end is not None:
            if self.test_start > self.test_end:
                raise ValueError("test_start must be on or before test_end")
            if self.validation_end >= self.test_start:
                raise ValueError("validation_end must be before test_start")
        return self


class WalkForwardConfig(BaseModel):
    """Placeholder for future walk-forward training support."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    window_days: int | None = Field(default=None, gt=0)
    step_days: int | None = Field(default=None, gt=0)


class EarlyStoppingConfig(BaseModel):
    """Placeholder for future early stopping support."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    patience: int = Field(default=5, gt=0)
    monitor: str = "validation_loss"


class TrainingConfig(BaseModel):
    """Configuration for a minimal but functional training run."""

    model_config = ConfigDict(extra="forbid", use_enum_values=False)

    experiment_name: str = "synthetic-training-run"
    experiment_id: int | None = None
    dataset_name: str = "synthetic_prices"
    dataset_version: str = "1"
    model_name: str = "synthetic_mlp"
    model_type: ModelType = ModelType.MLP
    task_type: TaskType = TaskType.REGRESSION
    target: TargetDefinition = Field(default_factory=TargetDefinition)
    feature_set: list[str] = Field(
        default_factory=lambda: ["feature_0", "feature_1", "feature_2", "feature_3"]
    )
    date_split: DateSplitConfig = Field(
        default_factory=lambda: DateSplitConfig(
            train_start=date(2024, 1, 1),
            train_end=date(2024, 3, 31),
            validation_start=date(2024, 4, 1),
            validation_end=date(2024, 4, 30),
            test_start=date(2024, 5, 1),
            test_end=date(2024, 5, 31),
        )
    )
    walk_forward: WalkForwardConfig = Field(default_factory=WalkForwardConfig)
    batch_size: int = Field(default=16, gt=0)
    epochs: int = Field(default=2, gt=0)
    optimizer: OptimizerName = OptimizerName.ADAM
    learning_rate: float = Field(default=1e-3, gt=0)
    loss_function: LossName = LossName.MSE
    early_stopping: EarlyStoppingConfig = Field(default_factory=EarlyStoppingConfig)
    sequence_length: int = Field(default=8, gt=0)
    hidden_size: int = Field(default=16, gt=0)
    artifact_dir: Path = Path("data/artifacts/experiments")
    seed: int = 7
    synthetic_rows_per_day: int = Field(default=4, gt=0)

    @model_validator(mode="after")
    def _validate_choices(self) -> TrainingConfig:
        if not self.feature_set:
            raise ValueError("feature_set must include at least one feature")
        if (
            self.task_type == TaskType.BINARY_CLASSIFICATION
            and self.loss_function != LossName.BCE_WITH_LOGITS
        ):
            raise ValueError("binary_classification requires bce_with_logits loss")
        return self

    def jsonable(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        return self.model_dump(mode="json")
