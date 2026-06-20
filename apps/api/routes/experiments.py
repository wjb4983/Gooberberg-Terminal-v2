"""Experiment API routes."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy import select

from quant_platform.common.enums import TaskType as JobTaskType
from quant_platform.config import get_settings
from quant_platform.data.storage.catalog import (
    MetadataCatalog,
    dataset_definitions,
    experiment_metrics,
    experiments,
    feature_sets,
    model_definitions,
)
from quant_platform.training.schemas import (
    EarlyStoppingConfig,
    LossName,
    OptimizerName,
    TargetDefinition,
    TaskType,
    WalkForwardConfig,
)

router = APIRouter(prefix="/api/v1/experiments", tags=["experiments"])


class SplitConfig(BaseModel):
    """Date split configuration for a queued training experiment."""

    train_start: date
    train_end: date
    validation_start: date
    validation_end: date
    test_start: date | None = None
    test_end: date | None = None


class TrainingParams(BaseModel):
    """Training hyperparameters accepted by the experiment queue endpoint."""

    batch_size: int = Field(default=16, gt=0)
    epochs: int = Field(default=2, gt=0)
    optimizer: OptimizerName = OptimizerName.ADAM
    learning_rate: float = Field(default=1e-3, gt=0)
    loss_function: LossName = LossName.MSE
    sequence_length: int = Field(default=8, gt=0)
    hidden_size: int = Field(default=16, gt=0)
    seed: int = 7
    synthetic_rows_per_day: int = Field(default=4, gt=0)
    early_stopping: EarlyStoppingConfig = Field(default_factory=EarlyStoppingConfig)
    walk_forward: WalkForwardConfig = Field(default_factory=WalkForwardConfig)


class ExperimentQueueRequest(BaseModel):
    """Payload for creating an experiment and queueing its training job."""

    name: str = Field(min_length=1)
    dataset_id: int
    feature_set_id: int | None = None
    model_id: int
    task_type: TaskType = TaskType.REGRESSION
    target: TargetDefinition = Field(default_factory=TargetDefinition)
    split: SplitConfig
    training: TrainingParams = Field(default_factory=TrainingParams)
    metadata: dict[str, Any] = Field(default_factory=dict)


class QueuedExperimentResponse(BaseModel):
    """Response returned after queueing a training experiment."""

    experiment_id: int
    job_id: int
    status: str
    payload: dict[str, Any]


class MetricResponse(BaseModel):
    """Experiment metric row."""

    id: int
    experiment_id: int
    name: str
    value: float
    step: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime | None = None


class ExperimentResponse(BaseModel):
    """Experiment row with related metrics and artifact links."""

    id: int
    name: str
    status: str
    model_id: int | None = None
    dataset_id: int | None = None
    feature_set_id: int | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    metrics: list[MetricResponse] = Field(default_factory=list)
    artifacts: dict[str, str] = Field(default_factory=dict)
    created_at: datetime | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None


class ExperimentListResponse(BaseModel):
    """Collection response for experiments."""

    experiments: list[ExperimentResponse]


def _catalog() -> MetadataCatalog:
    catalog = MetadataCatalog(get_settings().catalog_db_path)
    catalog.create_all()
    return catalog


def _require_row(
    catalog: MetadataCatalog, table: Any, row_id: int, label: str
) -> dict[str, Any]:
    with catalog.engine.connect() as connection:
        row = (
            connection.execute(select(table).where(table.c.id == row_id))
            .mappings()
            .first()
        )
    if row is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"{label} not found: {row_id}"
        )
    return dict(row)


def _metrics(catalog: MetadataCatalog, experiment_id: int) -> list[MetricResponse]:
    with catalog.engine.connect() as connection:
        rows = (
            connection.execute(
                select(experiment_metrics).where(
                    experiment_metrics.c.experiment_id == experiment_id
                )
            )
            .mappings()
            .all()
        )
    return [MetricResponse(**dict(row)) for row in rows]


def _experiment_response(catalog: MetadataCatalog, row: Any) -> ExperimentResponse:
    mapping = dict(row)
    metadata = mapping.get("metadata") or {}
    artifacts = metadata.get("artifacts") or metadata.get("artifact_links") or {}
    mapping["metadata"] = metadata
    return ExperimentResponse(
        **mapping,
        metrics=_metrics(catalog, int(mapping["id"])),
        artifacts=dict(artifacts),
    )


@router.get("", response_model=ExperimentListResponse)
def list_experiments() -> ExperimentListResponse:
    """List experiments with metrics and artifact links when present."""

    catalog = _catalog()
    return ExperimentListResponse(
        experiments=[
            _experiment_response(catalog, row)
            for row in catalog.list_rows("experiments")
        ]
    )


@router.post(
    "", response_model=QueuedExperimentResponse, status_code=status.HTTP_201_CREATED
)
def queue_experiment(request: ExperimentQueueRequest) -> QueuedExperimentResponse:
    """Create an experiment record and queue a training job in the metadata catalog."""

    catalog = _catalog()
    dataset = _require_row(catalog, dataset_definitions, request.dataset_id, "dataset")
    model = _require_row(
        catalog, model_definitions, request.model_id, "model definition"
    )
    feature_set = None
    if request.feature_set_id is not None:
        feature_set = _require_row(
            catalog, feature_sets, request.feature_set_id, "feature set"
        )

    feature_names = list((feature_set or {}).get("features") or [])
    parameters = {
        "task_type": request.task_type.value,
        "target": request.target.model_dump(mode="json"),
        "split": request.split.model_dump(mode="json"),
        "training": request.training.model_dump(mode="json"),
    }
    payload = {
        "experiment_name": request.name.strip(),
        "dataset_id": request.dataset_id,
        "dataset_name": dataset["name"],
        "dataset_version": dataset["version"],
        "feature_set_id": request.feature_set_id,
        "feature_set": feature_names,
        "model_id": request.model_id,
        "model_name": model["name"],
        "model_type": model.get("model_type"),
        **parameters,
    }
    experiment_id = catalog.insert_row(
        "experiments",
        {
            "name": request.name.strip(),
            "status": "queued",
            "model_id": request.model_id,
            "dataset_id": request.dataset_id,
            "feature_set_id": request.feature_set_id,
            "parameters": parameters,
            "metadata": {**request.metadata, "queued_payload": payload},
        },
    )
    payload["experiment_id"] = experiment_id
    job_id = catalog.insert_row(
        "jobs",
        {"job_type": JobTaskType.TRAIN.value, "status": "queued", "payload": payload},
    )
    return QueuedExperimentResponse(
        experiment_id=experiment_id, job_id=job_id, status="queued", payload=payload
    )


@router.get("/{experiment_id}", response_model=ExperimentResponse)
def get_experiment(experiment_id: int) -> ExperimentResponse:
    """Return an experiment with metrics and artifact links when present."""

    catalog = _catalog()
    row = _require_row(catalog, experiments, experiment_id, "experiment")
    return _experiment_response(catalog, row)
