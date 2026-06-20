"""Dataset API routes."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy import select

from quant_platform.common.enums import TaskType
from quant_platform.config import get_settings
from quant_platform.data.storage.catalog import (
    MetadataCatalog,
    coverage,
    dataset_definitions,
)
from quant_platform.datasets.registry import DatasetRegistry
from quant_platform.datasets.schemas import DatasetDefinition

router = APIRouter(prefix="/api/v1/datasets", tags=["datasets"])


class DatasetCreateRequest(BaseModel):
    """Request payload for registering a logical dataset."""

    definition: DatasetDefinition
    mirror_config: bool = False
    overwrite: bool = True


class DatasetResponse(BaseModel):
    """Dataset definition returned by the API."""

    id: int
    name: str
    version: str
    description: str | None = None
    schema_: dict[str, Any] = Field(alias="schema")
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime | None = None


class DatasetListResponse(BaseModel):
    """Collection response for registered datasets."""

    datasets: list[DatasetResponse]


class CoverageCreateRequest(BaseModel):
    """Request payload for manually recording dataset coverage."""

    symbol: str
    start_ts: datetime | None = None
    end_ts: datetime | None = None
    row_count: int | None = Field(default=None, ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class CoverageResponse(BaseModel):
    """Coverage row created for a dataset."""

    id: int
    dataset: str
    symbol: str
    start_ts: datetime | None = None
    end_ts: datetime | None = None
    row_count: int | None = None
    metadata: dict[str, Any]


class DatasetIngestRequest(BaseModel):
    """Request payload for queueing a dataset ingestion job."""

    provider: str | None = None
    symbols: list[str] = Field(default_factory=list)
    data_types: list[str] = Field(default_factory=list)
    start: str | None = None
    end: str | None = None
    resolution: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class DatasetIngestResponse(BaseModel):
    """Response returned after an ingestion job is queued."""

    job_id: int
    status: str
    job_type: str
    payload: dict[str, Any]


def _catalog() -> MetadataCatalog:
    settings = get_settings()
    return MetadataCatalog(settings.catalog_db_path)


def _dataset_response(row: Any) -> DatasetResponse:
    mapping = dict(row)
    return DatasetResponse(
        id=mapping["id"],
        name=mapping["name"],
        version=mapping["version"],
        description=mapping.get("description"),
        schema=mapping["schema"],
        metadata=mapping.get("metadata") or {},
        created_at=mapping.get("created_at"),
    )


def _get_dataset_row(catalog: MetadataCatalog, dataset_id: int) -> dict[str, Any]:
    with catalog.engine.connect() as connection:
        row = connection.execute(
            select(dataset_definitions).where(dataset_definitions.c.id == dataset_id)
        ).mappings().first()
    if row is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"dataset not found: {dataset_id}",
        )
    return dict(row)


@router.get("", response_model=DatasetListResponse)
def list_datasets() -> DatasetListResponse:
    """List registered dataset definitions."""

    catalog = _catalog()
    catalog.create_all()
    rows = catalog.list_rows("dataset_definitions")
    return DatasetListResponse(datasets=[_dataset_response(row) for row in rows])


@router.post("", response_model=DatasetResponse, status_code=status.HTTP_201_CREATED)
def create_dataset(request: DatasetCreateRequest) -> DatasetResponse:
    """Register a logical dataset definition."""

    catalog_path = get_settings().catalog_db_path
    registry = DatasetRegistry(catalog_path)
    dataset_id = registry.register(
        request.definition,
        mirror_config=request.mirror_config,
        overwrite=request.overwrite,
    )
    row = _get_dataset_row(registry.catalog, dataset_id)
    return _dataset_response(row)


@router.get("/{dataset_id}", response_model=DatasetResponse)
def get_dataset(dataset_id: int) -> DatasetResponse:
    """Return a registered dataset definition by id."""

    catalog = _catalog()
    catalog.create_all()
    return _dataset_response(_get_dataset_row(catalog, dataset_id))


@router.post(
    "/{dataset_id}/coverage",
    response_model=CoverageResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_coverage(
    dataset_id: int, request: CoverageCreateRequest
) -> CoverageResponse:
    """Record a coverage row for a dataset."""

    catalog = _catalog()
    catalog.create_all()
    dataset = _get_dataset_row(catalog, dataset_id)
    values = {
        "dataset": dataset["name"],
        "symbol": request.symbol.strip().upper(),
        "start_ts": request.start_ts,
        "end_ts": request.end_ts,
        "row_count": request.row_count,
        "metadata": {"dataset_id": dataset_id, **request.metadata},
    }
    row_id = catalog.insert_row("coverage", values)
    with catalog.engine.connect() as connection:
        row = connection.execute(
            select(coverage).where(coverage.c.id == row_id)
        ).mappings().one()
    return CoverageResponse(**dict(row))


@router.post("/{dataset_id}/ingest", response_model=DatasetIngestResponse)
def queue_dataset_ingest(
    dataset_id: int, request: DatasetIngestRequest
) -> DatasetIngestResponse:
    """Queue a dataset ingestion job placeholder in the metadata catalog."""

    catalog = _catalog()
    catalog.create_all()
    dataset = _get_dataset_row(catalog, dataset_id)
    payload = request.model_dump()
    payload["dataset_id"] = dataset_id
    payload["dataset_name"] = dataset["name"]
    job_id = catalog.insert_row(
        "jobs",
        {
            "job_type": TaskType.INGEST.value,
            "status": "queued",
            "payload": payload,
        },
    )
    return DatasetIngestResponse(
        job_id=job_id,
        status="queued",
        job_type=TaskType.INGEST.value,
        payload=payload,
    )
