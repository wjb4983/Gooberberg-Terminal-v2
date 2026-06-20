"""Ingestion API routes."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel, Field

from quant_platform.common.enums import AssetClass
from quant_platform.config import get_settings
from quant_platform.data.ingestion.coverage import CoverageStore
from quant_platform.data.ingestion.planner import (
    IngestionRequest,
    all_requested_partitions,
)
from quant_platform.data.storage.catalog import MetadataCatalog

router = APIRouter(prefix="/api/v1/ingestion", tags=["ingestion"])


class IngestionPlanRequest(BaseModel):
    """Request payload for planning missing ingestion partitions."""

    provider: str
    symbols: list[str]
    data_types: list[str]
    start: date
    end: date
    asset_class: str = AssetClass.EQUITY.value
    resolution: str | None = None


class IngestionPartitionResponse(BaseModel):
    """API representation of a planned ingestion partition."""

    provider: str
    asset_class: str
    data_type: str
    symbol: str
    date: date
    dataset: str
    resolution: str | None = None


class IngestionPlanResponse(BaseModel):
    """Planned ingestion partitions split by requested and missing counts."""

    requested: int
    missing: int
    partitions: list[IngestionPartitionResponse]


class IngestionManifestResponse(BaseModel):
    """Catalog row for an ingestion manifest."""

    id: int
    provider: str
    dataset: str
    status: str
    source_uri: str | None = None
    artifact_uri: str | None = None
    row_count: int | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime | None = None


class IngestionManifestListResponse(BaseModel):
    """Collection response for ingestion manifests."""

    manifests: list[IngestionManifestResponse]


def _catalog() -> MetadataCatalog:
    settings = get_settings()
    return MetadataCatalog(settings.catalog_db_path)


@router.post("/plan", response_model=IngestionPlanResponse)
def plan_ingestion(request: IngestionPlanRequest) -> IngestionPlanResponse:
    """Plan missing ingestion partitions without pulling provider data."""

    ingestion_request = IngestionRequest.create(
        provider=request.provider,
        symbols=request.symbols,
        data_types=request.data_types,
        start=request.start,
        end=request.end,
        asset_class=request.asset_class,
        resolution=request.resolution,
    )
    requested = all_requested_partitions(ingestion_request)
    catalog = _catalog()
    missing = CoverageStore(catalog).missing_partitions(requested)
    return IngestionPlanResponse(
        requested=len(requested),
        missing=len(missing),
        partitions=[
            IngestionPartitionResponse(
                provider=partition.provider,
                asset_class=partition.asset_class,
                data_type=partition.data_type,
                symbol=partition.symbol,
                date=partition.date,
                dataset=partition.dataset,
                resolution=partition.resolution,
            )
            for partition in missing
        ],
    )


@router.get("/manifests", response_model=IngestionManifestListResponse)
def list_manifests() -> IngestionManifestListResponse:
    """List ingestion manifest catalog rows."""

    catalog = _catalog()
    catalog.create_all()
    rows = catalog.list_rows("ingestion_manifests")
    return IngestionManifestListResponse(
        manifests=[IngestionManifestResponse(**dict(row)) for row in rows]
    )
