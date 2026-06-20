"""Job API routes."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy import select

from quant_platform.config import get_settings
from quant_platform.data.storage.catalog import MetadataCatalog, jobs

router = APIRouter(prefix="/api/v1/jobs", tags=["jobs"])


class JobResponse(BaseModel):
    """Metadata catalog job row."""

    id: int
    job_type: str
    status: str
    payload: dict[str, Any] = Field(default_factory=dict)
    result: dict[str, Any] | None = None
    error: str | None = None
    created_at: datetime | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None


class JobListResponse(BaseModel):
    """Collection response for jobs."""

    jobs: list[JobResponse]


def _catalog() -> MetadataCatalog:
    settings = get_settings()
    return MetadataCatalog(settings.catalog_db_path)


@router.get("", response_model=JobListResponse)
def list_jobs() -> JobListResponse:
    """List background jobs in the metadata catalog."""

    catalog = _catalog()
    catalog.create_all()
    return JobListResponse(
        jobs=[JobResponse(**dict(row)) for row in catalog.list_rows("jobs")]
    )


@router.get("/{job_id}", response_model=JobResponse)
def get_job(job_id: int) -> JobResponse:
    """Return a background job by id."""

    catalog = _catalog()
    catalog.create_all()
    with catalog.engine.connect() as connection:
        row = (
            connection.execute(select(jobs).where(jobs.c.id == job_id))
            .mappings()
            .first()
        )
    if row is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"job not found: {job_id}",
        )
    return JobResponse(**dict(row))
