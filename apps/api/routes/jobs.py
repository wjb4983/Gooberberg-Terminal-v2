"""Job API routes."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy import select

from quant_platform.common.enums import JobStatus
from quant_platform.config import get_settings
from quant_platform.data.storage.catalog import MetadataCatalog, job_logs, jobs

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


class JobLogResponse(BaseModel):
    """A user-visible lifecycle log entry for a background job."""

    id: int
    job_id: int
    level: str
    message: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime | None = None


class JobLogsResponse(BaseModel):
    """Collection response for job log entries."""

    job_id: int
    logs: list[JobLogResponse]


class JobBoardResponse(BaseModel):
    """Jobs grouped for queue monitoring dashboards."""

    queued: list[JobResponse]
    running: list[JobResponse]
    finished: list[JobResponse]


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


@router.get("/board", response_model=JobBoardResponse)
def job_board() -> JobBoardResponse:
    """Return queued, running, and finished jobs for monitoring."""

    catalog = _catalog()
    catalog.create_all()
    rows = [dict(row) for row in catalog.list_rows("jobs")]
    queued = [row for row in rows if row["status"] == JobStatus.QUEUED.value]
    running = [row for row in rows if row["status"] == JobStatus.RUNNING.value]
    finished_statuses = {
        JobStatus.SUCCEEDED.value,
        JobStatus.FAILED.value,
        JobStatus.CANCELLED.value,
    }
    finished = [row for row in rows if row["status"] in finished_statuses]
    return JobBoardResponse(
        queued=[JobResponse(**row) for row in queued],
        running=[JobResponse(**row) for row in running],
        finished=[JobResponse(**row) for row in finished],
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


@router.get("/{job_id}/logs", response_model=JobLogsResponse)
def get_job_logs(job_id: int) -> JobLogsResponse:
    """Return user-visible logs for a background job."""

    catalog = _catalog()
    catalog.create_all()
    with catalog.engine.connect() as connection:
        job_exists = connection.execute(
            select(jobs.c.id).where(jobs.c.id == job_id)
        ).first()
        rows = connection.execute(
            select(job_logs)
            .where(job_logs.c.job_id == job_id)
            .order_by(job_logs.c.created_at.asc(), job_logs.c.id.asc())
        ).mappings().all()
    if job_exists is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"job not found: {job_id}",
        )
    return JobLogsResponse(
        job_id=job_id,
        logs=[JobLogResponse(**dict(row)) for row in rows],
    )
