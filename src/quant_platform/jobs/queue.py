"""Queue helpers for creating platform background jobs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from redis import Redis
from rq import Queue
from rq.command import send_stop_job_command
from rq.exceptions import NoSuchJobError
from rq.job import Job
from sqlalchemy import select

from quant_platform.common.enums import JobStatus
from quant_platform.common.ids import new_job_id
from quant_platform.config.settings import Settings, get_settings
from quant_platform.data.storage.catalog import MetadataCatalog, job_logs, jobs
from quant_platform.jobs import tasks

DEFAULT_QUEUE_NAME = "quant-platform-jobs"
JsonObject = dict[str, Any]


@dataclass(frozen=True)
class QueuedJob:
    """Identifiers for a job tracked in both RQ and the metadata catalog."""

    catalog_job_id: int
    rq_job_id: str
    queue_name: str
    job_type: str
    status: str


@dataclass(frozen=True)
class CancelJobResult:
    """Outcome from cancelling or deleting a catalog-backed RQ job."""

    catalog_job_id: int
    status: str
    rq_job_id: str | None
    action: str
    warnings: list[str]


def append_job_log(
    catalog_job_id: int,
    message: str,
    *,
    level: str = "info",
    metadata: Mapping[str, Any] | None = None,
    settings: Settings | None = None,
    catalog: MetadataCatalog | None = None,
) -> int:
    """Append a user-visible lifecycle log entry for a catalog job."""

    resolved_catalog = catalog or _catalog(settings)
    return resolved_catalog.insert_row(
        "job_logs",
        {
            "job_id": catalog_job_id,
            "level": level,
            "message": message,
            "metadata": dict(metadata or {}),
        },
    )


def list_job_logs(
    catalog_job_id: int,
    *,
    catalog: MetadataCatalog | None = None,
) -> list[dict[str, Any]]:
    """Return log entries for a catalog job in display order."""

    resolved_catalog = catalog or _catalog()
    with resolved_catalog.engine.connect() as connection:
        rows = connection.execute(
            select(job_logs)
            .where(job_logs.c.job_id == catalog_job_id)
            .order_by(job_logs.c.created_at.asc(), job_logs.c.id.asc())
        ).mappings()
        return [dict(row) for row in rows]


def list_jobs_by_status(
    statuses: set[str] | None = None,
    *,
    limit: int = 100,
    catalog: MetadataCatalog | None = None,
) -> list[dict[str, Any]]:
    """Return recent jobs, optionally filtered by lifecycle status."""

    resolved_catalog = catalog or _catalog()
    statement = select(jobs).order_by(jobs.c.created_at.desc()).limit(limit)
    if statuses:
        statement = (
            select(jobs)
            .where(jobs.c.status.in_(sorted(statuses)))
            .order_by(jobs.c.created_at.desc())
            .limit(limit)
        )
    with resolved_catalog.engine.connect() as connection:
        return [dict(row) for row in connection.execute(statement).mappings()]


def redis_connection(settings: Settings | None = None) -> Redis:
    """Create a Redis client from application settings."""

    resolved = settings or get_settings()
    return Redis.from_url(resolved.redis_url)


def job_queue(
    name: str = DEFAULT_QUEUE_NAME,
    *,
    settings: Settings | None = None,
    connection: Redis | None = None,
) -> Queue:
    """Return an RQ queue using the configured Redis URL."""

    return Queue(name, connection=connection or redis_connection(settings))


def _catalog(settings: Settings | None = None) -> MetadataCatalog:
    resolved = settings or get_settings()
    catalog = MetadataCatalog(resolved.catalog_db_path)
    catalog.create_all()
    return catalog


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _job_row(catalog: MetadataCatalog, catalog_job_id: int) -> dict[str, Any] | None:
    with catalog.engine.connect() as connection:
        row = (
            connection.execute(select(jobs).where(jobs.c.id == catalog_job_id))
            .mappings()
            .first()
        )
    return dict(row) if row is not None else None


def cancel_job(
    catalog_job_id: int,
    *,
    settings: Settings | None = None,
    queue: Queue | None = None,
    catalog: MetadataCatalog | None = None,
) -> CancelJobResult:
    """Cancel/delete a catalog-backed RQ job and record a user-visible log entry."""

    resolved_catalog = catalog or _catalog(settings)
    row = _job_row(resolved_catalog, catalog_job_id)
    if row is None:
        raise ValueError(f"job not found: {catalog_job_id}")

    current_status = str(row.get("status") or "")
    payload = dict(row.get("payload") or {})
    rq_job_id = payload.get("rq_job_id")
    warnings: list[str] = []
    action = "already_finished"
    log_level = "info"

    terminal_statuses = {
        JobStatus.SUCCEEDED.value,
        JobStatus.FAILED.value,
        JobStatus.CANCELLED.value,
    }
    if current_status in terminal_statuses:
        message = f"Cancellation/delete requested, but job is already {current_status}."
        append_job_log(
            catalog_job_id,
            message,
            metadata={"rq_job_id": rq_job_id, "action": action},
            catalog=resolved_catalog,
        )
        return CancelJobResult(
            catalog_job_id, current_status, rq_job_id, action, warnings
        )

    if not rq_job_id:
        warning = (
            "Catalog job payload has no rq_job_id; no Redis/RQ job could be removed."
        )
        warnings.append(warning)
        action = "catalog_cancelled_missing_rq_id"
        log_level = "warning"
    else:
        resolved_queue = queue or job_queue(settings=settings)
        connection = getattr(resolved_queue, "connection", None)
        try:
            rq_job = Job.fetch(str(rq_job_id), connection=connection)
        except NoSuchJobError:
            warning = f"No matching RQ job found in Redis for rq_job_id={rq_job_id}."
            warnings.append(warning)
            action = "catalog_cancelled_missing_rq_job"
            log_level = "warning"
        except Exception as exc:  # Redis/RQ lookup failures should be visible to users.
            warning = f"Failed to look up RQ job {rq_job_id}: {exc}"
            warnings.append(warning)
            action = "catalog_cancelled_rq_lookup_failed"
            log_level = "warning"
        else:
            try:
                if current_status == JobStatus.RUNNING.value:
                    rq_job.meta["cancellation_requested"] = True
                    rq_job.save_meta()
                    send_stop_job_command(connection, str(rq_job_id))
                    rq_job.cancel()
                    action = "stop_requested"
                else:
                    rq_job.cancel()
                    rq_job.delete(remove_from_queue=True)
                    action = "deleted_from_queue"
            except Exception as exc:
                warning = f"Failed to cancel/delete RQ job {rq_job_id}: {exc}"
                warnings.append(warning)
                action = "catalog_cancelled_rq_delete_failed"
                log_level = "warning"

    completed_at = _utcnow()
    new_payload = {**payload, "cancellation_requested": True}
    if warnings:
        new_payload["cancellation_warnings"] = warnings
    resolved_catalog.update_row(
        "jobs",
        catalog_job_id,
        {
            "status": JobStatus.CANCELLED.value,
            "payload": new_payload,
            "completed_at": completed_at,
        },
    )
    append_job_log(
        catalog_job_id,
        "Cancellation/delete requested for job.",
        level=log_level,
        metadata={
            "rq_job_id": rq_job_id,
            "action": action,
            "previous_status": current_status,
            "warnings": warnings,
        },
        catalog=resolved_catalog,
    )
    return CancelJobResult(
        catalog_job_id=catalog_job_id,
        status=JobStatus.CANCELLED.value,
        rq_job_id=rq_job_id,
        action=action,
        warnings=warnings,
    )


def enqueue_job(
    job_type: str,
    task_path: str,
    payload: Mapping[str, Any],
    *,
    settings: Settings | None = None,
    queue: Queue | None = None,
) -> QueuedJob:
    """Persist metadata for a job and enqueue it in Redis/RQ."""

    rq_job_id = new_job_id()
    normalized_payload: JsonObject = {**dict(payload), "rq_job_id": rq_job_id}
    catalog_job_id = _catalog(settings).insert_row(
        "jobs",
        {
            "job_type": job_type,
            "status": JobStatus.QUEUED.value,
            "payload": normalized_payload,
        },
    )
    append_job_log(
        catalog_job_id,
        f"Queued {job_type} job.",
        metadata={"rq_job_id": rq_job_id, "task_path": task_path},
        settings=settings,
    )
    resolved_queue = queue or job_queue(settings=settings)
    resolved_queue.enqueue(
        task_path,
        catalog_job_id,
        dict(payload),
        job_id=rq_job_id,
    )
    return QueuedJob(
        catalog_job_id=catalog_job_id,
        rq_job_id=rq_job_id,
        queue_name=resolved_queue.name,
        job_type=job_type,
        status=JobStatus.QUEUED.value,
    )


def enqueue_ingestion_job(
    payload: Mapping[str, Any],
    *,
    settings: Settings | None = None,
    queue: Queue | None = None,
) -> QueuedJob:
    """Enqueue a market-data ingestion job."""

    return enqueue_job(
        "ingestion",
        f"{tasks.run_ingestion_job.__module__}.{tasks.run_ingestion_job.__name__}",
        payload,
        settings=settings,
        queue=queue,
    )


def enqueue_training_job(
    payload: Mapping[str, Any],
    *,
    settings: Settings | None = None,
    queue: Queue | None = None,
) -> QueuedJob:
    """Enqueue a model training job."""

    return enqueue_job(
        "training",
        f"{tasks.run_training_job.__module__}.{tasks.run_training_job.__name__}",
        payload,
        settings=settings,
        queue=queue,
    )


def enqueue_backtest_job(
    payload: Mapping[str, Any],
    *,
    settings: Settings | None = None,
    queue: Queue | None = None,
) -> QueuedJob:
    """Enqueue a backtest job."""

    return enqueue_job(
        "backtest",
        f"{tasks.run_backtest_job.__module__}.{tasks.run_backtest_job.__name__}",
        payload,
        settings=settings,
        queue=queue,
    )


__all__ = [
    "DEFAULT_QUEUE_NAME",
    "QueuedJob",
    "CancelJobResult",
    "append_job_log",
    "cancel_job",
    "enqueue_backtest_job",
    "enqueue_ingestion_job",
    "enqueue_job",
    "enqueue_training_job",
    "job_queue",
    "list_job_logs",
    "list_jobs_by_status",
    "redis_connection",
]
