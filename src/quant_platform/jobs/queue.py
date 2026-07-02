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
from sqlalchemy import select, update

from quant_platform.common.enums import JobStatus
from quant_platform.common.ids import new_job_id
from quant_platform.config.settings import Settings, get_settings
from quant_platform.data.storage.catalog import (
    MetadataCatalog,
    experiments,
    job_logs,
    jobs,
)
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
class ReconciledJob:
    """Outcome from reconciling one catalog job against RQ state."""

    catalog_job_id: int
    previous_status: str
    status: str
    rq_job_id: str | None
    action: str
    message: str


@dataclass(frozen=True)
class ReconcileJobsResult:
    """Summary returned after reconciling catalog jobs against RQ state."""

    checked: int
    reconciled: list[ReconciledJob]


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


def _registry_job_ids(registry: Any) -> set[str]:
    """Return job ids from an RQ registry-like object."""

    if hasattr(registry, "get_job_ids"):
        return {str(job_id) for job_id in registry.get_job_ids()}
    if hasattr(registry, "get_jobs"):
        return {str(getattr(job, "id", job)) for job in registry.get_jobs()}
    return set()


def _rq_job_exists(rq_job_id: str, queue: Queue) -> bool:
    """Return whether an RQ job id exists in live queue/registry state."""

    if hasattr(queue, "fetch_job") and queue.fetch_job(rq_job_id) is not None:
        return True

    if hasattr(queue, "get_job_ids") and rq_job_id in {
        str(job_id) for job_id in queue.get_job_ids()
    }:
        return True

    queue_jobs = getattr(queue, "jobs", None)
    if queue_jobs is not None:
        if any(str(getattr(job, "id", job)) == rq_job_id for job in queue_jobs):
            return True

    registry_names = (
        "queued_job_registry",
        "started_job_registry",
        "deferred_job_registry",
        "scheduled_job_registry",
        "failed_job_registry",
        "finished_job_registry",
    )
    for name in registry_names:
        registry = getattr(queue, name, None)
        if registry is not None and rq_job_id in _registry_job_ids(registry):
            return True

    registry_factory_names = (
        "get_started_job_registry",
        "get_deferred_job_registry",
        "get_scheduler",
        "get_failed_job_registry",
        "get_finished_job_registry",
    )
    for name in registry_factory_names:
        factory = getattr(queue, name, None)
        if factory is None:
            continue
        registry = factory()
        if rq_job_id in _registry_job_ids(registry):
            return True
    return False


def _update_linked_experiment(
    catalog: MetadataCatalog,
    experiment_id: Any,
    *,
    status: str,
    completed_at: datetime,
) -> None:
    """Move a queued/running linked experiment to the reconciled terminal status."""

    if experiment_id is None:
        return
    with catalog.engine.begin() as connection:
        connection.execute(
            update(experiments)
            .where(experiments.c.id == int(experiment_id))
            .where(
                experiments.c.status.in_(
                    [JobStatus.QUEUED.value, JobStatus.RUNNING.value]
                )
            )
            .values(status=status, completed_at=completed_at)
        )


def _mark_reconciled_job(
    catalog: MetadataCatalog,
    row: Mapping[str, Any],
    *,
    status: JobStatus,
    action: str,
    message: str,
) -> ReconciledJob:
    """Mark one catalog job terminal and append a reconciliation log."""

    completed_at = _utcnow()
    catalog_job_id = int(row["id"])
    previous_status = str(row.get("status") or "")
    payload = dict(row.get("payload") or {})
    rq_job_id = payload.get("rq_job_id")
    new_payload = {
        **payload,
        "reconciliation": {
            "action": action,
            "message": message,
            "previous_status": previous_status,
            "completed_at": completed_at.isoformat(),
        },
    }
    catalog.update_row(
        "jobs",
        catalog_job_id,
        {
            "status": status.value,
            "payload": new_payload,
            "error": message if status == JobStatus.FAILED else row.get("error"),
            "completed_at": completed_at,
        },
    )
    _update_linked_experiment(
        catalog,
        payload.get("experiment_id"),
        status=status.value,
        completed_at=completed_at,
    )
    append_job_log(
        catalog_job_id,
        message,
        level="warning",
        metadata={
            "rq_job_id": rq_job_id,
            "action": action,
            "previous_status": previous_status,
            "reconciled_status": status.value,
        },
        catalog=catalog,
    )
    return ReconciledJob(
        catalog_job_id=catalog_job_id,
        previous_status=previous_status,
        status=status.value,
        rq_job_id=str(rq_job_id) if rq_job_id else None,
        action=action,
        message=message,
    )


def reconcile_stale_jobs(
    statuses: set[str] | None = None,
    *,
    stale_status: JobStatus = JobStatus.CANCELLED,
    settings: Settings | None = None,
    queue: Queue | None = None,
    catalog: MetadataCatalog | None = None,
) -> ReconcileJobsResult:
    """Reconcile queued/running catalog jobs that are missing from RQ state."""

    watched_statuses = statuses or {JobStatus.QUEUED.value, JobStatus.RUNNING.value}
    resolved_catalog = catalog or _catalog(settings)
    resolved_queue = queue or job_queue(settings=settings)
    rows = list_jobs_by_status(
        set(watched_statuses), limit=10_000, catalog=resolved_catalog
    )
    reconciled: list[ReconciledJob] = []
    for row in rows:
        payload = dict(row.get("payload") or {})
        rq_job_id = payload.get("rq_job_id")
        if not rq_job_id:
            reconciled.append(
                _mark_reconciled_job(
                    resolved_catalog,
                    row,
                    status=JobStatus.FAILED,
                    action="missing_rq_job_id",
                    message="missing rq_job_id; job was never enqueued",
                )
            )
            continue
        if not _rq_job_exists(str(rq_job_id), resolved_queue):
            reconciled.append(
                _mark_reconciled_job(
                    resolved_catalog,
                    row,
                    status=stale_status,
                    action="stale_rq_job_missing",
                    message=(
                        "RQ job not found in queued, started, deferred, scheduled, "
                        "failed, or finished registries; catalog job marked stale"
                    ),
                )
            )
    return ReconcileJobsResult(checked=len(rows), reconciled=reconciled)

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
    "ReconciledJob",
    "ReconcileJobsResult",
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
    "reconcile_stale_jobs",
    "redis_connection",
]
