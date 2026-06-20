"""Queue helpers for creating platform background jobs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from redis import Redis
from rq import Queue

from quant_platform.common.enums import JobStatus
from quant_platform.common.ids import new_job_id
from quant_platform.config.settings import Settings, get_settings
from quant_platform.data.storage.catalog import MetadataCatalog
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
    "enqueue_backtest_job",
    "enqueue_ingestion_job",
    "enqueue_job",
    "enqueue_training_job",
    "job_queue",
    "redis_connection",
]
