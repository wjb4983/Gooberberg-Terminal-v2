"""RQ task functions for platform background jobs."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import update

from quant_platform.common.enums import JobStatus, Provider
from quant_platform.config.settings import get_settings
from quant_platform.data.ingestion.service import IngestionService
from quant_platform.data.providers import (
    FakeMarketDataProvider,
    MassiveMarketDataProvider,
)
from quant_platform.data.storage.catalog import MetadataCatalog, jobs

JsonObject = dict[str, Any]


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _catalog() -> MetadataCatalog:
    settings = get_settings()
    catalog = MetadataCatalog(settings.catalog_db_path)
    catalog.create_all()
    return catalog


def _update_job(
    catalog_job_id: int,
    *,
    status: JobStatus,
    result: Mapping[str, Any] | None = None,
    error: str | None = None,
    started_at: datetime | None = None,
    completed_at: datetime | None = None,
) -> None:
    values: dict[str, Any] = {"status": status.value}
    if result is not None:
        values["result"] = dict(result)
    if error is not None:
        values["error"] = error
    if started_at is not None:
        values["started_at"] = started_at
    if completed_at is not None:
        values["completed_at"] = completed_at

    catalog = _catalog()
    with catalog.engine.begin() as connection:
        statement = update(jobs).where(jobs.c.id == catalog_job_id).values(**values)
        connection.execute(statement)


def _run_with_metadata(
    catalog_job_id: int, handler: Any, payload: JsonObject
) -> JsonObject:
    _update_job(catalog_job_id, status=JobStatus.RUNNING, started_at=_utcnow())
    try:
        result = handler(payload)
    except Exception as exc:
        _update_job(
            catalog_job_id,
            status=JobStatus.FAILED,
            error=str(exc),
            completed_at=_utcnow(),
        )
        raise
    _update_job(
        catalog_job_id,
        status=JobStatus.SUCCEEDED,
        result=result,
        completed_at=_utcnow(),
    )
    return result


def _provider_from_payload(payload: Mapping[str, Any]) -> Any:
    provider_value = payload.get("provider") or get_settings().default_provider.value
    provider = str(provider_value).lower()
    if provider == "fake":
        return FakeMarketDataProvider()
    if provider == Provider.MASSIVE.value:
        return MassiveMarketDataProvider()
    msg = f"unsupported ingestion provider: {provider}"
    raise ValueError(msg)


def _handle_ingestion(payload: JsonObject) -> JsonObject:
    service = IngestionService(
        provider=_provider_from_payload(payload),
        catalog=_catalog(),
        data_root=get_settings().data_lake_root,
    )
    result = service.ingest_market_data(
        symbols=payload["symbols"],
        data_types=payload["data_types"],
        start=payload["start"],
        end=payload["end"],
        asset_class=payload.get("asset_class", "equity"),
        resolution=payload.get("resolution"),
    )
    return {
        "requested": result.requested,
        "skipped": result.skipped,
        "written": result.written,
        "artifacts": [path.as_posix() for path in result.artifacts],
        "manifests": [path.as_posix() for path in result.manifests],
    }


def _handle_training(payload: JsonObject) -> JsonObject:
    """Record a lightweight training job until model training is implemented."""

    return {"status": "recorded", "job": "training", "payload": payload}


def _handle_backtest(payload: JsonObject) -> JsonObject:
    """Record a lightweight backtest job until backtesting is implemented."""

    return {"status": "recorded", "job": "backtest", "payload": payload}


def run_ingestion_job(catalog_job_id: int, payload: JsonObject) -> JsonObject:
    """Execute an ingestion job and persist lifecycle metadata."""

    return _run_with_metadata(catalog_job_id, _handle_ingestion, payload)


def run_training_job(catalog_job_id: int, payload: JsonObject) -> JsonObject:
    """Execute a training job placeholder and persist lifecycle metadata."""

    return _run_with_metadata(catalog_job_id, _handle_training, payload)


def run_backtest_job(catalog_job_id: int, payload: JsonObject) -> JsonObject:
    """Execute a backtest job placeholder and persist lifecycle metadata."""

    return _run_with_metadata(catalog_job_id, _handle_backtest, payload)


__all__ = ["run_backtest_job", "run_ingestion_job", "run_training_job"]
