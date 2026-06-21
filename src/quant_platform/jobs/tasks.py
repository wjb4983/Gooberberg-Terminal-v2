"""RQ task functions for platform background jobs."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select, update

from quant_platform.common.enums import JobStatus, Provider
from quant_platform.config.settings import get_settings
from quant_platform.data.ingestion.service import IngestionService
from quant_platform.data.providers import (
    FakeMarketDataProvider,
    MassiveMarketDataProvider,
)
from quant_platform.data.storage.catalog import MetadataCatalog, experiments, jobs
from quant_platform.training.runner import run_training
from quant_platform.training.schemas import TrainingConfig

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


def _append_job_log(
    catalog_job_id: int,
    message: str,
    *,
    level: str = "info",
    metadata: Mapping[str, Any] | None = None,
) -> None:
    catalog = _catalog()
    catalog.insert_row(
        "job_logs",
        {
            "job_id": catalog_job_id,
            "level": level,
            "message": message,
            "metadata": dict(metadata or {}),
        },
    )


def _run_with_metadata(
    catalog_job_id: int, handler: Any, payload: JsonObject
) -> JsonObject:
    _update_job(catalog_job_id, status=JobStatus.RUNNING, started_at=_utcnow())
    _append_job_log(
        catalog_job_id,
        "Job started.",
        metadata={
            "symbols": payload.get("symbols", []),
            "data_types": payload.get("data_types", []),
        },
    )
    try:
        result = handler({**payload, "catalog_job_id": catalog_job_id})
    except Exception as exc:
        _append_job_log(catalog_job_id, f"Job failed: {exc}", level="error")
        _update_job(
            catalog_job_id,
            status=JobStatus.FAILED,
            error=str(exc),
            completed_at=_utcnow(),
        )
        raise
    _append_job_log(catalog_job_id, "Job completed.", metadata={"result": result})
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
    symbols = payload["symbols"]
    data_types = payload["data_types"]
    _append_job_log(
        int(payload["catalog_job_id"]),
        "Preparing ingestion request.",
        metadata={
            "provider": (
                payload.get("provider") or get_settings().default_provider.value
            ),
            "symbols": symbols,
            "data_types": data_types,
            "start": payload["start"],
            "end": payload["end"],
        },
    )
    service = IngestionService(
        provider=_provider_from_payload(payload),
        catalog=_catalog(),
        data_root=get_settings().data_lake_root,
    )
    result = service.ingest_market_data(
        symbols=symbols,
        data_types=data_types,
        start=payload["start"],
        end=payload["end"],
        asset_class=payload.get("asset_class", "equity"),
        resolution=payload.get("resolution"),
    )
    _append_job_log(
        int(payload["catalog_job_id"]),
        "Ingestion wrote artifacts.",
        metadata={"written": result.written, "skipped": result.skipped},
    )
    return {
        "requested": result.requested,
        "skipped": result.skipped,
        "written": result.written,
        "artifacts": [path.as_posix() for path in result.artifacts],
        "manifests": [path.as_posix() for path in result.manifests],
    }


def _experiment_metadata(
    catalog: MetadataCatalog, experiment_id: int
) -> dict[str, Any]:
    with catalog.engine.connect() as connection:
        row = (
            connection.execute(
                select(experiments.c.metadata).where(experiments.c.id == experiment_id)
            )
            .mappings()
            .first()
        )
    if row is None:
        raise ValueError(f"experiment does not exist: {experiment_id}")
    return dict(row.get("metadata") or {})


def _update_experiment(
    experiment_id: int,
    *,
    status: str,
    metadata: Mapping[str, Any] | None = None,
    started_at: datetime | None = None,
    completed_at: datetime | None = None,
) -> None:
    catalog = _catalog()
    values: dict[str, Any] = {"status": status}
    if metadata is not None:
        values["metadata"] = dict(metadata)
    if started_at is not None:
        values["started_at"] = started_at
    if completed_at is not None:
        values["completed_at"] = completed_at
    updated_rows = catalog.update_row("experiments", experiment_id, values)
    if updated_rows == 0:
        raise ValueError(f"experiment does not exist: {experiment_id}")


def _training_config_from_payload(payload: Mapping[str, Any]) -> TrainingConfig:
    config_payload = dict(payload)
    config_payload.pop("catalog_job_id", None)
    config_payload.pop("rq_job_id", None)
    if "split" in config_payload and "date_split" not in config_payload:
        config_payload["date_split"] = config_payload.pop("split")
    training = config_payload.pop("training", None)
    if isinstance(training, Mapping):
        config_payload.update(dict(training))
    allowed_fields = set(TrainingConfig.model_fields)
    config_payload = {
        key: value for key, value in config_payload.items() if key in allowed_fields
    }
    return TrainingConfig.model_validate(config_payload)


def _artifact_links(result: Any) -> dict[str, str]:
    links: dict[str, str] = {}
    artifact_dir = getattr(result, "artifact_dir", None)
    if artifact_dir is not None:
        links["artifact_dir"] = str(artifact_dir)
    manifest = getattr(result, "manifest", None)
    manifest_files = getattr(manifest, "files", None)
    if isinstance(manifest_files, Mapping):
        links.update({str(key): str(value) for key, value in manifest_files.items()})
    return links


def _training_result_payload(result: Any) -> JsonObject:
    if hasattr(result, "model_dump"):
        return dict(result.model_dump(mode="json"))
    if isinstance(result, Mapping):
        return dict(result)
    return {"result": str(result)}


def _handle_training(payload: JsonObject) -> JsonObject:
    """Run model training and persist linked experiment lifecycle metadata."""

    experiment_id = int(payload["experiment_id"])
    started_at = _utcnow()
    current_metadata = _experiment_metadata(_catalog(), experiment_id)
    _update_experiment(
        experiment_id,
        status="running",
        metadata=current_metadata,
        started_at=started_at,
    )
    try:
        result = run_training(_training_config_from_payload(payload))
    except Exception as exc:
        failed_metadata = {
            **_experiment_metadata(_catalog(), experiment_id),
            "error": str(exc)[:500],
        }
        _update_experiment(
            experiment_id,
            status="failed",
            metadata=failed_metadata,
            completed_at=_utcnow(),
        )
        raise

    artifacts = _artifact_links(result)
    succeeded_metadata = {
        **_experiment_metadata(_catalog(), experiment_id),
        "artifacts": artifacts,
        "artifact_links": artifacts,
    }
    _update_experiment(
        experiment_id,
        status="succeeded",
        metadata=succeeded_metadata,
        completed_at=_utcnow(),
    )
    return _training_result_payload(result)


def _handle_backtest(payload: JsonObject) -> JsonObject:
    """Record a lightweight backtest job until backtesting is implemented."""

    return {"status": "recorded", "job": "backtest", "payload": payload}


def run_ingestion_job(catalog_job_id: int, payload: JsonObject) -> JsonObject:
    """Execute an ingestion job and persist lifecycle metadata."""

    return _run_with_metadata(catalog_job_id, _handle_ingestion, payload)


def run_training_job(catalog_job_id: int, payload: JsonObject) -> JsonObject:
    """Execute a training job and persist lifecycle metadata."""

    return _run_with_metadata(catalog_job_id, _handle_training, payload)


def run_backtest_job(catalog_job_id: int, payload: JsonObject) -> JsonObject:
    """Execute a backtest job placeholder and persist lifecycle metadata."""

    return _run_with_metadata(catalog_job_id, _handle_backtest, payload)


__all__ = ["run_backtest_job", "run_ingestion_job", "run_training_job"]
