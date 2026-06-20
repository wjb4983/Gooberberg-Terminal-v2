"""Dataset page orchestration helpers for Streamlit and API surfaces."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from sqlalchemy import select

from quant_platform.common.enums import (
    AssetClass,
    DataType,
    JobStatus,
    Provider,
    TaskType,
)
from quant_platform.config import get_settings
from quant_platform.data.ingestion.coverage import CoverageStore
from quant_platform.data.ingestion.planner import (
    IngestionPartition,
    IngestionRequest,
    all_requested_partitions,
)
from quant_platform.data.storage.catalog import MetadataCatalog, jobs
from quant_platform.datasets.registry import DatasetRegistry
from quant_platform.datasets.schemas import DatasetDefinition


@dataclass(frozen=True)
class ValidationCheck:
    """One user-facing validation check for a dataset ingestion request."""

    name: str
    passed: bool
    message: str


@dataclass(frozen=True)
class CoveragePreview:
    """Summary and table-ready rows for requested and missing coverage."""

    requested_count: int
    missing_count: int
    rows: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class QueuedDatasetIngestion:
    """Catalog metadata returned after queueing a dataset ingestion placeholder."""

    dataset_id: int
    job_id: int
    status: str
    payload: dict[str, Any]


def provider_options() -> list[str]:
    """Return supported provider values for UI controls."""

    return [provider.value for provider in Provider]


def data_type_options() -> list[str]:
    """Return supported data type values for UI controls."""

    return [data_type.value for data_type in DataType]


def asset_class_options() -> list[str]:
    """Return supported asset class values for UI controls."""

    return [asset_class.value for asset_class in AssetClass]


def parse_asset_universe(raw_symbols: str) -> list[str]:
    """Parse comma/newline separated symbols while preserving user order."""

    candidates = raw_symbols.replace("\n", ",").split(",")
    symbols: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        symbol = candidate.strip().upper()
        if symbol and symbol not in seen:
            symbols.append(symbol)
            seen.add(symbol)
    return symbols


def build_definition(
    *,
    name: str,
    version: str,
    provider: str,
    asset_universe: list[str],
    data_types: list[str],
    start: date,
    end: date,
    asset_class: str,
    resolution: str | None,
    description: str | None = None,
) -> DatasetDefinition:
    """Build and validate a logical dataset definition from page inputs."""

    return DatasetDefinition(
        name=name.strip(),
        version=version.strip() or "1",
        provider=Provider(provider),
        asset_universe=asset_universe,
        data_types=[DataType(data_type) for data_type in data_types],
        resolution=resolution.strip() if resolution and resolution.strip() else None,
        date_range={"start": start, "end": end},
        description=(
            description.strip() if description and description.strip() else None
        ),
        metadata={"asset_class": AssetClass(asset_class).value},
    )


def validate_definition_inputs(
    *,
    name: str,
    asset_universe: list[str],
    data_types: list[str],
    start: date,
    end: date,
    provider: str,
    asset_class: str,
) -> tuple[ValidationCheck, ...]:
    """Return basic validation checks without raising page-level exceptions."""

    checks = [
        ValidationCheck("Dataset name", bool(name.strip()), "Name is required."),
        ValidationCheck(
            "Asset universe",
            bool(asset_universe),
            "At least one symbol or selector is required.",
        ),
        ValidationCheck(
            "Data types", bool(data_types), "At least one data type is required."
        ),
        ValidationCheck(
            "Date range", start <= end, "Start date must be on or before end date."
        ),
        ValidationCheck(
            "Provider", provider in provider_options(), "Provider must be supported."
        ),
        ValidationCheck(
            "Asset class",
            asset_class in asset_class_options(),
            "Asset class must be supported.",
        ),
    ]
    return tuple(checks)


def checks_passed(checks: tuple[ValidationCheck, ...]) -> bool:
    """Return True when all validation checks pass."""

    return all(check.passed for check in checks)


def ingestion_request_from_definition(
    definition: DatasetDefinition,
) -> IngestionRequest:
    """Create an ingestion request from a logical dataset definition."""

    if definition.date_range.start is None or definition.date_range.end is None:
        msg = "dataset date range must include both start and end dates"
        raise ValueError(msg)
    return IngestionRequest.create(
        provider=definition.provider.value,
        symbols=definition.asset_universe,
        data_types=[data_type.value for data_type in definition.data_types],
        start=definition.date_range.start,
        end=definition.date_range.end,
        asset_class=str(
            definition.metadata.get("asset_class", AssetClass.EQUITY.value)
        ),
        resolution=definition.resolution,
    )


def preview_coverage(
    definition: DatasetDefinition, catalog: MetadataCatalog | None = None
) -> CoveragePreview:
    """Plan missing partitions for a dataset without pulling provider data."""

    resolved_catalog = catalog or MetadataCatalog(get_settings().catalog_db_path)
    request = ingestion_request_from_definition(definition)
    requested = all_requested_partitions(request)
    missing = CoverageStore(resolved_catalog).missing_partitions(requested)
    missing_keys = {partition.key for partition in missing}
    return CoveragePreview(
        requested_count=len(requested),
        missing_count=len(missing),
        rows=tuple(
            _partition_row(partition, partition.key in missing_keys)
            for partition in requested
        ),
    )


def register_dataset(
    definition: DatasetDefinition,
    *,
    mirror_config: bool = True,
    catalog_path: str | Path | None = None,
    config_dir: str | Path | None = None,
) -> int:
    """Persist a dataset definition in the metadata catalog."""

    registry = DatasetRegistry(
        catalog_path or get_settings().catalog_db_path,
        config_dir=config_dir,
    )
    return registry.register(definition, mirror_config=mirror_config, overwrite=True)


def queue_ingestion(
    definition: DatasetDefinition,
    *,
    dataset_id: int | None = None,
    catalog: MetadataCatalog | None = None,
    config_dir: str | Path | None = None,
) -> QueuedDatasetIngestion:
    """Queue an ingestion placeholder in the catalog for worker pickup."""

    resolved_catalog = catalog or MetadataCatalog(get_settings().catalog_db_path)
    resolved_catalog.create_all()
    resolved_dataset_id = dataset_id or register_dataset(
        definition,
        mirror_config=True,
        catalog_path=resolved_catalog.path,
        config_dir=config_dir,
    )
    request = ingestion_request_from_definition(definition)
    payload = {
        "dataset_id": resolved_dataset_id,
        "dataset_name": definition.name,
        "provider": request.provider,
        "symbols": list(request.symbols),
        "data_types": list(request.data_types),
        "start": request.start.isoformat(),
        "end": request.end.isoformat(),
        "asset_class": request.asset_class,
        "resolution": request.resolution,
    }
    job_id = resolved_catalog.insert_row(
        "jobs",
        {
            "job_type": TaskType.INGEST.value,
            "status": JobStatus.QUEUED.value,
            "payload": payload,
        },
    )
    return QueuedDatasetIngestion(
        dataset_id=resolved_dataset_id,
        job_id=job_id,
        status=JobStatus.QUEUED.value,
        payload=payload,
    )


def job_status_rows(limit: int = 25) -> list[dict[str, Any]]:
    """Return latest ingestion-like jobs formatted for display."""

    catalog = MetadataCatalog(get_settings().catalog_db_path)
    catalog.create_all()
    with catalog.engine.connect() as connection:
        rows = connection.execute(
            select(jobs)
            .where(jobs.c.job_type.in_([TaskType.INGEST.value, "ingestion"]))
            .order_by(jobs.c.created_at.desc())
            .limit(limit)
        ).mappings()
        return [_job_row(dict(row)) for row in rows]


def _partition_row(partition: IngestionPartition, missing: bool) -> dict[str, Any]:
    return {
        "provider": partition.provider,
        "asset_class": partition.asset_class,
        "data_type": partition.data_type,
        "symbol": partition.symbol,
        "date": partition.date.isoformat(),
        "dataset": partition.dataset,
        "resolution": partition.resolution or "",
        "status": "missing" if missing else "covered",
    }


def _job_row(row: dict[str, Any]) -> dict[str, Any]:
    payload = dict(row.get("payload") or {})
    return {
        "id": row["id"],
        "status": row["status"],
        "dataset": payload.get("dataset_name", ""),
        "provider": payload.get("provider", ""),
        "symbols": ", ".join(payload.get("symbols", [])),
        "data_types": ", ".join(payload.get("data_types", [])),
        "created_at": row.get("created_at"),
        "started_at": row.get("started_at"),
        "completed_at": row.get("completed_at"),
        "error": row.get("error"),
    }
