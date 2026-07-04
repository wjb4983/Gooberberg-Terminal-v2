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

WORKFLOW_INTENT_OPTIONS: tuple[tuple[str, str], ...] = (
    ("Supervised forecasting", "supervised_forecasting"),
    ("Learned regime switching", "learned_regime_switching"),
    ("Backtesting/allocation", "backtesting_allocation"),
    ("Raw market data", "raw_market_data"),
)

_REGIME_WORKFLOW_INTENT = "learned_regime_switching"

_PROVIDER_OPTIONS_BY_ASSET_CLASS: dict[str, tuple[str, ...]] = {
    AssetClass.EQUITY.value: (
        Provider.MASSIVE.value,
        Provider.POLYGON.value,
        Provider.ALPACA.value,
        Provider.YAHOO.value,
    ),
    AssetClass.ETF.value: (
        Provider.MASSIVE.value,
        Provider.POLYGON.value,
        Provider.ALPACA.value,
        Provider.YAHOO.value,
    ),
    AssetClass.INDEX.value: (
        Provider.POLYGON.value,
        Provider.YAHOO.value,
        Provider.MASSIVE.value,
    ),
    AssetClass.FOREX.value: (Provider.POLYGON.value, Provider.MASSIVE.value),
    AssetClass.CRYPTO.value: (
        Provider.POLYGON.value,
        Provider.MASSIVE.value,
        Provider.ALPACA.value,
    ),
    AssetClass.FUTURE.value: (Provider.POLYGON.value, Provider.MASSIVE.value),
    AssetClass.OPTION.value: (Provider.POLYGON.value, Provider.MASSIVE.value),
}

_DATA_TYPE_OPTIONS_BY_ASSET_CLASS: dict[str, tuple[str, ...]] = {
    AssetClass.EQUITY.value: tuple(data_type.value for data_type in DataType),
    AssetClass.ETF.value: tuple(data_type.value for data_type in DataType),
    AssetClass.INDEX.value: (
        DataType.BARS.value,
        DataType.DAILY_BARS.value,
        DataType.NEWS.value,
    ),
    AssetClass.FOREX.value: (
        DataType.TRADES.value,
        DataType.QUOTES.value,
        DataType.BARS.value,
        DataType.DAILY_BARS.value,
        DataType.NEWS.value,
    ),
    AssetClass.CRYPTO.value: (
        DataType.TRADES.value,
        DataType.QUOTES.value,
        DataType.BARS.value,
        DataType.DAILY_BARS.value,
        DataType.NEWS.value,
    ),
    AssetClass.FUTURE.value: (
        DataType.TRADES.value,
        DataType.QUOTES.value,
        DataType.BARS.value,
        DataType.DAILY_BARS.value,
        DataType.NEWS.value,
    ),
    AssetClass.OPTION.value: (
        DataType.TRADES.value,
        DataType.QUOTES.value,
        DataType.BARS.value,
        DataType.DAILY_BARS.value,
    ),
}

_REGIME_DATA_TYPE_OPTIONS = (DataType.DAILY_BARS.value, DataType.BARS.value)
_REGIME_DEFAULT_SYMBOLS: dict[str, tuple[str, ...]] = {
    AssetClass.EQUITY.value: ("SPY", "QQQ", "IWM", "DIA", "VTI"),
    AssetClass.ETF.value: ("SPY", "QQQ", "IWM", "TLT", "GLD"),
    AssetClass.INDEX.value: ("SPY", "QQQ", "IWM", "DIA", "VIXY"),
    AssetClass.FOREX.value: ("EURUSD", "USDJPY", "GBPUSD", "DXY"),
    AssetClass.CRYPTO.value: ("BTCUSD", "ETHUSD", "SOLUSD"),
}
_DEFAULT_SYMBOLS: dict[str, tuple[str, ...]] = {
    AssetClass.EQUITY.value: ("AAPL", "MSFT", "SPY"),
    AssetClass.ETF.value: ("SPY", "QQQ", "IWM"),
    AssetClass.INDEX.value: ("SPY", "QQQ", "DIA"),
    AssetClass.FOREX.value: ("EURUSD", "USDJPY", "GBPUSD"),
    AssetClass.CRYPTO.value: ("BTCUSD", "ETHUSD", "SOLUSD"),
    AssetClass.FUTURE.value: ("ES", "NQ", "CL"),
    AssetClass.OPTION.value: ("SPY", "QQQ", "AAPL"),
}


def workflow_intent_options() -> list[tuple[str, str]]:
    """Return display labels and stable values for workflow intent controls."""

    return list(WORKFLOW_INTENT_OPTIONS)


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


def provider_options_for_asset_class(asset_class: str) -> list[str]:
    """Return provider options relevant to the selected asset class."""

    return list(
        _PROVIDER_OPTIONS_BY_ASSET_CLASS.get(asset_class, tuple(provider_options()))
    )


def data_type_options() -> list[str]:
    """Return supported data type values for UI controls."""

    return [data_type.value for data_type in DataType]


def data_type_options_for_asset_class(
    asset_class: str, workflow_intent: str | None = None
) -> list[str]:
    """Return data types relevant to the selected asset class and workflow intent."""

    asset_options = _DATA_TYPE_OPTIONS_BY_ASSET_CLASS.get(
        asset_class, tuple(data_type_options())
    )
    if workflow_intent == _REGIME_WORKFLOW_INTENT:
        return [
            option for option in _REGIME_DATA_TYPE_OPTIONS if option in asset_options
        ]
    return list(asset_options)


def default_resolution_for_context(
    asset_class: str, workflow_intent: str | None = None
) -> str:
    """Return the suggested resolution for the selected dataset context."""

    if workflow_intent == _REGIME_WORKFLOW_INTENT:
        return "1d"
    if (
        asset_class in {AssetClass.CRYPTO.value, AssetClass.FOREX.value}
        and workflow_intent == "raw_market_data"
    ):
        return "1h"
    return "1d"


def default_symbols_for_context(
    asset_class: str, workflow_intent: str | None = None
) -> list[str]:
    """Return suggested symbols for the selected asset class and workflow intent."""

    defaults = (
        _REGIME_DEFAULT_SYMBOLS
        if workflow_intent == _REGIME_WORKFLOW_INTENT
        else _DEFAULT_SYMBOLS
    )
    return list(defaults.get(asset_class, _DEFAULT_SYMBOLS[AssetClass.EQUITY.value]))


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
    workflow_intent: str | None = None,
    regime_source: str | None = None,
    target_style: str | None = None,
    preferred_task_type: str | None = None,
    feature_requirements: Any | None = None,
    labeling_intent: str | None = None,
    metadata: dict[str, Any] | None = None,
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
        metadata=_definition_metadata(
            asset_class=asset_class,
            workflow_intent=workflow_intent,
            regime_source=regime_source,
            target_style=target_style,
            preferred_task_type=preferred_task_type,
            feature_requirements=feature_requirements,
            labeling_intent=labeling_intent,
            extra_metadata=metadata,
        ),
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
    provider_options_allowed: list[str] | None = None,
    data_type_options_allowed: list[str] | None = None,
) -> tuple[ValidationCheck, ...]:
    """Return basic validation checks without raising page-level exceptions."""

    allowed_providers = provider_options_allowed or provider_options_for_asset_class(
        asset_class
    )
    allowed_data_types = data_type_options_allowed or data_type_options_for_asset_class(
        asset_class
    )
    unsupported_data_types = sorted(set(data_types) - set(allowed_data_types))
    checks = [
        ValidationCheck("Dataset name", bool(name.strip()), "Name is required."),
        ValidationCheck(
            "Asset universe",
            bool(asset_universe),
            "At least one symbol or selector is required.",
        ),
        ValidationCheck(
            "Data types",
            bool(data_types) and not unsupported_data_types,
            "At least one relevant data type is required."
            if not data_types
            else f"Unsupported for this context: {', '.join(unsupported_data_types)}.",
        ),
        ValidationCheck(
            "Date range", start <= end, "Start date must be on or before end date."
        ),
        ValidationCheck(
            "Provider",
            provider in allowed_providers,
            "Provider must be supported for this asset class.",
        ),
        ValidationCheck(
            "Asset class",
            asset_class in asset_class_options(),
            "Asset class must be supported.",
        ),
    ]
    return tuple(checks)


def _definition_metadata(
    *,
    asset_class: str,
    workflow_intent: str | None,
    regime_source: str | None,
    target_style: str | None,
    preferred_task_type: str | None,
    feature_requirements: Any | None,
    labeling_intent: str | None,
    extra_metadata: dict[str, Any] | None,
) -> dict[str, Any]:
    resolved_metadata = dict(extra_metadata or {})
    resolved_metadata["asset_class"] = AssetClass(asset_class).value
    optional_metadata = {
        "workflow_intent": workflow_intent,
        "regime_source": regime_source,
        "target_style": target_style,
        "preferred_task_type": preferred_task_type,
        "feature_requirements": feature_requirements,
        "labeling_intent": labeling_intent,
    }
    for key, value in optional_metadata.items():
        if value is not None:
            resolved_metadata[key] = value
    if workflow_intent == _REGIME_WORKFLOW_INTENT:
        resolved_metadata.setdefault("regime_source", "real_market_data")
        resolved_metadata.setdefault("preferred_task_type", "regime_classification")
    return resolved_metadata


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
        "metadata": dict(definition.metadata),
    }
    job_id = resolved_catalog.insert_row(
        "jobs",
        {
            "job_type": TaskType.INGEST.value,
            "status": JobStatus.QUEUED.value,
            "payload": payload,
        },
    )
    resolved_catalog.insert_row(
        "job_logs",
        {
            "job_id": job_id,
            "level": "info",
            "message": "Queued dataset ingestion job.",
            "metadata": payload,
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
