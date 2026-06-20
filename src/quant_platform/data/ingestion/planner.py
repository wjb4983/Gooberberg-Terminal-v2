"""Plan market-data ingestion by subtracting existing catalog coverage."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date, datetime, timedelta

from quant_platform.common.enums import AssetClass, DataType
from quant_platform.data.providers.base import DateLike


def normalize_date(value: DateLike) -> date:
    """Normalize provider date inputs to a calendar date."""

    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return datetime.fromisoformat(value).date()


def normalize_symbol(value: str) -> str:
    """Normalize a market symbol for catalog and storage partitions."""

    symbol = value.strip().upper()
    if not symbol:
        msg = "symbol cannot be empty"
        raise ValueError(msg)
    return symbol


def normalize_value(value: str | AssetClass | DataType) -> str:
    """Normalize enum-like identifiers for catalog comparisons."""

    return value.value if hasattr(value, "value") else str(value)


def date_range(start: DateLike, end: DateLike) -> tuple[date, ...]:
    """Return an inclusive tuple of dates from ``start`` through ``end``."""

    start_date = normalize_date(start)
    end_date = normalize_date(end)
    if end_date < start_date:
        msg = "end date must be on or after start date"
        raise ValueError(msg)
    days = (end_date - start_date).days
    return tuple(start_date + timedelta(days=offset) for offset in range(days + 1))


@dataclass(frozen=True)
class IngestionPartition:
    """A single provider/symbol/data-type/date partition to ingest."""

    provider: str
    asset_class: str
    data_type: str
    symbol: str
    date: date
    dataset: str
    resolution: str | None = None

    @property
    def key(self) -> tuple[str, str, str, str, date, str | None]:
        """Stable dedupe key for the partition."""

        return (
            self.provider,
            self.asset_class,
            self.data_type,
            self.symbol,
            self.date,
            self.resolution,
        )


def dataset_name(asset_class: str | AssetClass, data_type: str | DataType) -> str:
    """Build the catalog dataset name for a market-data partition."""

    return f"market_data.{normalize_value(asset_class)}.{normalize_value(data_type)}"


@dataclass(frozen=True)
class IngestionRequest:
    """High-level request for market-data ingestion."""

    provider: str
    symbols: tuple[str, ...]
    data_types: tuple[str, ...]
    start: date
    end: date
    asset_class: str = AssetClass.EQUITY.value
    resolution: str | None = None

    @classmethod
    def create(
        cls,
        *,
        provider: str,
        symbols: Iterable[str],
        data_types: Iterable[str | DataType],
        start: DateLike,
        end: DateLike,
        asset_class: str | AssetClass = AssetClass.EQUITY,
        resolution: str | None = None,
    ) -> IngestionRequest:
        """Create a normalized ingestion request."""

        normalized_symbols = tuple(normalize_symbol(symbol) for symbol in symbols)
        normalized_types = tuple(normalize_value(data_type) for data_type in data_types)
        if not normalized_symbols:
            msg = "at least one symbol is required"
            raise ValueError(msg)
        if not normalized_types:
            msg = "at least one data type is required"
            raise ValueError(msg)
        return cls(
            provider=provider.strip().lower(),
            symbols=normalized_symbols,
            data_types=normalized_types,
            start=normalize_date(start),
            end=normalize_date(end),
            asset_class=normalize_value(asset_class),
            resolution=resolution,
        )


def all_requested_partitions(
    request: IngestionRequest,
) -> tuple[IngestionPartition, ...]:
    """Expand a request into date/symbol/data-type partitions."""

    return tuple(
        IngestionPartition(
            provider=request.provider,
            asset_class=request.asset_class,
            data_type=data_type,
            symbol=symbol,
            date=day,
            dataset=dataset_name(request.asset_class, data_type),
            resolution=request.resolution if data_type == DataType.BARS.value else None,
        )
        for symbol in request.symbols
        for data_type in request.data_types
        for day in date_range(request.start, request.end)
    )


__all__ = [
    "IngestionPartition",
    "IngestionRequest",
    "all_requested_partitions",
    "dataset_name",
    "date_range",
    "normalize_date",
    "normalize_symbol",
    "normalize_value",
]
