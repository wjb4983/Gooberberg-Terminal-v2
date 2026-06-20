"""Data lake path builders for market, derived, snapshot, and artifact data."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from quant_platform.common.enums import AssetClass, DataType, Provider
from quant_platform.common.paths import expand_path
from quant_platform.common.time import parse_date

PathLike = str | Path
DateLike = str | date | datetime


def _value(value: str | AssetClass | DataType | Provider) -> str:
    """Return a stable path value for enum-like inputs."""

    return value.value if hasattr(value, "value") else str(value)


def _clean(value: str | AssetClass | DataType | Provider) -> str:
    """Normalize a required path segment."""

    text = _value(value).strip().lower().replace(" ", "_")
    if not text:
        msg = "path segments cannot be empty"
        raise ValueError(msg)
    if "/" in text or "\\" in text:
        msg = f"path segment cannot contain separators: {value!r}"
        raise ValueError(msg)
    return text


def _symbol(value: str) -> str:
    """Normalize ticker, underlying, and contract identifiers for partition values."""

    text = value.strip().upper()
    if not text:
        msg = "symbols and contracts cannot be empty"
        raise ValueError(msg)
    if "/" in text or "\\" in text:
        msg = f"symbol cannot contain separators: {value!r}"
        raise ValueError(msg)
    return text


def date_partitions(value: DateLike) -> tuple[str, str, str]:
    """Return ``year=YYYY/month=MM/day=DD`` partition segments for a date."""

    parsed = parse_date(value)
    return (f"year={parsed:%Y}", f"month={parsed:%m}", f"day={parsed:%d}")


def partition(name: str, value: str | AssetClass | DataType | Provider) -> str:
    """Build a Hive-style partition segment."""

    return f"{_clean(name)}={_clean(value)}"


def symbol_partition(name: str, value: str) -> str:
    """Build a Hive-style partition segment whose value is an uppercase market symbol."""

    return f"{_clean(name)}={_symbol(value)}"


@dataclass(frozen=True)
class DataLakePaths:
    """Generate deterministic data lake paths rooted at ``root``."""

    root: PathLike

    @property
    def base(self) -> Path:
        """Expanded absolute data lake root."""

        return expand_path(self.root)

    def path(self, *parts: str) -> Path:
        """Build a path below the data lake root."""

        return self.base.joinpath(*parts)

    def market_data(
        self,
        *,
        provider: str | Provider,
        asset_class: str | AssetClass,
        data_type: str | DataType,
        symbol: str,
        date: DateLike | None = None,
        resolution: str | None = None,
    ) -> Path:
        """Build a path for equity or ETF market data."""

        parts = [
            "market_data",
            partition("provider", provider),
            partition("asset_class", asset_class),
            partition("data_type", data_type),
            symbol_partition("symbol", symbol),
        ]
        if resolution is not None:
            parts.append(partition("resolution", resolution))
        if date is not None:
            parts.extend(date_partitions(date))
        return self.path(*parts)

    def equity(
        self,
        *,
        provider: str | Provider,
        data_type: str | DataType,
        symbol: str,
        date: DateLike | None = None,
        resolution: str | None = None,
    ) -> Path:
        """Build a path for equity market data."""

        return self.market_data(
            provider=provider,
            asset_class=AssetClass.EQUITY,
            data_type=data_type,
            symbol=symbol,
            date=date,
            resolution=resolution,
        )

    def etf(
        self,
        *,
        provider: str | Provider,
        data_type: str | DataType,
        symbol: str,
        date: DateLike | None = None,
        resolution: str | None = None,
    ) -> Path:
        """Build a path for ETF market data."""

        return self.market_data(
            provider=provider,
            asset_class=AssetClass.ETF,
            data_type=data_type,
            symbol=symbol,
            date=date,
            resolution=resolution,
        )

    def option_chain(
        self, *, provider: str | Provider, underlying: str, date: DateLike | None = None
    ) -> Path:
        """Build a path for option chain snapshots for an underlying symbol."""

        parts = [
            "market_data",
            partition("provider", provider),
            partition("asset_class", AssetClass.OPTION),
            partition("data_type", "chains"),
            symbol_partition("underlying", underlying),
        ]
        if date is not None:
            parts.extend(date_partitions(date))
        return self.path(*parts)

    def option_contract(
        self,
        *,
        provider: str | Provider,
        data_type: str | DataType,
        underlying: str,
        contract: str,
        date: DateLike | None = None,
        resolution: str | None = None,
    ) -> Path:
        """Build a path for option contract trades, quotes, or bars."""

        parts = [
            "market_data",
            partition("provider", provider),
            partition("asset_class", AssetClass.OPTION),
            partition("data_type", data_type),
            symbol_partition("underlying", underlying),
            symbol_partition("contract", contract),
        ]
        if resolution is not None:
            parts.append(partition("resolution", resolution))
        if date is not None:
            parts.extend(date_partitions(date))
        return self.path(*parts)

    def snapshot(
        self,
        *,
        provider: str | Provider,
        asset_class: str | AssetClass,
        symbol: str,
        date: DateLike | None = None,
    ) -> Path:
        """Build a path for provider snapshots."""

        parts = [
            "snapshots",
            partition("provider", provider),
            partition("asset_class", asset_class),
            symbol_partition("symbol", symbol),
        ]
        if date is not None:
            parts.extend(date_partitions(date))
        return self.path(*parts)

    def derived(
        self,
        *,
        dataset: str,
        asset_class: str | AssetClass | None = None,
        symbol: str | None = None,
        date: DateLike | None = None,
        version: str | None = None,
    ) -> Path:
        """Build a path for derived datasets such as features or analytics."""

        parts = ["derived", partition("dataset", dataset)]
        if version is not None:
            parts.append(partition("version", version))
        if asset_class is not None:
            parts.append(partition("asset_class", asset_class))
        if symbol is not None:
            parts.append(symbol_partition("symbol", symbol))
        if date is not None:
            parts.extend(date_partitions(date))
        return self.path(*parts)

    def artifact(
        self,
        *,
        artifact_type: str,
        name: str,
        version: str | None = None,
        run_id: str | None = None,
    ) -> Path:
        """Build a path for model, report, and export artifacts."""

        parts = [
            "artifacts",
            partition("artifact_type", artifact_type),
            partition("name", name),
        ]
        if version is not None:
            parts.append(partition("version", version))
        if run_id is not None:
            parts.append(partition("run_id", run_id))
        return self.path(*parts)


def join_filename(path: Path, filename: str) -> Path:
    """Append a validated filename to a generated storage path."""

    if not filename or "/" in filename or "\\" in filename:
        msg = f"invalid filename: {filename!r}"
        raise ValueError(msg)
    return path / filename


__all__ = [
    "DataLakePaths",
    "DateLike",
    "PathLike",
    "date_partitions",
    "join_filename",
    "partition",
    "symbol_partition",
]
