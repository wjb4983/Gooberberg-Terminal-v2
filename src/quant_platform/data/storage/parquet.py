"""Parquet-specific storage path helpers."""

from __future__ import annotations

from pathlib import Path

from quant_platform.common.enums import AssetClass, DataType, Provider
from quant_platform.data.storage.lake import (
    DataLakePaths,
    DateLike,
    join_filename,
)

DEFAULT_PARQUET_FILENAME = "part-00000.parquet"


class ParquetPaths(DataLakePaths):
    """Generate data lake paths that terminate in a Parquet filename."""

    def parquet(
        self, directory: Path, filename: str = DEFAULT_PARQUET_FILENAME
    ) -> Path:
        """Append a Parquet filename to a generated directory path."""

        if not filename.endswith(".parquet"):
            msg = f"parquet filename must end with .parquet: {filename!r}"
            raise ValueError(msg)
        return join_filename(directory, filename)

    def market_data_file(
        self,
        *,
        provider: str | Provider,
        asset_class: str | AssetClass,
        data_type: str | DataType,
        symbol: str,
        date: DateLike | None = None,
        resolution: str | None = None,
        filename: str = DEFAULT_PARQUET_FILENAME,
    ) -> Path:
        """Build a Parquet file path for equity or ETF market data."""

        return self.parquet(
            self.market_data(
                provider=provider,
                asset_class=asset_class,
                data_type=data_type,
                symbol=symbol,
                date=date,
                resolution=resolution,
            ),
            filename,
        )

    def equity_file(
        self,
        *,
        provider: str | Provider,
        data_type: str | DataType,
        symbol: str,
        date: DateLike | None = None,
        resolution: str | None = None,
        filename: str = DEFAULT_PARQUET_FILENAME,
    ) -> Path:
        """Build a Parquet file path for equity data."""

        return self.parquet(
            self.equity(
                provider=provider,
                data_type=data_type,
                symbol=symbol,
                date=date,
                resolution=resolution,
            ),
            filename,
        )

    def etf_file(
        self,
        *,
        provider: str | Provider,
        data_type: str | DataType,
        symbol: str,
        date: DateLike | None = None,
        resolution: str | None = None,
        filename: str = DEFAULT_PARQUET_FILENAME,
    ) -> Path:
        """Build a Parquet file path for ETF data."""

        return self.parquet(
            self.etf(
                provider=provider,
                data_type=data_type,
                symbol=symbol,
                date=date,
                resolution=resolution,
            ),
            filename,
        )

    def option_chain_file(
        self,
        *,
        provider: str | Provider,
        underlying: str,
        date: DateLike | None = None,
        filename: str = DEFAULT_PARQUET_FILENAME,
    ) -> Path:
        """Build a Parquet file path for option chains."""

        return self.parquet(
            self.option_chain(provider=provider, underlying=underlying, date=date),
            filename,
        )

    def option_contract_file(
        self,
        *,
        provider: str | Provider,
        data_type: str | DataType,
        underlying: str,
        contract: str,
        date: DateLike | None = None,
        resolution: str | None = None,
        filename: str = DEFAULT_PARQUET_FILENAME,
    ) -> Path:
        """Build a Parquet file path for option contract trades, quotes, or bars."""

        return self.parquet(
            self.option_contract(
                provider=provider,
                data_type=data_type,
                underlying=underlying,
                contract=contract,
                date=date,
                resolution=resolution,
            ),
            filename,
        )

    def snapshot_file(
        self,
        *,
        provider: str | Provider,
        asset_class: str | AssetClass,
        symbol: str,
        date: DateLike | None = None,
        filename: str = DEFAULT_PARQUET_FILENAME,
    ) -> Path:
        """Build a Parquet file path for snapshots."""

        return self.parquet(
            self.snapshot(
                provider=provider, asset_class=asset_class, symbol=symbol, date=date
            ),
            filename,
        )

    def derived_file(
        self,
        *,
        dataset: str,
        asset_class: str | AssetClass | None = None,
        symbol: str | None = None,
        date: DateLike | None = None,
        version: str | None = None,
        filename: str = DEFAULT_PARQUET_FILENAME,
    ) -> Path:
        """Build a Parquet file path for derived datasets."""

        return self.parquet(
            self.derived(
                dataset=dataset,
                asset_class=asset_class,
                symbol=symbol,
                date=date,
                version=version,
            ),
            filename,
        )

    def artifact_file(
        self,
        *,
        artifact_type: str,
        name: str,
        version: str | None = None,
        run_id: str | None = None,
        filename: str = DEFAULT_PARQUET_FILENAME,
    ) -> Path:
        """Build a Parquet file path for artifacts."""

        return self.parquet(
            self.artifact(
                artifact_type=artifact_type, name=name, version=version, run_id=run_id
            ),
            filename,
        )


__all__ = ["DEFAULT_PARQUET_FILENAME", "ParquetPaths"]
