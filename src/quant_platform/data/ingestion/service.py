"""High-level ingestion service for provider-backed market data pulls."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import polars as pl

from quant_platform.common.enums import AssetClass, DataType
from quant_platform.data.ingestion.coverage import CoverageStore
from quant_platform.data.ingestion.manifests import (
    DEFAULT_MANIFEST_ROOT,
    ManifestWriter,
)
from quant_platform.data.ingestion.planner import (
    IngestionPartition,
    IngestionRequest,
    all_requested_partitions,
)
from quant_platform.data.providers.base import DateLike, MarketDataProvider
from quant_platform.data.storage.catalog import MetadataCatalog
from quant_platform.data.storage.parquet import ParquetPaths


@dataclass(frozen=True)
class IngestionResult:
    """Summary of an ingestion run."""

    requested: int
    skipped: int
    written: int
    artifacts: tuple[Path, ...]
    manifests: tuple[Path, ...]


class IngestionService:
    """Plan missing partitions, pull provider data, and append parquet artifacts."""

    def __init__(
        self,
        *,
        provider: MarketDataProvider,
        catalog: MetadataCatalog,
        data_root: str | Path = "data/lake",
        manifest_root: str | Path = DEFAULT_MANIFEST_ROOT,
    ) -> None:
        self.provider = provider
        self.catalog = catalog
        self.coverage = CoverageStore(catalog)
        self.manifests = ManifestWriter(catalog=catalog, root=manifest_root)
        self.paths = ParquetPaths(data_root)

    def ingest_market_data(
        self,
        *,
        symbols: tuple[str, ...] | list[str],
        data_types: tuple[str | DataType, ...] | list[str | DataType],
        start: DateLike,
        end: DateLike,
        asset_class: str | AssetClass = AssetClass.EQUITY,
        resolution: str | None = None,
    ) -> IngestionResult:
        """Ingest missing market-data partitions from the configured provider."""

        request = IngestionRequest.create(
            provider=self.provider.name,
            symbols=symbols,
            data_types=data_types,
            start=start,
            end=end,
            asset_class=asset_class,
            resolution=resolution,
        )
        requested = all_requested_partitions(request)
        missing = self.coverage.missing_partitions(requested)
        artifacts: list[Path] = []
        manifests: list[Path] = []
        for partition in missing:
            started_at = datetime.now(UTC)
            frame = self._pull_partition(partition)
            artifact_path = self._write_partition(partition, frame)
            manifest, manifest_path, _ = self.manifests.write_success(
                partition,
                artifact_path=artifact_path,
                row_count=frame.height,
                started_at=started_at,
            )
            self.coverage.mark_covered(
                partition,
                row_count=frame.height,
                artifact_uri=artifact_path.as_posix(),
                manifest_uri=manifest_path.as_posix(),
            )
            artifacts.append(artifact_path)
            manifests.append(manifest_path)
            _ = manifest
        return IngestionResult(
            requested=len(requested),
            skipped=len(requested) - len(missing),
            written=len(artifacts),
            artifacts=tuple(artifacts),
            manifests=tuple(manifests),
        )

    def _pull_partition(self, partition: IngestionPartition) -> pl.DataFrame:
        if partition.asset_class != AssetClass.EQUITY.value:
            msg = (
                "unsupported asset class for ingestion service: "
                f"{partition.asset_class}"
            )
            raise ValueError(msg)
        if partition.data_type == DataType.TRADES.value:
            return self.provider.equity_trades(partition.symbol, start=partition.date)
        if partition.data_type == DataType.QUOTES.value:
            return self.provider.equity_quotes(partition.symbol, start=partition.date)
        if partition.data_type == DataType.BARS.value:
            return self.provider.aggregate_bars(
                partition.symbol,
                start=partition.date,
                resolution=partition.resolution or "1min",
            )
        msg = f"unsupported data type for ingestion service: {partition.data_type}"
        raise ValueError(msg)

    def _write_partition(
        self, partition: IngestionPartition, frame: pl.DataFrame
    ) -> Path:
        filename = f"part-{datetime.now(UTC):%Y%m%dT%H%M%S%f}-{uuid4().hex[:8]}.parquet"
        path = self.paths.market_data_file(
            provider=partition.provider,
            asset_class=partition.asset_class,
            data_type=partition.data_type,
            symbol=partition.symbol,
            date=partition.date,
            resolution=partition.resolution,
            filename=filename,
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.write_parquet(path)
        return path


__all__ = ["IngestionResult", "IngestionService"]
