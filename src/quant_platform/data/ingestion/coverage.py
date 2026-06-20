"""Coverage helpers for ingestion partition planning."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import UTC, datetime, time
from typing import Any

from sqlalchemy import insert, select

from quant_platform.data.ingestion.planner import IngestionPartition
from quant_platform.data.storage.catalog import MetadataCatalog, coverage


def _day_bounds(partition: IngestionPartition) -> tuple[datetime, datetime]:
    start = datetime.combine(partition.date, time.min, tzinfo=UTC)
    end = datetime.combine(partition.date, time.max, tzinfo=UTC)
    return start, end


def _metadata_matches(metadata: dict[str, Any], partition: IngestionPartition) -> bool:
    return (
        metadata.get("provider") == partition.provider
        and metadata.get("asset_class") == partition.asset_class
        and metadata.get("data_type") == partition.data_type
        and metadata.get("date") == partition.date.isoformat()
        and metadata.get("resolution") == partition.resolution
    )


class CoverageStore:
    """Read and update ingestion coverage rows in the metadata catalog."""

    def __init__(self, catalog: MetadataCatalog) -> None:
        self.catalog = catalog
        self.catalog.create_all()

    def covered_partitions(
        self, partitions: Iterable[IngestionPartition]
    ) -> set[tuple[str, str, str, str, object, str | None]]:
        """Return keys for requested partitions already covered by successful writes."""

        requested = tuple(partitions)
        if not requested:
            return set()
        with self.catalog.engine.connect() as connection:
            rows = list(connection.execute(select(coverage)).mappings().all())
        covered: set[tuple[str, str, str, str, object, str | None]] = set()
        for partition in requested:
            for row in rows:
                metadata = dict(row["metadata"] or {})
                if (
                    row["dataset"] == partition.dataset
                    and row["symbol"] == partition.symbol
                    and row["row_count"] is not None
                    and row["row_count"] >= 0
                    and _metadata_matches(metadata, partition)
                ):
                    covered.add(partition.key)
                    break
        return covered

    def missing_partitions(
        self, partitions: Iterable[IngestionPartition]
    ) -> tuple[IngestionPartition, ...]:
        """Return only partitions that are not yet represented in coverage."""

        requested = tuple(partitions)
        covered = self.covered_partitions(requested)
        return tuple(
            partition for partition in requested if partition.key not in covered
        )

    def mark_covered(
        self,
        partition: IngestionPartition,
        *,
        row_count: int,
        artifact_uri: str,
        manifest_uri: str,
    ) -> int:
        """Insert a coverage row after a partition is written successfully."""

        start_ts, end_ts = _day_bounds(partition)
        values = {
            "dataset": partition.dataset,
            "symbol": partition.symbol,
            "start_ts": start_ts,
            "end_ts": end_ts,
            "row_count": row_count,
            "metadata": {
                "provider": partition.provider,
                "asset_class": partition.asset_class,
                "data_type": partition.data_type,
                "date": partition.date.isoformat(),
                "resolution": partition.resolution,
                "artifact_uri": artifact_uri,
                "manifest_uri": manifest_uri,
            },
        }
        with self.catalog.engine.begin() as connection:
            result = connection.execute(insert(coverage).values(**values))
        return int(result.inserted_primary_key[0])


__all__ = ["CoverageStore"]
