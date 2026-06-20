from __future__ import annotations

from datetime import UTC, date, datetime, time
from pathlib import Path

from sqlalchemy import insert

from quant_platform.data.ingestion.coverage import CoverageStore
from quant_platform.data.ingestion.planner import IngestionPartition
from quant_platform.data.storage.catalog import MetadataCatalog, coverage


def catalog(tmp_path: Path) -> MetadataCatalog:
    return MetadataCatalog(tmp_path / "catalog" / "metadata.sqlite")


def partition(
    *, data_type: str = "trades", day: date = date(2026, 6, 20)
) -> IngestionPartition:
    return IngestionPartition(
        provider="fake",
        asset_class="equity",
        data_type=data_type,
        symbol="AAPL",
        date=day,
        dataset=f"market_data.equity.{data_type}",
    )


def test_mark_covered_records_partition_metadata(
    tmp_path: Path,
) -> None:
    store = CoverageStore(catalog(tmp_path))
    requested = partition()

    row_id = store.mark_covered(
        requested,
        row_count=3,
        artifact_uri="lake/fake/aapl.parquet",
        manifest_uri="catalog/manifests/aapl.json",
    )

    assert row_id == 1
    assert store.covered_partitions([requested]) == {requested.key}
    assert store.missing_partitions([requested]) == ()


def test_coverage_requires_metadata_match_and_non_negative_rows(
    tmp_path: Path,
) -> None:
    store = CoverageStore(catalog(tmp_path))
    requested = partition()
    start_ts = datetime.combine(requested.date, time.min, tzinfo=UTC)
    end_ts = datetime.combine(requested.date, time.max, tzinfo=UTC)

    with store.catalog.engine.begin() as connection:
        connection.execute(
            insert(coverage).values(
                dataset=requested.dataset,
                symbol=requested.symbol,
                start_ts=start_ts,
                end_ts=end_ts,
                row_count=-1,
                metadata={
                    "provider": requested.provider,
                    "asset_class": requested.asset_class,
                    "data_type": requested.data_type,
                    "date": requested.date.isoformat(),
                    "resolution": requested.resolution,
                },
            )
        )
        connection.execute(
            insert(coverage).values(
                dataset=requested.dataset,
                symbol=requested.symbol,
                start_ts=start_ts,
                end_ts=end_ts,
                row_count=3,
                metadata={
                    "provider": "other",
                    "asset_class": requested.asset_class,
                    "data_type": requested.data_type,
                    "date": requested.date.isoformat(),
                    "resolution": requested.resolution,
                },
            )
        )

    assert store.covered_partitions([requested]) == set()
    assert store.missing_partitions([requested]) == (requested,)


def test_empty_coverage_queries_do_not_touch_catalog_rows(tmp_path: Path) -> None:
    store = CoverageStore(catalog(tmp_path))

    assert store.covered_partitions([]) == set()
    assert store.missing_partitions([]) == ()
