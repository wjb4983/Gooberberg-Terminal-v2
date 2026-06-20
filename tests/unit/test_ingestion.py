from __future__ import annotations

from pathlib import Path

import polars as pl

from quant_platform.common.enums import DataType
from quant_platform.data.ingestion.coverage import CoverageStore
from quant_platform.data.ingestion.planner import (
    IngestionRequest,
    all_requested_partitions,
)
from quant_platform.data.ingestion.service import IngestionService
from quant_platform.data.providers.fake import FakeMarketDataProvider
from quant_platform.data.storage.catalog import MetadataCatalog


def catalog(tmp_path: Path) -> MetadataCatalog:
    return MetadataCatalog(tmp_path / "catalog" / "metadata.sqlite")


def request() -> IngestionRequest:
    return IngestionRequest.create(
        provider="fake",
        symbols=["AAPL"],
        data_types=[DataType.TRADES],
        start="2026-06-20",
        end="2026-06-22",
    )


def test_no_coverage_marks_all_partitions_missing(tmp_path: Path) -> None:
    store = CoverageStore(catalog(tmp_path))
    partitions = all_requested_partitions(request())

    assert store.missing_partitions(partitions) == partitions


def test_full_coverage_marks_no_partitions_missing(tmp_path: Path) -> None:
    store = CoverageStore(catalog(tmp_path))
    partitions = all_requested_partitions(request())
    for partition in partitions:
        store.mark_covered(
            partition,
            row_count=3,
            artifact_uri="data/lake/example.parquet",
            manifest_uri="data/catalog/ingestion_manifests/example.json",
        )

    assert store.missing_partitions(partitions) == ()


def test_partial_coverage_only_returns_uncovered_partitions(tmp_path: Path) -> None:
    store = CoverageStore(catalog(tmp_path))
    partitions = all_requested_partitions(request())
    store.mark_covered(
        partitions[0],
        row_count=3,
        artifact_uri="data/lake/example.parquet",
        manifest_uri="data/catalog/ingestion_manifests/example.json",
    )

    missing = store.missing_partitions(partitions)

    assert missing == partitions[1:]


def test_ingestion_creates_manifest_and_appends_parquet(tmp_path: Path) -> None:
    service = IngestionService(
        provider=FakeMarketDataProvider(),
        catalog=catalog(tmp_path),
        data_root=tmp_path / "lake",
        manifest_root=tmp_path / "catalog" / "ingestion_manifests",
    )

    result = service.ingest_market_data(
        symbols=["aapl"],
        data_types=[DataType.TRADES],
        start="2026-06-20",
        end="2026-06-20",
    )

    assert result.requested == 1
    assert result.skipped == 0
    assert result.written == 1
    assert result.artifacts[0].exists()
    assert result.manifests[0].exists()
    assert "size_bytes" in result.manifests[0].read_text()
    assert pl.read_parquet(result.artifacts[0]).height == 3


class CountingFakeProvider:
    name = "fake"

    def __init__(self) -> None:
        self.delegate = FakeMarketDataProvider()
        self.calls: list[str] = []

    def equity_trades(self, symbol: str, *, start, end=None):
        self.calls.append(f"trades:{symbol}:{start}")
        return self.delegate.equity_trades(symbol, start=start, end=end)

    def equity_quotes(self, symbol: str, *, start, end=None):
        return self.delegate.equity_quotes(symbol, start=start, end=end)

    def aggregate_bars(self, symbol: str, *, start, end=None, resolution="1min"):
        return self.delegate.aggregate_bars(
            symbol, start=start, end=end, resolution=resolution
        )

    def option_chain(self, underlying: str, *, as_of=None):
        return self.delegate.option_chain(underlying, as_of=as_of)

    def option_contract_trades(self, contract: str, *, start, end=None):
        return self.delegate.option_contract_trades(contract, start=start, end=end)

    def option_contract_quotes(self, contract: str, *, start, end=None):
        return self.delegate.option_contract_quotes(contract, start=start, end=end)

    def snapshot(self, symbol: str):
        return self.delegate.snapshot(symbol)


def test_duplicate_ingestion_prevention_with_fake_provider(tmp_path: Path) -> None:
    provider = CountingFakeProvider()
    service = IngestionService(
        provider=provider,
        catalog=catalog(tmp_path),
        data_root=tmp_path / "lake",
        manifest_root=tmp_path / "catalog" / "ingestion_manifests",
    )

    first = service.ingest_market_data(
        symbols=["AAPL"],
        data_types=[DataType.TRADES],
        start="2026-06-20",
        end="2026-06-20",
    )
    second = service.ingest_market_data(
        symbols=["AAPL"],
        data_types=[DataType.TRADES],
        start="2026-06-20",
        end="2026-06-20",
    )

    assert first.written == 1
    assert second.written == 0
    assert second.skipped == 1
    assert provider.calls == ["trades:AAPL:2026-06-20"]
    parquet_files = list((tmp_path / "lake").glob("**/*.parquet"))
    assert parquet_files == [first.artifacts[0]]
