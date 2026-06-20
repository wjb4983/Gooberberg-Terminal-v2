from __future__ import annotations

from pathlib import Path

import pytest

from quant_platform.common.enums import AssetClass, DataType, Provider
from quant_platform.data.storage.lake import DataLakePaths, date_partitions
from quant_platform.data.storage.parquet import ParquetPaths


def parts(path: Path, root: Path) -> tuple[str, ...]:
    return path.relative_to(root.resolve()).parts


def test_equity_path_partitions(tmp_path: Path) -> None:
    lake = DataLakePaths(tmp_path)

    path = lake.equity(
        provider=Provider.MASSIVE,
        data_type=DataType.TRADES,
        symbol="aapl",
        date="2026-06-20",
    )

    assert parts(path, tmp_path) == (
        "market_data",
        "provider=massive",
        "asset_class=equity",
        "data_type=trades",
        "symbol=AAPL",
        "year=2026",
        "month=06",
        "day=20",
    )


def test_etf_bar_path_includes_resolution(tmp_path: Path) -> None:
    lake = DataLakePaths(tmp_path)

    path = lake.etf(
        provider="polygon",
        data_type=DataType.BARS,
        symbol="spy",
        resolution="1min",
        date="2026-01-05",
    )

    assert parts(path, tmp_path) == (
        "market_data",
        "provider=polygon",
        "asset_class=etf",
        "data_type=bars",
        "symbol=SPY",
        "resolution=1min",
        "year=2026",
        "month=01",
        "day=05",
    )


@pytest.mark.parametrize("data_type", [DataType.TRADES, DataType.QUOTES, DataType.BARS])
def test_option_contract_paths_include_underlying_contract_resolution_and_date(
    tmp_path: Path, data_type: DataType
) -> None:
    lake = DataLakePaths(tmp_path)

    path = lake.option_contract(
        provider=Provider.MASSIVE,
        data_type=data_type,
        underlying="msft",
        contract="O:MSFT260620C00400000",
        resolution="1hour",
        date="2026-06-20",
    )

    assert parts(path, tmp_path) == (
        "market_data",
        "provider=massive",
        "asset_class=option",
        f"data_type={data_type.value}",
        "underlying=MSFT",
        "contract=O:MSFT260620C00400000",
        "resolution=1hour",
        "year=2026",
        "month=06",
        "day=20",
    )


def test_option_chain_path_includes_underlying_and_date(tmp_path: Path) -> None:
    lake = DataLakePaths(tmp_path)

    path = lake.option_chain(
        provider=Provider.POLYGON, underlying="qqq", date="2026-03-14"
    )

    assert parts(path, tmp_path) == (
        "market_data",
        "provider=polygon",
        "asset_class=option",
        "data_type=chains",
        "underlying=QQQ",
        "year=2026",
        "month=03",
        "day=14",
    )


def test_snapshot_path_partitions(tmp_path: Path) -> None:
    lake = DataLakePaths(tmp_path)

    path = lake.snapshot(
        provider=Provider.ALPACA,
        asset_class=AssetClass.EQUITY,
        symbol="nvda",
        date="2026-06-20",
    )

    assert parts(path, tmp_path) == (
        "snapshots",
        "provider=alpaca",
        "asset_class=equity",
        "symbol=NVDA",
        "year=2026",
        "month=06",
        "day=20",
    )


def test_derived_dataset_path_partitions(tmp_path: Path) -> None:
    lake = DataLakePaths(tmp_path)

    path = lake.derived(
        dataset="daily_factors",
        version="v1",
        asset_class=AssetClass.ETF,
        symbol="iwm",
        date="2026-06-20",
    )

    assert parts(path, tmp_path) == (
        "derived",
        "dataset=daily_factors",
        "version=v1",
        "asset_class=etf",
        "symbol=IWM",
        "year=2026",
        "month=06",
        "day=20",
    )


def test_artifact_path_partitions(tmp_path: Path) -> None:
    lake = DataLakePaths(tmp_path)

    path = lake.artifact(
        artifact_type="model", name="momentum_ranker", version="v2", run_id="run_123"
    )

    assert parts(path, tmp_path) == (
        "artifacts",
        "artifact_type=model",
        "name=momentum_ranker",
        "version=v2",
        "run_id=run_123",
    )


def test_parquet_file_appends_default_filename(tmp_path: Path) -> None:
    parquet = ParquetPaths(tmp_path)

    path = parquet.equity_file(
        provider=Provider.MASSIVE,
        data_type=DataType.QUOTES,
        symbol="tsla",
        date="2026-06-20",
    )

    assert parts(path, tmp_path) == (
        "market_data",
        "provider=massive",
        "asset_class=equity",
        "data_type=quotes",
        "symbol=TSLA",
        "year=2026",
        "month=06",
        "day=20",
        "part-00000.parquet",
    )


def test_date_partitions_accept_date_like_values() -> None:
    assert date_partitions("2026-06-20") == ("year=2026", "month=06", "day=20")


def test_parquet_rejects_non_parquet_filename(tmp_path: Path) -> None:
    parquet = ParquetPaths(tmp_path)

    with pytest.raises(ValueError, match="must end with .parquet"):
        parquet.equity_file(
            provider=Provider.MASSIVE,
            data_type=DataType.TRADES,
            symbol="AAPL",
            filename="part.json",
        )
