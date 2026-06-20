from __future__ import annotations

import polars as pl

from quant_platform.data.providers import FakeMarketDataProvider
from quant_platform.data.providers.base import MarketDataProvider


def test_fake_provider_matches_provider_protocol() -> None:
    provider = FakeMarketDataProvider()

    assert isinstance(provider, MarketDataProvider)
    assert provider.name == "fake"


def test_fake_equity_trades_are_deterministic() -> None:
    provider = FakeMarketDataProvider()

    first = provider.equity_trades("aapl", start="2026-06-20")
    second = provider.equity_trades("AAPL", start="2026-06-20")

    assert first.equals(second)
    assert first.select("symbol").to_series().to_list() == ["AAPL", "AAPL", "AAPL"]
    assert first.schema["timestamp"] == pl.Datetime
    assert first.select("size").to_series().to_list() == [100, 125, 150]


def test_fake_equity_quotes_and_bars_have_expected_columns() -> None:
    provider = FakeMarketDataProvider()

    quotes = provider.equity_quotes("msft", start="2026-06-20")
    bars = provider.aggregate_bars("msft", start="2026-06-20", resolution="5min")

    assert quotes.columns == [
        "symbol",
        "timestamp",
        "bid_price",
        "bid_size",
        "ask_price",
        "ask_size",
        "exchange",
    ]
    assert bars.select("resolution").to_series().to_list() == ["5min", "5min", "5min"]
    assert set(["open", "high", "low", "close", "volume", "vwap"]).issubset(
        bars.columns
    )


def test_fake_option_chain_and_contract_market_data() -> None:
    provider = FakeMarketDataProvider()

    chain = provider.option_chain("spy", as_of="2026-06-20")
    contract = chain.item(0, "contract")
    trades = provider.option_contract_trades(contract, start="2026-06-20")
    quotes = provider.option_contract_quotes(contract.lower(), start="2026-06-20")

    assert chain.shape == (2, 9)
    assert chain.select("underlying").to_series().to_list() == ["SPY", "SPY"]
    assert chain.select("option_type").to_series().to_list() == ["call", "put"]
    assert trades.select("contract").to_series().to_list() == [contract, contract]
    assert quotes.select("contract").to_series().to_list() == [contract, contract]


def test_fake_snapshot_is_single_row_latest_view() -> None:
    provider = FakeMarketDataProvider()

    snapshot = provider.snapshot("nvda")

    assert snapshot.shape == (1, 6)
    assert snapshot.item(0, "symbol") == "NVDA"
    assert snapshot.item(0, "bid_price") < snapshot.item(0, "ask_price")


def test_fake_provider_rejects_empty_symbols() -> None:
    provider = FakeMarketDataProvider()

    try:
        provider.equity_trades("   ", start="2026-06-20")
    except ValueError as exc:
        assert str(exc) == "symbol cannot be empty"
    else:  # pragma: no cover - protects against false positives
        raise AssertionError("empty symbol should be rejected")
