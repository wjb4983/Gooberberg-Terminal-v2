from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import polars as pl
import pytest

from quant_platform.portfolio.service import PortfolioService


@dataclass(frozen=True)
class Account:
    account_hash: str


class FakeSchwabProvider:
    def __init__(
        self,
        *,
        holdings: dict[str, list[dict[str, Any]]] | None = None,
        histories: dict[str, dict[str, Any]] | None = None,
        failures: set[str] | None = None,
    ) -> None:
        self.holdings = holdings or {}
        self.histories = histories or {}
        self.failures = failures or set()
        self.history_calls: list[str] = []
        self.history_kwargs: list[dict[str, Any]] = []

    def account_numbers(self) -> list[Account]:
        return [Account(account_hash=account_hash) for account_hash in self.holdings]

    def account_holdings(self, account_hash: str) -> pl.DataFrame:
        if f"holdings:{account_hash}" in self.failures:
            raise RuntimeError("holdings unavailable")
        return pl.DataFrame(self.holdings.get(account_hash, []))

    def historical_prices(self, symbol: str, **_kwargs: Any) -> dict[str, Any]:
        symbol = symbol.upper()
        self.history_calls.append(symbol)
        self.history_kwargs.append(dict(_kwargs))
        if f"history:{symbol}" in self.failures:
            raise RuntimeError("history unavailable")
        return self.histories.get(symbol, {"candles": []})


def candles(*closes: float) -> dict[str, Any]:
    return dated_candles(1_704_067_200_000, *closes)


def dated_candles(start_ms: int, *closes: float) -> dict[str, Any]:
    return {
        "candles": [
            {"datetime": start_ms + index * 86_400_000, "close": close}
            for index, close in enumerate(closes)
        ]
    }


def test_summary_normal_holdings_allocations_and_metrics() -> None:
    provider = FakeSchwabProvider(
        holdings={
            "hash-1": [
                {
                    "account_hash": "hash-1",
                    "masked_account_label": "Account ****1111",
                    "symbol": "AAA",
                    "asset_type": "EQUITY",
                    "quantity": 2.0,
                    "market_value": 220.0,
                    "current_price": 110.0,
                    "cost_basis": 200.0,
                    "unrealized_pnl": 20.0,
                },
                {
                    "account_hash": "hash-1",
                    "masked_account_label": "Account ****1111",
                    "symbol": "CASH",
                    "asset_type": "CASH",
                    "quantity": 50.0,
                    "market_value": 50.0,
                },
            ]
        },
        histories={
            "AAA": candles(100.0, 105.0, 110.0),
            "SPY": candles(400.0, 404.0, 408.0),
        },
    )

    summary = PortfolioService(provider).summary()

    assert summary["warnings"] == []
    assert summary["totals"]["total_value"] == 270.0
    assert summary["totals"]["cash_value"] == 50.0
    assert summary["allocation_by_symbol"] == [
        {"symbol": "AAA", "market_value": 220.0, "weight": 220.0 / 270.0},
        {"symbol": "CASH", "market_value": 50.0, "weight": 50.0 / 270.0},
    ]
    assert summary["allocation_by_asset_type"] == [
        {"asset_type": "CASH", "market_value": 50.0, "weight": 50.0 / 270.0},
        {"asset_type": "EQUITY", "market_value": 220.0, "weight": 220.0 / 270.0},
    ]
    assert summary["lookback_metrics"]["MAX"]["weights"]["AAA"] == 220.0 / 270.0
    assert summary["lookback_metrics"]["MAX"]["total_return"] > 0.0
    assert summary["metadata"]["provider"] == "FakeSchwabProvider"
    assert summary["metadata"]["benchmark_symbol"] == "SPY"
    assert summary["metadata"]["benchmark_refreshed_at"] is not None
    assert provider.history_calls == ["AAA", "SPY"]


def test_summary_warns_and_continues_when_price_history_is_missing() -> None:
    provider = FakeSchwabProvider(
        holdings={
            "hash-1": [
                {
                    "symbol": "AAA",
                    "asset_type": "EQUITY",
                    "quantity": 1.0,
                    "market_value": 10.0,
                },
                {
                    "symbol": "BBB",
                    "asset_type": "EQUITY",
                    "quantity": 1.0,
                    "market_value": 20.0,
                },
            ]
        },
        histories={"AAA": candles(10.0, 11.0), "SPY": candles(100.0, 101.0)},
        failures={"history:BBB"},
    )

    summary = PortfolioService(provider).summary()

    assert summary["totals"]["total_value"] == 30.0
    assert summary["lookback_metrics"]["MAX"]["weights"] == {"AAA": 1.0}
    assert [warning["code"] for warning in summary["warnings"]] == [
        "price_history_failed",
        "metrics_symbol_skipped",
    ]
    assert summary["warnings"][0]["symbol"] == "BBB"


def test_summary_uses_deepest_complete_history_for_staggered_symbols() -> None:
    provider = FakeSchwabProvider(
        holdings={
            "hash-1": [
                {
                    "symbol": "AAA",
                    "asset_type": "EQUITY",
                    "quantity": 1.0,
                    "market_value": 14.0,
                },
                {
                    "symbol": "BBB",
                    "asset_type": "EQUITY",
                    "quantity": 1.0,
                    "market_value": 22.0,
                },
            ]
        },
        histories={
            "AAA": dated_candles(1_704_067_200_000, 10.0, 11.0, 12.0, 13.0, 14.0),
            "BBB": dated_candles(1_704_326_400_000, 20.0, 22.0),
            "SPY": dated_candles(1_704_067_200_000, 100.0, 101.0, 102.0, 103.0, 104.0),
        },
    )

    summary = PortfolioService(provider).summary()

    assert "lookback_metrics_failed" not in {
        warning["code"] for warning in summary["warnings"]
    }
    assert summary["lookback_metrics"]["MAX"]["total_return"] == pytest.approx(
        36.0 / 33.0 - 1.0
    )
    assert summary["lookback_metrics"]["3Y"]["weights"] == pytest.approx(
        {"AAA": 14.0 / 36.0, "BBB": 22.0 / 36.0}
    )


def test_summary_empty_holdings_returns_zero_response() -> None:
    provider = FakeSchwabProvider(
        holdings={"hash-empty": []}, histories={"SPY": candles(100.0, 101.0)}
    )

    summary = PortfolioService(provider).summary()

    assert summary["totals"] == {
        "total_value": 0,
        "securities_value": 0,
        "cash_value": 0,
        "cost_basis": 0,
        "unrealized_pnl": 0,
    }
    assert summary["allocation_by_symbol"] == []
    assert summary["allocation_by_asset_type"] == []
    assert summary["holdings"] == []
    assert summary["lookback_metrics"]["MAX"]["weights"] == {"CASH": 1.0}
    assert summary["warnings"] == []


def test_summary_cash_only_holdings_have_zero_metrics_without_security_history() -> (
    None
):
    provider = FakeSchwabProvider(
        holdings={
            "hash-1": [
                {
                    "symbol": "CASH",
                    "asset_type": "CASH",
                    "quantity": 125.0,
                    "market_value": 125.0,
                }
            ]
        },
        histories={"SPY": candles(100.0, 99.0)},
    )

    summary = PortfolioService(provider).summary()

    assert summary["totals"]["total_value"] == 125.0
    assert summary["allocation_by_asset_type"] == [
        {"asset_type": "CASH", "market_value": 125.0, "weight": 1.0}
    ]
    assert summary["lookback_metrics"]["MAX"]["weights"] == {"CASH": 1.0}
    assert summary["lookback_metrics"]["MAX"]["total_return"] == 0.0
    assert provider.history_calls == ["SPY"]


def test_summary_benchmark_failure_warns_but_keeps_portfolio_metrics() -> None:
    provider = FakeSchwabProvider(
        holdings={
            "hash-1": [
                {
                    "symbol": "AAA",
                    "asset_type": "EQUITY",
                    "quantity": 1.0,
                    "market_value": 12.0,
                }
            ]
        },
        histories={"AAA": candles(10.0, 12.0)},
        failures={"history:SPY"},
    )

    summary = PortfolioService(provider).summary()

    assert summary["lookback_metrics"]["MAX"]["total_return"] == pytest.approx(0.2)
    assert summary["lookback_metrics"]["MAX"]["beta"] == 0.0
    assert summary["metadata"]["benchmark_refreshed_at"] is None
    assert [warning["code"] for warning in summary["warnings"]] == [
        "benchmark_history_failed"
    ]


def test_summary_reuses_cached_holdings_and_price_histories() -> None:
    provider = FakeSchwabProvider(
        holdings={
            "hash-1": [
                {
                    "symbol": "AAA",
                    "asset_type": "EQUITY",
                    "quantity": 1.0,
                    "market_value": 10.0,
                }
            ]
        },
        histories={"AAA": candles(10.0, 11.0), "SPY": candles(100.0, 101.0)},
    )
    provider.holding_calls = []
    original_account_holdings = provider.account_holdings

    def counted_account_holdings(account_hash: str) -> pl.DataFrame:
        provider.holding_calls.append(account_hash)
        return original_account_holdings(account_hash)

    provider.account_holdings = counted_account_holdings  # type: ignore[method-assign]
    service = PortfolioService(provider)

    first = service.summary()
    second = service.summary()

    assert provider.holding_calls == ["hash-1"]
    assert provider.history_calls == ["AAA", "SPY"]
    assert first["metadata"]["stale_data"] is False
    assert second["metadata"]["stale_data"] is True
    assert [warning["code"] for warning in second["warnings"]].count("stale_data") == 3
    assert (
        second["metadata"]["holdings_refreshed_at"]
        == first["metadata"]["holdings_refreshed_at"]
    )
    assert (
        second["metadata"]["prices_refreshed_at"]
        == first["metadata"]["prices_refreshed_at"]
    )


def test_summary_requests_long_daily_price_history_for_lookback_metrics() -> None:
    provider = FakeSchwabProvider(
        holdings={
            "hash-1": [
                {
                    "symbol": "AAA",
                    "asset_type": "EQUITY",
                    "quantity": 1.0,
                    "market_value": 11.0,
                }
            ]
        },
        histories={"AAA": candles(10.0, 11.0), "SPY": candles(100.0, 101.0)},
    )

    PortfolioService(provider).summary()

    assert provider.history_calls == ["AAA", "SPY"]
    assert provider.history_kwargs == [
        {"period_type": "year", "period": 20},
        {"period_type": "year", "period": 20},
    ]
