"""Deterministic fake market data provider for tests and local demos."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta

import polars as pl

from quant_platform.data.providers.base import BaseMarketDataProvider, DateLike


def _parse_date(value: DateLike | None) -> date:
    if value is None:
        return date(2026, 6, 20)
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return datetime.fromisoformat(value).date()


def _symbol(value: str) -> str:
    text = value.strip().upper()
    if not text:
        msg = "symbol cannot be empty"
        raise ValueError(msg)
    return text


def _base_price(identifier: str) -> float:
    return 100.0 + (sum(ord(char) for char in identifier.upper()) % 50)


def _timestamps(day: date, count: int, step: timedelta) -> list[datetime]:
    start = datetime.combine(day, time(14, 30))
    return [start + index * step for index in range(count)]


@dataclass(frozen=True)
class FakeMarketDataProvider(BaseMarketDataProvider):
    """Small deterministic provider suitable for repeatable tests and UI demos."""

    name: str = "fake"

    def equity_trades(
        self, symbol: str, *, start: DateLike, end: DateLike | None = None
    ) -> pl.DataFrame:
        ticker = _symbol(symbol)
        day = _parse_date(start)
        base = _base_price(ticker)
        timestamps = _timestamps(day, 3, timedelta(minutes=1))
        return pl.DataFrame(
            {
                "symbol": [ticker] * 3,
                "timestamp": timestamps,
                "price": [base, base + 0.25, base + 0.5],
                "size": [100, 125, 150],
                "exchange": ["FAKE"] * 3,
                "trade_id": [f"{ticker}-{day:%Y%m%d}-{idx}" for idx in range(1, 4)],
            }
        )

    def equity_quotes(
        self, symbol: str, *, start: DateLike, end: DateLike | None = None
    ) -> pl.DataFrame:
        ticker = _symbol(symbol)
        day = _parse_date(start)
        base = _base_price(ticker)
        timestamps = _timestamps(day, 3, timedelta(minutes=1))
        return pl.DataFrame(
            {
                "symbol": [ticker] * 3,
                "timestamp": timestamps,
                "bid_price": [base - 0.05, base + 0.2, base + 0.45],
                "bid_size": [10, 11, 12],
                "ask_price": [base + 0.05, base + 0.3, base + 0.55],
                "ask_size": [12, 13, 14],
                "exchange": ["FAKE"] * 3,
            }
        )

    def aggregate_bars(
        self,
        symbol: str,
        *,
        start: DateLike,
        end: DateLike | None = None,
        resolution: str = "1min",
    ) -> pl.DataFrame:
        ticker = _symbol(symbol)
        day = _parse_date(start)
        base = _base_price(ticker)
        timestamps = _timestamps(day, 3, timedelta(minutes=1))
        return pl.DataFrame(
            {
                "symbol": [ticker] * 3,
                "timestamp": timestamps,
                "resolution": [resolution] * 3,
                "open": [base, base + 0.3, base + 0.6],
                "high": [base + 0.4, base + 0.7, base + 1.0],
                "low": [base - 0.2, base + 0.1, base + 0.4],
                "close": [base + 0.25, base + 0.55, base + 0.85],
                "volume": [1_000, 1_100, 1_200],
                "vwap": [base + 0.1, base + 0.4, base + 0.7],
            }
        )

    def option_chain(
        self, underlying: str, *, as_of: DateLike | None = None
    ) -> pl.DataFrame:
        ticker = _symbol(underlying)
        day = _parse_date(as_of)
        expiration = day + timedelta(days=30)
        base = round(_base_price(ticker))
        contracts = [
            (
                f"O:{ticker}{expiration:%y%m%d}C{(base + 5) * 1000:08d}",
                "call",
                base + 5,
            ),
            (
                f"O:{ticker}{expiration:%y%m%d}P{(base - 5) * 1000:08d}",
                "put",
                base - 5,
            ),
        ]
        return pl.DataFrame(
            {
                "underlying": [ticker, ticker],
                "contract": [contract for contract, _, _ in contracts],
                "as_of": [day, day],
                "expiration": [expiration, expiration],
                "option_type": [kind for _, kind, _ in contracts],
                "strike": [strike for _, _, strike in contracts],
                "bid_price": [1.2, 1.1],
                "ask_price": [1.35, 1.25],
                "open_interest": [1000, 900],
            }
        )

    def option_contract_trades(
        self, contract: str, *, start: DateLike, end: DateLike | None = None
    ) -> pl.DataFrame:
        contract_symbol = _symbol(contract)
        day = _parse_date(start)
        timestamps = _timestamps(day, 2, timedelta(minutes=2))
        return pl.DataFrame(
            {
                "contract": [contract_symbol] * 2,
                "timestamp": timestamps,
                "price": [1.25, 1.3],
                "size": [5, 8],
                "exchange": ["FAKE"] * 2,
                "trade_id": [
                    f"{contract_symbol}-{day:%Y%m%d}-{idx}" for idx in range(1, 3)
                ],
            }
        )

    def option_contract_quotes(
        self, contract: str, *, start: DateLike, end: DateLike | None = None
    ) -> pl.DataFrame:
        contract_symbol = _symbol(contract)
        day = _parse_date(start)
        timestamps = _timestamps(day, 2, timedelta(minutes=2))
        return pl.DataFrame(
            {
                "contract": [contract_symbol] * 2,
                "timestamp": timestamps,
                "bid_price": [1.2, 1.25],
                "bid_size": [10, 11],
                "ask_price": [1.35, 1.4],
                "ask_size": [12, 13],
                "exchange": ["FAKE"] * 2,
            }
        )

    def snapshot(self, symbol: str) -> pl.DataFrame:
        ticker = _symbol(symbol)
        day = date(2026, 6, 20)
        base = _base_price(ticker)
        return pl.DataFrame(
            {
                "symbol": [ticker],
                "timestamp": [datetime.combine(day, time(20, 0))],
                "last_price": [base + 0.5],
                "bid_price": [base + 0.45],
                "ask_price": [base + 0.55],
                "volume": [3_300],
            }
        )


__all__ = ["FakeMarketDataProvider"]
