"""Provider interfaces for market and reference data ingestion."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date, datetime
from typing import Protocol, runtime_checkable

import polars as pl

DateLike = str | date | datetime


@runtime_checkable
class MarketDataProvider(Protocol):
    """Protocol implemented by market data providers."""

    @property
    def name(self) -> str:
        """Stable provider identifier used in storage partitions."""
        ...

    def equity_trades(
        self, symbol: str, *, start: DateLike, end: DateLike | None = None
    ) -> pl.DataFrame:
        """Return equity trade prints for ``symbol`` between ``start`` and ``end``."""
        ...

    def equity_quotes(
        self, symbol: str, *, start: DateLike, end: DateLike | None = None
    ) -> pl.DataFrame:
        """Return equity bid/ask quote updates for ``symbol``."""
        ...

    def aggregate_bars(
        self,
        symbol: str,
        *,
        start: DateLike,
        end: DateLike | None = None,
        resolution: str = "1min",
    ) -> pl.DataFrame:
        """Return OHLCV aggregate bars for ``symbol`` at ``resolution``."""
        ...

    def option_chain(
        self, underlying: str, *, as_of: DateLike | None = None
    ) -> pl.DataFrame:
        """Return the option chain for ``underlying`` as of ``as_of``."""
        ...

    def option_contract_trades(
        self, contract: str, *, start: DateLike, end: DateLike | None = None
    ) -> pl.DataFrame:
        """Return option contract trade prints for ``contract``."""
        ...

    def option_contract_quotes(
        self, contract: str, *, start: DateLike, end: DateLike | None = None
    ) -> pl.DataFrame:
        """Return option contract bid/ask quote updates for ``contract``."""
        ...

    def snapshot(self, symbol: str) -> pl.DataFrame:
        """Return a latest market snapshot for ``symbol`` where supported."""
        ...


class BaseMarketDataProvider(ABC):
    """Abstract base class for provider implementations."""

    name: str

    @abstractmethod
    def equity_trades(
        self, symbol: str, *, start: DateLike, end: DateLike | None = None
    ) -> pl.DataFrame:
        """Return equity trade prints for ``symbol`` between ``start`` and ``end``."""
        raise NotImplementedError

    @abstractmethod
    def equity_quotes(
        self, symbol: str, *, start: DateLike, end: DateLike | None = None
    ) -> pl.DataFrame:
        """Return equity bid/ask quote updates for ``symbol``."""
        raise NotImplementedError

    @abstractmethod
    def aggregate_bars(
        self,
        symbol: str,
        *,
        start: DateLike,
        end: DateLike | None = None,
        resolution: str = "1min",
    ) -> pl.DataFrame:
        """Return OHLCV aggregate bars for ``symbol`` at ``resolution``."""
        raise NotImplementedError

    @abstractmethod
    def option_chain(
        self, underlying: str, *, as_of: DateLike | None = None
    ) -> pl.DataFrame:
        """Return the option chain for ``underlying`` as of ``as_of``."""
        raise NotImplementedError

    @abstractmethod
    def option_contract_trades(
        self, contract: str, *, start: DateLike, end: DateLike | None = None
    ) -> pl.DataFrame:
        """Return option contract trade prints for ``contract``."""
        raise NotImplementedError

    @abstractmethod
    def option_contract_quotes(
        self, contract: str, *, start: DateLike, end: DateLike | None = None
    ) -> pl.DataFrame:
        """Return option contract bid/ask quote updates for ``contract``."""
        raise NotImplementedError

    def snapshot(self, symbol: str) -> pl.DataFrame:
        """Return a latest market snapshot for ``symbol`` where supported."""
        msg = f"{self.name} does not implement snapshots"
        raise NotImplementedError(msg)


__all__ = ["BaseMarketDataProvider", "DateLike", "MarketDataProvider"]
