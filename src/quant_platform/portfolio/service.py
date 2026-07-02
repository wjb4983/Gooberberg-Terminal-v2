"""Schwab-backed portfolio summary service.

The service is intentionally defensive: Schwab account, holding, price-history, and
benchmark calls can fail independently, so recoverable problems are returned as
warnings while still producing the portions of the portfolio response that can be
computed safely.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any

import pandas as pd  # type: ignore[import-untyped]
import polars as pl

from quant_platform.data.providers.schwab import SchwabProvider
from quant_platform.portfolio.metrics import (
    LOOKBACKS,
    Lookback,
    compute_portfolio_metrics,
)

DEFAULT_BENCHMARK_SYMBOL = "SPY"
PRICE_HISTORY_PERIOD = 10


@dataclass(frozen=True)
class PortfolioWarning:
    """Recoverable portfolio service warning for API/UI display."""

    code: str
    message: str
    symbol: str | None = None
    account_hash: str | None = None


@dataclass(frozen=True)
class PortfolioServiceMetadata:
    """Display metadata describing how the portfolio summary was refreshed."""

    provider: str
    benchmark_symbol: str
    refreshed_at: str
    holdings_refreshed_at: str
    prices_refreshed_at: str
    benchmark_refreshed_at: str | None
    account_hashes: list[str]
    lookbacks: list[str]


class PortfolioService:
    """Build account, allocation, holding, and lookback summaries from Schwab."""

    def __init__(self, provider: SchwabProvider | None = None) -> None:
        self.provider = provider or SchwabProvider()

    def summary(
        self,
        *,
        account_hashes: Iterable[str] | None = None,
        benchmark_symbol: str = DEFAULT_BENCHMARK_SYMBOL,
    ) -> dict[str, Any]:
        """Return a UI-friendly portfolio summary using Schwab provider data."""

        refreshed_at = _utc_now()
        warnings: list[PortfolioWarning] = []
        resolved_hashes = self._resolve_account_hashes(account_hashes, warnings)
        holdings = self._fetch_holdings(resolved_hashes, warnings)
        holding_rows = _holding_rows(holdings)
        totals = _account_totals(holding_rows)
        allocation_by_symbol = _allocation_by_symbol(
            holding_rows, totals["total_value"]
        )
        allocation_by_asset_type = _allocation_by_asset_type(
            holding_rows, totals["total_value"]
        )

        price_histories = self._fetch_price_histories(holding_rows, warnings)
        prices_refreshed_at = _utc_now()
        benchmark_prices = self._fetch_benchmark_history(benchmark_symbol, warnings)
        benchmark_refreshed_at = _utc_now() if benchmark_prices is not None else None
        lookback_metrics = _lookback_metrics(
            holding_rows,
            price_histories,
            benchmark_prices,
            warnings,
        )

        metadata = PortfolioServiceMetadata(
            provider=self.provider.__class__.__name__,
            benchmark_symbol=benchmark_symbol,
            refreshed_at=refreshed_at,
            holdings_refreshed_at=refreshed_at,
            prices_refreshed_at=prices_refreshed_at,
            benchmark_refreshed_at=benchmark_refreshed_at,
            account_hashes=resolved_hashes,
            lookbacks=list(LOOKBACKS),
        )
        return {
            "totals": totals,
            "allocation_by_symbol": allocation_by_symbol,
            "allocation_by_asset_type": allocation_by_asset_type,
            "holdings": holding_rows,
            "lookback_metrics": lookback_metrics,
            "warnings": [asdict(warning) for warning in warnings],
            "metadata": asdict(metadata),
        }

    def _resolve_account_hashes(
        self, account_hashes: Iterable[str] | None, warnings: list[PortfolioWarning]
    ) -> list[str]:
        if account_hashes is not None:
            return [account_hash for account_hash in account_hashes if account_hash]
        try:
            return [account.account_hash for account in self.provider.account_numbers()]
        except Exception as exc:  # noqa: BLE001 - returned as partial response warning
            warnings.append(
                PortfolioWarning(
                    code="account_lookup_failed",
                    message=f"Unable to fetch Schwab account list: {exc}",
                )
            )
            return []

    def _fetch_holdings(
        self, account_hashes: list[str], warnings: list[PortfolioWarning]
    ) -> pl.DataFrame:
        frames: list[pl.DataFrame] = []
        for account_hash in account_hashes:
            try:
                frames.append(self.provider.account_holdings(account_hash))
            except Exception as exc:  # noqa: BLE001 - preserve partial accounts
                warnings.append(
                    PortfolioWarning(
                        code="holdings_failed",
                        message=f"Unable to fetch holdings for account: {exc}",
                        account_hash=account_hash,
                    )
                )
        if not frames:
            return pl.DataFrame()
        non_empty = [frame for frame in frames if frame.height > 0]
        return pl.concat(non_empty, how="diagonal") if non_empty else pl.DataFrame()

    def _fetch_price_histories(
        self, holdings: list[dict[str, Any]], warnings: list[PortfolioWarning]
    ) -> dict[str, pd.Series]:
        histories: dict[str, pd.Series] = {}
        for symbol in _priced_symbols(holdings):
            try:
                payload = self.provider.historical_prices(
                    symbol, period=PRICE_HISTORY_PERIOD
                )
                history = _history_to_series(payload)
                if history.empty:
                    raise ValueError("price history contained no candles")
                histories[symbol] = history
            except Exception as exc:  # noqa: BLE001 - skip symbol, keep response
                warnings.append(
                    PortfolioWarning(
                        code="price_history_failed",
                        message=f"Unable to fetch price history for {symbol}: {exc}",
                        symbol=symbol,
                    )
                )
        return histories

    def _fetch_benchmark_history(
        self, benchmark_symbol: str, warnings: list[PortfolioWarning]
    ) -> pd.Series | None:
        try:
            payload = self.provider.historical_prices(
                benchmark_symbol, period=PRICE_HISTORY_PERIOD
            )
            history = _history_to_series(payload)
            if history.empty:
                raise ValueError("benchmark history contained no candles")
            return history
        except Exception as exc:  # noqa: BLE001 - beta can degrade to zero
            warnings.append(
                PortfolioWarning(
                    code="benchmark_history_failed",
                    message=(
                        f"Unable to fetch benchmark history for "
                        f"{benchmark_symbol}: {exc}"
                    ),
                    symbol=benchmark_symbol,
                )
            )
            return None


def _holding_rows(holdings: pl.DataFrame) -> list[dict[str, Any]]:
    if holdings.is_empty():
        return []
    rows = holdings.to_dicts()
    for row in rows:
        row["symbol"] = str(row.get("symbol") or "").upper() or None
        row["asset_type"] = str(row.get("asset_type") or "UNKNOWN").upper()
        row["quantity"] = _float(row.get("quantity"))
        row["market_value"] = _float(row.get("market_value"))
        row["current_price"] = _float(row.get("current_price"))
        row["cost_basis"] = _float(row.get("cost_basis"))
        row["average_price"] = _float(row.get("average_price"))
        row["unrealized_pnl"] = _float(row.get("unrealized_pnl"))
        row["unrealized_pnl_percent"] = _float(row.get("unrealized_pnl_percent"))
        row["is_cash"] = _is_cash(row)
    return rows


def _account_totals(rows: list[dict[str, Any]]) -> dict[str, float]:
    securities_value = sum(_market_value(row) for row in rows if not row["is_cash"])
    cash_value = sum(_market_value(row) for row in rows if row["is_cash"])
    cost_basis = sum(row.get("cost_basis") or 0.0 for row in rows)
    return {
        "total_value": securities_value + cash_value,
        "securities_value": securities_value,
        "cash_value": cash_value,
        "cost_basis": cost_basis,
        "unrealized_pnl": sum(row.get("unrealized_pnl") or 0.0 for row in rows),
    }


def _allocation_by_symbol(
    rows: list[dict[str, Any]], total: float
) -> list[dict[str, Any]]:
    buckets: dict[str, float] = {}
    for row in rows:
        symbol = "CASH" if row["is_cash"] else row.get("symbol")
        if symbol:
            buckets[symbol] = buckets.get(symbol, 0.0) + _market_value(row)
    return _allocation_rows(buckets, total, "symbol")


def _allocation_by_asset_type(
    rows: list[dict[str, Any]], total: float
) -> list[dict[str, Any]]:
    buckets: dict[str, float] = {}
    for row in rows:
        asset_type = "CASH" if row["is_cash"] else row.get("asset_type") or "UNKNOWN"
        buckets[asset_type] = buckets.get(asset_type, 0.0) + _market_value(row)
    return _allocation_rows(buckets, total, "asset_type")


def _allocation_rows(
    buckets: Mapping[str, float], total: float, key: str
) -> list[dict[str, Any]]:
    return [
        {key: name, "market_value": value, "weight": value / total if total else 0.0}
        for name, value in sorted(buckets.items())
    ]


def _lookback_metrics(
    holdings: list[dict[str, Any]],
    histories: dict[str, pd.Series],
    benchmark: pd.Series | None,
    warnings: list[PortfolioWarning],
) -> dict[str, dict[str, Any]]:
    cash = sum(_market_value(row) for row in holdings if row["is_cash"])
    positions = {
        row["symbol"]: row["quantity"]
        for row in holdings
        if not row["is_cash"] and row.get("symbol") in histories and row.get("quantity")
    }
    missing = sorted(
        {
            row["symbol"]
            for row in holdings
            if not row["is_cash"]
            and row.get("symbol")
            and row.get("symbol") not in histories
        }
    )
    for symbol in missing:
        warnings.append(
            PortfolioWarning(
                code="metrics_symbol_skipped",
                message=(
                    f"Lookback metrics excluded {symbol} because "
                    "price history is unavailable."
                ),
                symbol=symbol,
            )
        )
    prices = _prices_frame(histories, benchmark)
    metrics: dict[str, dict[str, Any]] = {}
    for lookback in LOOKBACKS:
        try:
            result = compute_portfolio_metrics(
                prices,
                positions,
                cash=cash,
                benchmark_prices=benchmark,
                lookback=lookback,
            )
            metrics[lookback] = asdict(result)
        except Exception as exc:  # noqa: BLE001 - isolate lookback failures
            warnings.append(
                PortfolioWarning(
                    code="lookback_metrics_failed",
                    message=f"Unable to compute {lookback} metrics: {exc}",
                )
            )
            metrics[lookback] = _zero_metrics(lookback)
    return metrics


def _prices_frame(
    histories: dict[str, pd.Series], benchmark: pd.Series | None
) -> pd.DataFrame:
    if histories:
        return pd.DataFrame(histories).sort_index().ffill().dropna(how="all")
    index = (
        benchmark.index
        if benchmark is not None and not benchmark.empty
        else pd.DatetimeIndex([pd.Timestamp.now(tz=UTC)])
    )
    return pd.DataFrame(index=index)


def _history_to_series(payload: Mapping[str, Any]) -> pd.Series:
    candles = payload.get("candles", [])
    if not isinstance(candles, list):
        return pd.Series(dtype="float64")
    values: list[tuple[pd.Timestamp, float]] = []
    for candle in candles:
        if not isinstance(candle, Mapping) or candle.get("close") is None:
            continue
        raw_time = candle.get("datetime") or candle.get("date") or candle.get("time")
        timestamp = pd.to_datetime(raw_time, unit="ms", utc=True).tz_convert(None)
        values.append((timestamp, float(candle["close"])))
    if not values:
        return pd.Series(dtype="float64")
    index, closes = zip(*values, strict=True)
    return pd.Series(
        closes, index=pd.DatetimeIndex(index), dtype="float64"
    ).sort_index()


def _priced_symbols(rows: list[dict[str, Any]]) -> list[str]:
    return sorted(
        {
            row["symbol"]
            for row in rows
            if not row["is_cash"] and row.get("symbol") and row.get("quantity")
        }
    )


def _is_cash(row: Mapping[str, Any]) -> bool:
    asset_type = str(row.get("asset_type") or "").upper()
    symbol = str(row.get("symbol") or "").upper()
    return asset_type in {"CASH", "CASH_EQUIVALENT", "MONEY_MARKET"} or symbol in {
        "CASH",
        "CASH_EQUIVALENT",
    }


def _market_value(row: Mapping[str, Any]) -> float:
    value = _float(row.get("market_value"))
    if value is not None:
        return value
    quantity = _float(row.get("quantity")) or 0.0
    price = _float(row.get("current_price")) or 0.0
    return quantity * price


def _float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _zero_metrics(lookback: Lookback) -> dict[str, Any]:
    return {
        "lookback": lookback,
        "weights": {"CASH": 1.0},
        "total_return": 0.0,
        "annualized_volatility": 0.0,
        "sharpe_ratio": 0.0,
        "beta": 0.0,
        "max_drawdown": 0.0,
    }


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()
