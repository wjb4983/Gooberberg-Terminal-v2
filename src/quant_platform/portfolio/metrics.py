"""Deterministic portfolio analytics for marked-to-market positions.

The functions in this module intentionally define conservative edge-case behavior:
missing prices for held assets raise ``ValueError``; cash-only, single-day, and
zero-volatility inputs return zero risk/return metrics instead of NaN; benchmark
beta is computed only on the inner intersection of portfolio and benchmark return
dates.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final, Literal

import pandas as pd  # type: ignore[import-untyped]

Lookback = Literal["1M", "3M", "6M", "YTD", "1Y", "3Y", "MAX"]
LOOKBACKS: Final[tuple[Lookback, ...]] = ("1M", "3M", "6M", "YTD", "1Y", "3Y", "MAX")
TRADING_DAYS_PER_YEAR: Final[int] = 252
CASH_SYMBOL: Final[str] = "CASH"


@dataclass(frozen=True)
class PortfolioMetrics:
    """Summary metrics for one portfolio lookback window."""

    lookback: Lookback
    weights: dict[str, float]
    total_return: float
    annualized_volatility: float
    sharpe_ratio: float
    beta: float
    max_drawdown: float


def _as_datetime_index(prices: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    if not isinstance(prices.index, pd.DatetimeIndex):
        prices = prices.copy()
        prices.index = pd.to_datetime(prices.index)
    return prices.sort_index()


def _held_symbols(positions: dict[str, float]) -> list[str]:
    return [symbol for symbol, quantity in positions.items() if quantity != 0.0]


def _validate_prices(prices: pd.DataFrame, positions: dict[str, float]) -> pd.DataFrame:
    normalized = _as_datetime_index(prices)
    assert isinstance(normalized, pd.DataFrame)
    held = _held_symbols(positions)
    missing_columns = [symbol for symbol in held if symbol not in normalized.columns]
    if missing_columns:
        raise ValueError(
            f"Missing prices for held symbols: {', '.join(missing_columns)}"
        )
    held_prices = normalized[held] if held else normalized.iloc[:, 0:0]
    if held and held_prices.isna().any().any():
        raise ValueError("Missing price observations for held symbols")
    return normalized.astype(float)


def portfolio_weights(
    positions: dict[str, float],
    latest_prices: pd.Series | dict[str, float],
    *,
    cash: float = 0.0,
) -> dict[str, float]:
    """Compute current market-value weights, including cash when non-zero.

    A cash-only or zero-value portfolio is represented as ``{"CASH": 1.0}``.
    Missing prices for non-zero positions raise ``ValueError``.
    """

    prices = pd.Series(latest_prices, dtype="float64")
    values: dict[str, float] = {}
    for symbol, quantity in positions.items():
        if quantity == 0.0:
            continue
        if symbol not in prices or pd.isna(prices[symbol]):
            raise ValueError(f"Missing price for held symbol: {symbol}")
        values[symbol] = float(quantity) * float(prices[symbol])
    if cash != 0.0:
        values[CASH_SYMBOL] = float(cash)
    total_value = sum(values.values())
    if total_value == 0.0:
        return {CASH_SYMBOL: 1.0}
    return {symbol: value / total_value for symbol, value in values.items()}


def portfolio_value_series(
    prices: pd.DataFrame, positions: dict[str, float], *, cash: float = 0.0
) -> pd.Series:
    """Return daily marked-to-market portfolio value."""

    validated = _validate_prices(prices, positions)
    held = _held_symbols(positions)
    if not held:
        return pd.Series(float(cash), index=validated.index, name="portfolio_value")
    quantities = pd.Series(
        {symbol: positions[symbol] for symbol in held}, dtype="float64"
    )
    values = validated[held].mul(quantities, axis="columns").sum(
        axis="columns"
    ) + float(cash)
    values.name = "portfolio_value"
    return values


def daily_portfolio_returns(
    prices: pd.DataFrame, positions: dict[str, float], *, cash: float = 0.0
) -> pd.Series:
    """Compute close-to-close daily portfolio returns.

    Single-day histories produce an empty series. Cash-only portfolios produce
    zero returns for each available return date.
    """

    values = portfolio_value_series(prices, positions, cash=cash)
    returns = values.pct_change().dropna()
    returns.name = "portfolio_return"
    return returns


def annualized_volatility(
    returns: pd.Series, *, annualization_factor: int = TRADING_DAYS_PER_YEAR
) -> float:
    """Compute sample annualized volatility, returning zero when undefined."""

    if len(returns) < 2:
        return 0.0
    std = float(returns.astype(float).std(ddof=1))
    if std == 0.0 or math.isnan(std):
        return 0.0
    return std * math.sqrt(annualization_factor)


def sharpe_ratio(
    returns: pd.Series,
    *,
    annualization_factor: int = TRADING_DAYS_PER_YEAR,
    risk_free_rate: float = 0.0,
) -> float:
    """Compute annualized Sharpe ratio, returning zero for zero-volatility returns."""

    volatility = annualized_volatility(
        returns, annualization_factor=annualization_factor
    )
    if volatility == 0.0:
        return 0.0
    daily_risk_free = risk_free_rate / annualization_factor
    excess_daily_return = float((returns.astype(float) - daily_risk_free).mean())
    return excess_daily_return * annualization_factor / volatility


def beta_vs_benchmark(
    portfolio_returns: pd.Series, benchmark_prices: pd.Series | None
) -> float:
    """Compute beta using dates shared by portfolio returns and benchmark returns."""

    if benchmark_prices is None or len(portfolio_returns) < 2:
        return 0.0
    benchmark = _as_datetime_index(benchmark_prices.astype(float))
    assert isinstance(benchmark, pd.Series)
    if benchmark.isna().any():
        raise ValueError("Missing benchmark price observations")
    benchmark_returns = benchmark.pct_change().dropna()
    aligned = pd.concat(
        [portfolio_returns.astype(float), benchmark_returns],
        axis="columns",
        join="inner",
    ).dropna()
    if len(aligned) < 2:
        return 0.0
    benchmark_variance = float(aligned.iloc[:, 1].var(ddof=1))
    if benchmark_variance == 0.0 or math.isnan(benchmark_variance):
        return 0.0
    covariance = float(aligned.iloc[:, 0].cov(aligned.iloc[:, 1]))
    return covariance / benchmark_variance


def total_return(values: pd.Series) -> float:
    """Compute total return from first to last portfolio value."""

    if len(values) < 2:
        return 0.0
    start = float(values.iloc[0])
    if start == 0.0:
        return 0.0
    return float(values.iloc[-1]) / start - 1.0


def max_drawdown(values: pd.Series) -> float:
    """Compute maximum drawdown as a non-positive fraction."""

    if values.empty:
        return 0.0
    running_max = values.astype(float).cummax()
    drawdowns = values.astype(float) / running_max - 1.0
    minimum = float(drawdowns.min())
    if math.isnan(minimum):
        return 0.0
    return min(minimum, 0.0)


def slice_lookback(
    data: pd.DataFrame | pd.Series, lookback: Lookback
) -> pd.DataFrame | pd.Series:
    """Slice a time series to a supported lookback window.

    Supported lookbacks are evaluated in the display order ``1M``, ``3M``,
    ``6M``, ``YTD``, ``1Y``, ``3Y``, and ``MAX``. Calendar windows are inclusive
    of the computed start date. ``YTD`` begins on January 1 of the last date's
    year, and ``MAX`` returns the full history.
    """

    if lookback not in LOOKBACKS:
        raise ValueError(f"Unsupported lookback: {lookback}")
    normalized = _as_datetime_index(data)
    if normalized.empty or lookback == "MAX":
        return normalized
    end = normalized.index.max()
    if lookback == "YTD":
        start = pd.Timestamp(year=end.year, month=1, day=1)
    else:
        offsets = {
            "1M": pd.DateOffset(months=1),
            "3M": pd.DateOffset(months=3),
            "6M": pd.DateOffset(months=6),
            "1Y": pd.DateOffset(years=1),
            "3Y": pd.DateOffset(years=3),
        }
        start = end - offsets[lookback]
    return normalized.loc[normalized.index >= start]


def compute_portfolio_metrics(
    prices: pd.DataFrame,
    positions: dict[str, float],
    *,
    cash: float = 0.0,
    benchmark_prices: pd.Series | None = None,
    lookback: Lookback = "MAX",
    annualization_factor: int = TRADING_DAYS_PER_YEAR,
    risk_free_rate: float = 0.0,
) -> PortfolioMetrics:
    """Compute portfolio weights, return, risk, beta, and drawdown for a lookback."""

    sliced_prices = slice_lookback(prices, lookback)
    assert isinstance(sliced_prices, pd.DataFrame)
    values = portfolio_value_series(sliced_prices, positions, cash=cash)
    returns = values.pct_change().dropna()
    latest_prices = (
        sliced_prices.iloc[-1]
        if not sliced_prices.empty
        else pd.Series(dtype="float64")
    )
    return PortfolioMetrics(
        lookback=lookback,
        weights=portfolio_weights(positions, latest_prices, cash=cash),
        total_return=total_return(values),
        annualized_volatility=annualized_volatility(
            returns, annualization_factor=annualization_factor
        ),
        sharpe_ratio=sharpe_ratio(
            returns,
            annualization_factor=annualization_factor,
            risk_free_rate=risk_free_rate,
        ),
        beta=beta_vs_benchmark(
            returns,
            (
                slice_lookback(benchmark_prices, lookback)
                if benchmark_prices is not None
                else None
            ),
        ),
        max_drawdown=max_drawdown(values),
    )
