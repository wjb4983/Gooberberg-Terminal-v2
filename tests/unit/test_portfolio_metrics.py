"""Deterministic tests for portfolio analytics."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from quant_platform.portfolio.metrics import (
    LOOKBACKS,
    annualized_volatility,
    beta_vs_benchmark,
    compute_portfolio_metrics,
    daily_portfolio_returns,
    portfolio_weights,
    sharpe_ratio,
    slice_lookback,
)


def test_lookbacks_are_in_required_order() -> None:
    assert LOOKBACKS == ("1M", "3M", "6M", "YTD", "1Y", "3Y", "MAX")


def test_weights_include_cash_and_market_values() -> None:
    weights = portfolio_weights(
        {"AAA": 2.0, "BBB": 1.0},
        pd.Series({"AAA": 10.0, "BBB": 20.0}),
        cash=10.0,
    )

    assert weights == {"AAA": 0.4, "BBB": 0.4, "CASH": 0.2}


def test_cash_only_portfolio_has_zero_returns_and_zero_metrics() -> None:
    prices = pd.DataFrame(
        {"AAA": [10.0, 11.0, 12.0]},
        index=pd.date_range("2024-01-01", periods=3, freq="D"),
    )

    returns = daily_portfolio_returns(prices, {}, cash=100.0)
    metrics = compute_portfolio_metrics(prices, {}, cash=100.0)

    assert returns.tolist() == [0.0, 0.0]
    assert metrics.weights == {"CASH": 1.0}
    assert metrics.total_return == 0.0
    assert metrics.annualized_volatility == 0.0
    assert metrics.sharpe_ratio == 0.0
    assert metrics.beta == 0.0
    assert metrics.max_drawdown == 0.0


def test_daily_returns_total_return_drawdown_volatility_and_sharpe() -> None:
    prices = pd.DataFrame(
        {"AAA": [100.0, 110.0, 99.0, 121.0]},
        index=pd.date_range("2024-01-01", periods=4, freq="D"),
    )

    returns = daily_portfolio_returns(prices, {"AAA": 1.0})
    metrics = compute_portfolio_metrics(prices, {"AAA": 1.0})

    expected_returns = pd.Series([0.10, -0.10, 22.0 / 99.0], index=returns.index)
    pd.testing.assert_series_equal(returns, expected_returns, check_names=False)
    assert metrics.total_return == pytest.approx(0.21)
    assert metrics.max_drawdown == pytest.approx(-0.10)
    assert metrics.annualized_volatility == pytest.approx(
        float(expected_returns.std(ddof=1)) * math.sqrt(252)
    )
    assert metrics.sharpe_ratio == pytest.approx(
        float(expected_returns.mean())
        / float(expected_returns.std(ddof=1))
        * math.sqrt(252)
    )


def test_missing_prices_raise_for_held_assets() -> None:
    prices = pd.DataFrame(
        {"AAA": [100.0, None, 102.0]},
        index=pd.date_range("2024-01-01", periods=3, freq="D"),
    )

    with pytest.raises(ValueError, match="Missing price observations"):
        daily_portfolio_returns(prices, {"AAA": 1.0})

    with pytest.raises(ValueError, match="Missing prices for held symbols"):
        daily_portfolio_returns(prices, {"BBB": 1.0})


def test_single_day_and_zero_volatility_histories_return_zero_risk_metrics() -> None:
    one_day = pd.DataFrame({"AAA": [100.0]}, index=[pd.Timestamp("2024-01-01")])
    flat = pd.Series([0.01, 0.01], index=pd.date_range("2024-01-02", periods=2))

    assert daily_portfolio_returns(one_day, {"AAA": 1.0}).empty
    assert compute_portfolio_metrics(one_day, {"AAA": 1.0}).total_return == 0.0
    assert annualized_volatility(flat) == 0.0
    assert sharpe_ratio(flat) == 0.0


def test_beta_aligns_inner_dates_against_benchmark() -> None:
    portfolio_returns = pd.Series(
        [0.01, 0.02, 0.04],
        index=pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
    )
    benchmark_prices = pd.Series(
        [100.0, 110.0, 132.0, 184.8],
        index=pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"]),
    )

    assert beta_vs_benchmark(portfolio_returns, benchmark_prices) == pytest.approx(0.1)


def test_compute_metrics_beta_uses_lookback_sliced_benchmark() -> None:
    dates = pd.date_range("2024-01-01", periods=4, freq="D")
    prices = pd.DataFrame({"AAA": [100.0, 110.0, 121.0, 133.1]}, index=dates)
    benchmark = pd.Series([50.0, 55.0, 60.5, 66.55], index=dates)

    metrics = compute_portfolio_metrics(
        prices,
        {"AAA": 1.0},
        benchmark_prices=benchmark,
        lookback="MAX",
    )

    assert metrics.beta == pytest.approx(1.0)


def test_slice_lookback_uses_calendar_boundaries() -> None:
    dates = pd.to_datetime(["2023-12-31", "2024-01-01", "2024-03-31", "2024-04-30"])
    series = pd.Series([1.0, 2.0, 3.0, 4.0], index=dates)

    assert slice_lookback(series, "1M").index.tolist() == [
        pd.Timestamp("2024-03-31"),
        pd.Timestamp("2024-04-30"),
    ]
    assert slice_lookback(series, "3M").index.tolist() == [
        pd.Timestamp("2024-03-31"),
        pd.Timestamp("2024-04-30"),
    ]
    assert slice_lookback(series, "YTD").index.tolist() == [
        pd.Timestamp("2024-01-01"),
        pd.Timestamp("2024-03-31"),
        pd.Timestamp("2024-04-30"),
    ]
    assert slice_lookback(series, "MAX").equals(series)


def test_unsupported_lookback_raises() -> None:
    series = pd.Series([1.0], index=[pd.Timestamp("2024-01-01")])

    with pytest.raises(ValueError, match="Unsupported lookback"):
        slice_lookback(series, "2Y")  # type: ignore[arg-type]
