"""Performance metrics for backtest outputs."""

from __future__ import annotations

import math

import pandas as pd  # type: ignore[import-untyped]


def total_return(equity_curve: pd.DataFrame, equity_column: str = "equity") -> float:
    """Compute total return from the first to last equity value."""

    if equity_curve.empty:
        return 0.0
    start = float(equity_curve[equity_column].iloc[0])
    end = float(equity_curve[equity_column].iloc[-1])
    if start == 0:
        return 0.0
    return end / start - 1.0


def sharpe_ratio(
    equity_curve: pd.DataFrame,
    *,
    equity_column: str = "equity",
    annualization_factor: int = 252,
) -> float:
    """Compute an annualized Sharpe ratio from equity returns."""

    if len(equity_curve) < 2:
        return 0.0
    returns = equity_curve[equity_column].astype(float).pct_change().dropna()
    std = float(returns.std(ddof=1))
    if returns.empty or std == 0.0 or math.isnan(std):
        return 0.0
    return float(returns.mean() / std * math.sqrt(annualization_factor))


def max_drawdown(equity_curve: pd.DataFrame, equity_column: str = "equity") -> float:
    """Compute maximum drawdown as a negative fraction."""

    if equity_curve.empty:
        return 0.0
    equity = equity_curve[equity_column].astype(float)
    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    return float(drawdown.min())


def win_rate(trades: pd.DataFrame, pnl_column: str = "pnl") -> float:
    """Compute the fraction of closed trades with positive PnL."""

    if trades.empty or pnl_column not in trades:
        return 0.0
    closed = trades[trades[pnl_column].notna()]
    if closed.empty:
        return 0.0
    return float((closed[pnl_column] > 0).mean())


def turnover(
    trades: pd.DataFrame,
    equity_curve: pd.DataFrame,
    *,
    notional_column: str = "notional",
    equity_column: str = "equity",
) -> float:
    """Compute traded notional divided by average equity."""

    if trades.empty or equity_curve.empty or notional_column not in trades:
        return 0.0
    average_equity = float(equity_curve[equity_column].mean())
    if average_equity == 0.0:
        return 0.0
    return float(trades[notional_column].abs().sum() / average_equity)


def compute_metrics(
    equity_curve: pd.DataFrame,
    trades: pd.DataFrame,
    *,
    annualization_factor: int = 252,
) -> dict[str, float | int]:
    """Compute the initial metric set for a completed backtest."""

    return {
        "total_return": total_return(equity_curve),
        "sharpe": sharpe_ratio(
            equity_curve, annualization_factor=annualization_factor
        ),
        "max_drawdown": max_drawdown(equity_curve),
        "win_rate": win_rate(trades),
        "turnover": turnover(trades, equity_curve),
        "number_of_trades": int(len(trades)),
    }
