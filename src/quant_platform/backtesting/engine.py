"""Minimal backtesting engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd  # type: ignore[import-untyped]

from quant_platform.backtesting.metrics import compute_metrics
from quant_platform.backtesting.schemas import BacktestConfig, BacktestResult


@dataclass
class _Position:
    quantity: float = 0.0
    entry_price: float = 0.0
    entry_timestamp: Any = None
    bars_held: int = 0


def _required_columns(config: BacktestConfig) -> set[str]:
    return {
        config.timestamp_column,
        config.symbol_column,
        config.price_column,
        config.signal_column,
    }


def _fill_price(row: pd.Series, side: float, config: BacktestConfig) -> float:
    price = float(row[config.price_column])
    if config.bid_ask.enabled:
        ask_column = config.bid_ask.ask_column
        bid_column = config.bid_ask.bid_column
        if side > 0 and ask_column in row and pd.notna(row[ask_column]):
            price = float(row[ask_column])
        if side < 0 and bid_column in row and pd.notna(row[bid_column]):
            price = float(row[bid_column])
    if config.slippage.enabled and config.slippage.basis_points:
        signed_bps = config.slippage.basis_points * (1 if side > 0 else -1)
        multiplier = 1.0 + signed_bps / 10_000.0
        price *= multiplier
    return price


def _commission(notional: float, config: BacktestConfig) -> float:
    if not config.commission.enabled:
        return 0.0
    variable_fee = abs(notional) * config.commission.basis_points / 10_000.0
    return config.commission.per_trade + variable_fee


def _target_quantity(row: pd.Series, equity: float, config: BacktestConfig) -> float:
    signal = float(row[config.signal_column])
    price = float(row[config.price_column])
    if price <= 0 or signal == 0:
        return 0.0
    notional = config.position_sizing.fixed_notional * abs(signal)
    if config.position_sizing.mode.value == "equal_weight":
        notional = equity * config.exposure.max_gross_exposure * abs(signal)
    elif config.position_sizing.mode.value == "target_weight":
        weight = float(row.get(config.position_sizing.target_weight_column, signal))
        notional = equity * min(abs(weight), config.exposure.max_gross_exposure)
        signal = 1.0 if weight > 0 else -1.0 if weight < 0 else 0.0
    notional = min(notional, equity * config.exposure.max_gross_exposure)
    return (notional / price) * (1.0 if signal > 0 else -1.0)


def run_backtest(
    market_data: pd.DataFrame,
    *,
    config: BacktestConfig | None = None,
) -> BacktestResult:
    """Run a simple bar-by-bar backtest over signal-enriched market data.

    The input must contain timestamp, symbol, price, and signal columns. Signals are
    interpreted as target direction/exposure and converted into target quantities
    using the configured placeholder position-sizing model.
    """

    config = config or BacktestConfig()
    missing = _required_columns(config) - set(market_data.columns)
    if missing:
        raise ValueError(f"market_data missing required columns: {sorted(missing)}")

    data = market_data.sort_values([config.timestamp_column, config.symbol_column])
    cash = config.initial_cash
    positions: dict[str, _Position] = {}
    latest_prices: dict[str, float] = {}
    equity_rows: list[dict[str, Any]] = []
    trade_rows: list[dict[str, Any]] = []

    for timestamp, frame in data.groupby(config.timestamp_column, sort=True):
        mark_to_market = sum(
            position.quantity * latest_prices.get(symbol, position.entry_price)
            for symbol, position in positions.items()
        )
        equity = cash + mark_to_market
        for _, row in frame.iterrows():
            symbol = str(row[config.symbol_column])
            latest_prices[symbol] = float(row[config.price_column])
            position = positions.setdefault(symbol, _Position())
            if config.holding_period.max_bars is not None and position.quantity != 0:
                position.bars_held += 1
                if position.bars_held >= config.holding_period.max_bars:
                    row = row.copy()
                    row[config.signal_column] = 0.0
            target_quantity = _target_quantity(row, equity, config)
            delta = target_quantity - position.quantity
            if abs(delta) <= 1e-12:
                continue
            side = 1.0 if delta > 0 else -1.0
            fill_price = _fill_price(row, side, config)
            notional = delta * fill_price
            fee = _commission(notional, config)
            realized_pnl = None
            if position.quantity != 0 and delta * position.quantity < 0:
                closing_quantity = min(abs(delta), abs(position.quantity))
                direction = 1.0 if position.quantity > 0 else -1.0
                gross_pnl = (
                    closing_quantity
                    * (fill_price - position.entry_price)
                    * direction
                )
                realized_pnl = gross_pnl - fee
            cash -= notional + fee
            new_quantity = position.quantity + delta
            if new_quantity == 0:
                position.entry_price = 0.0
                position.entry_timestamp = None
                position.bars_held = 0
            elif position.quantity == 0 or position.quantity * new_quantity < 0:
                position.entry_price = fill_price
                position.entry_timestamp = timestamp
                position.bars_held = 0
            position.quantity = new_quantity
            trade_rows.append(
                {
                    "timestamp": timestamp,
                    "symbol": symbol,
                    "quantity": delta,
                    "price": fill_price,
                    "notional": notional,
                    "commission": fee,
                    "pnl": realized_pnl,
                }
            )
        market_value = sum(
            position.quantity * latest_prices.get(symbol, position.entry_price)
            for symbol, position in positions.items()
        )
        equity_rows.append(
            {
                "timestamp": timestamp,
                "cash": cash,
                "market_value": market_value,
                "equity": cash + market_value,
            }
        )

    equity_curve = pd.DataFrame(equity_rows)
    trades = pd.DataFrame(trade_rows)
    metrics = compute_metrics(
        equity_curve, trades, annualization_factor=config.annualization_factor
    )
    return BacktestResult(
        equity_curve=equity_curve,
        trades=trades,
        metrics=metrics,
        config=config,
    )
