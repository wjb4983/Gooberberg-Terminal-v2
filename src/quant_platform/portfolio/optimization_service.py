"""Portfolio optimization orchestration service.

This module bridges Schwab-backed portfolio summaries into the normalized payloads
used by optimization strategies. Real optimization backends are intentionally not
wired yet; strategy runs therefore return deterministic placeholder results that
preserve current allocations and surface implementation/data-quality warnings.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict
from typing import Any

from quant_platform.portfolio.metrics import CASH_SYMBOL, LOOKBACKS, Lookback
from quant_platform.portfolio.optimization import (
    STRATEGY_METADATA,
    OptimizedPortfolioResult,
    PortfolioOptimizationStrategy,
)
from quant_platform.portfolio.service import DEFAULT_BENCHMARK_SYMBOL, PortfolioService

PLACEHOLDER_WARNING = (
    "Optimization backend is not implemented yet; result uses current allocation "
    "or mock weights."
)
_COVARIANCE_STRATEGIES = frozenset(
    {
        PortfolioOptimizationStrategy.EQUAL_RISK_PARITY,
        PortfolioOptimizationStrategy.MEAN_VARIANCE_SHRINKAGE,
        PortfolioOptimizationStrategy.BLACK_LITTERMAN_MOMENTUM_VIEWS,
    }
)


class PortfolioOptimizationService:
    """Run portfolio optimization strategies against Schwab portfolio summaries."""

    def __init__(self, portfolio_service: PortfolioService | None = None) -> None:
        self.portfolio_service = portfolio_service or PortfolioService(provider=None)

    def run_selected_strategies(
        self,
        account_hashes: Iterable[str] | None = None,
        benchmark_symbol: str = DEFAULT_BENCHMARK_SYMBOL,
        lookback: Lookback = "MAX",
        strategy_ids: Iterable[str | PortfolioOptimizationStrategy] | None = None,
    ) -> list[OptimizedPortfolioResult]:
        """Return deterministic placeholder optimization results for strategies.

        The method intentionally reuses :class:`PortfolioService` so holdings,
        allocation, historical prices, benchmark data, and lookback metrics are
        fetched and computed through the same defensive Schwab infrastructure as
        the portfolio dashboard.
        """

        if lookback not in LOOKBACKS:
            raise ValueError(f"Unsupported lookback: {lookback}")

        summary = self.portfolio_service.summary(
            account_hashes=account_hashes,
            benchmark_symbol=benchmark_symbol,
        )
        strategy_list = _resolve_strategy_ids(strategy_ids)
        strategy_input = _strategy_input_from_summary(summary)
        current_weights = strategy_input["weights"]
        metrics = summary.get("lookback_metrics", {}).get(lookback, {})
        base_warnings = _base_warnings(summary, strategy_input, lookback)

        results: list[OptimizedPortfolioResult] = []
        for strategy_id in strategy_list:
            metadata = STRATEGY_METADATA[strategy_id]
            warnings = [PLACEHOLDER_WARNING, *base_warnings]
            if strategy_id in _COVARIANCE_STRATEGIES:
                warnings.append(
                    "Covariance estimates are unavailable until the optimization "
                    "backend is implemented."
                )
            target_weights = _target_weights_for_strategy(strategy_id, current_weights)
            results.append(
                OptimizedPortfolioResult(
                    strategy_id=strategy_id,
                    strategy_name=metadata.strategy_name,
                    target_weights=target_weights,
                    expected_return=_metric(metrics, "total_return"),
                    volatility=_metric(metrics, "annualized_volatility"),
                    sharpe=_metric(metrics, "sharpe_ratio"),
                    max_drawdown=_metric(metrics, "max_drawdown"),
                    turnover=0.0,
                    leverage=_portfolio_leverage(target_weights),
                    warnings=warnings,
                    constraints={
                        "benchmark_symbol": benchmark_symbol,
                        "lookback": lookback,
                        "long_only": True,
                        "max_leverage": 1.0,
                        "input_positions": strategy_input["positions"],
                        "input_holdings": strategy_input["holdings"],
                    },
                    is_placeholder=True,
                )
            )
        return results


def _resolve_strategy_ids(
    strategy_ids: Iterable[str | PortfolioOptimizationStrategy] | None,
) -> list[PortfolioOptimizationStrategy]:
    raw_ids = (
        list(strategy_ids) if strategy_ids is not None else list(STRATEGY_METADATA)
    )
    resolved: list[PortfolioOptimizationStrategy] = []
    for raw_id in raw_ids:
        strategy_id = (
            raw_id
            if isinstance(raw_id, PortfolioOptimizationStrategy)
            else PortfolioOptimizationStrategy(str(raw_id))
        )
        if strategy_id not in resolved:
            resolved.append(strategy_id)
    return resolved


def _strategy_input_from_summary(summary: dict[str, Any]) -> dict[str, Any]:
    weights = {
        str(row.get("symbol") or CASH_SYMBOL): round(
            float(row.get("weight") or 0.0), 12
        )
        for row in summary.get("allocation_by_symbol", [])
    }
    if not weights:
        weights = {CASH_SYMBOL: 1.0}
    holdings = [
        _holding_input(row, weights) for row in summary.get("holdings", [])
    ]
    positions = {
        holding["symbol"]: holding["quantity"]
        for holding in holdings
        if not holding["is_cash"] and holding["quantity"] != 0.0
    }
    return {"holdings": holdings, "positions": positions, "weights": weights}


def _holding_input(
    row: dict[str, Any], weights: dict[str, float]
) -> dict[str, Any]:
    symbol = CASH_SYMBOL if row.get("is_cash") else str(row.get("symbol") or "").upper()
    return {
        "symbol": symbol or CASH_SYMBOL,
        "asset_type": str(row.get("asset_type") or "UNKNOWN").upper(),
        "quantity": float(row.get("quantity") or 0.0),
        "market_value": float(row.get("market_value") or 0.0),
        "weight": weights.get(symbol or CASH_SYMBOL, 0.0),
        "is_cash": bool(row.get("is_cash")),
    }


def _base_warnings(
    summary: dict[str, Any], strategy_input: dict[str, Any], lookback: Lookback
) -> list[str]:
    summary_warnings = summary.get("warnings", [])
    warnings = [
        str(warning.get("message") or warning.get("code"))
        for warning in summary_warnings
    ]
    weights = strategy_input["weights"]
    if set(weights) == {CASH_SYMBOL}:
        warnings.append("Portfolio is cash-only; optimization keeps 100% cash.")
    if any(
        warning.get("code") in {"price_history_failed", "metrics_symbol_skipped"}
        for warning in summary_warnings
    ):
        warnings.append("One or more holdings are missing price histories.")
    metrics = summary.get("lookback_metrics", {}).get(lookback)
    if not metrics:
        warnings.append(f"Incomplete lookback data for {lookback}.")
    warnings.append("Leverage assumes a long-only portfolio with max leverage of 1.0.")
    return _dedupe(warnings)


def _target_weights_for_strategy(
    strategy_id: PortfolioOptimizationStrategy, current_weights: dict[str, float]
) -> dict[str, float]:
    if strategy_id == PortfolioOptimizationStrategy.FIXED_SP500_KMLM:
        return {"KMLM": 0.4, "SPY": 0.6}
    return dict(sorted(current_weights.items()))


def _portfolio_leverage(weights: dict[str, float]) -> float:
    return round(sum(abs(weight) for weight in weights.values()), 12)


def _metric(metrics: dict[str, Any], key: str) -> float | None:
    value = metrics.get(key)
    return float(value) if value is not None else None


def _dedupe(values: list[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


def results_to_dicts(
    results: Iterable[OptimizedPortfolioResult],
) -> list[dict[str, Any]]:
    """Serialize optimization results for API callers that expect dictionaries."""

    return [asdict(result) for result in results]


__all__ = ["PortfolioOptimizationService", "PLACEHOLDER_WARNING", "results_to_dicts"]
