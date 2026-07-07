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
    STRATEGY_REGISTRY,
    OptimizedPortfolioResult,
    PortfolioOptimizationStrategy,
    StrategyExecutionInput,
)
from quant_platform.portfolio.service import (
    DEFAULT_BENCHMARK_SYMBOL,
    PortfolioService,
    _history_to_series,
)

PLACEHOLDER_WARNING = (
    "Optimization backend is not implemented yet; result uses current allocation "
    "or mock weights."
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
        universe_symbols: Iterable[str] | None = None,
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
        universe = _optimization_universe_from_summary(summary, universe_symbols)
        prices, history_warnings = self._price_history_for_universe(
            summary,
            universe["symbols"],
            benchmark_symbol,
            candidate_symbols=_candidate_symbols_from_universe(universe),
        )
        universe = _filter_universe_for_price_history(universe, prices)
        metrics = summary.get("lookback_metrics", {}).get(lookback, {})
        base_warnings = _base_warnings(
            summary, strategy_input, lookback, history_warnings
        )
        base_constraints = {
            "benchmark_symbol": benchmark_symbol,
            "lookback": lookback,
            "long_only": True,
            "max_leverage": 1.0,
            "input_positions": strategy_input["positions"],
            "input_holdings": strategy_input["holdings"],
            "universe": universe,
        }
        execution_input = StrategyExecutionInput(
            prices=prices,
            current_weights=strategy_input["weights"],
            lookback_metrics=metrics,
            benchmark_data=summary.get("benchmark", {}),
            constraints=base_constraints,
            universe=universe,
        )

        results: list[OptimizedPortfolioResult] = []
        for strategy_id in strategy_list:
            entry = STRATEGY_REGISTRY[strategy_id]
            runner_result = entry.runner(execution_input)
            warnings = [*base_warnings, *runner_result.warnings]
            if runner_result.is_placeholder:
                warnings.insert(0, PLACEHOLDER_WARNING)
            target_weights = runner_result.target_weights
            results.append(
                OptimizedPortfolioResult(
                    strategy_id=strategy_id,
                    strategy_name=entry.metadata.strategy_name,
                    target_weights=target_weights,
                    expected_return=_metric(metrics, "total_return"),
                    volatility=_metric(metrics, "annualized_volatility"),
                    sharpe=_metric(metrics, "sharpe_ratio"),
                    max_drawdown=_metric(metrics, "max_drawdown"),
                    turnover=0.0,
                    leverage=_portfolio_leverage(target_weights),
                    warnings=_dedupe(warnings),
                    constraints={**base_constraints, **runner_result.constraints},
                    is_placeholder=runner_result.is_placeholder,
                )
            )
        return results

    def _price_history_for_universe(
        self,
        summary: dict[str, Any],
        symbols: Iterable[str],
        benchmark_symbol: str,
        candidate_symbols: Iterable[str] = (),
    ) -> tuple[dict[str, Any], list[str]]:
        """Return price histories for current holdings and candidate symbols."""

        del benchmark_symbol  # Reserved for parity with benchmark-aware callers.
        prices = dict(summary.get("price_history") or {})
        warnings: list[str] = []
        candidate_symbol_set = set(_normalize_symbols(candidate_symbols))
        missing_symbols = [
            symbol
            for symbol in _normalize_symbols(symbols)
            if symbol not in prices and symbol != CASH_SYMBOL
        ]
        for symbol in missing_symbols:
            try:
                payload, _refreshed_at, _from_cache = (
                    self.portfolio_service._cached_historical_prices(symbol)
                )
                history = _history_to_series(payload)
                if history.empty:
                    raise ValueError("price history contained no candles")
                prices[symbol] = history
            except Exception:  # noqa: BLE001 - skip symbol, keep response
                if symbol in candidate_symbol_set:
                    warnings.append(
                        f"Candidate symbol {symbol} is missing price history."
                    )
        return prices, _dedupe(warnings)


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
    holdings = [_holding_input(row, weights) for row in summary.get("holdings", [])]
    positions = {
        holding["symbol"]: holding["quantity"]
        for holding in holdings
        if not holding["is_cash"] and holding["quantity"] != 0.0
    }
    return {"holdings": holdings, "positions": positions, "weights": weights}


def _optimization_universe_from_summary(
    summary: dict[str, Any], universe_symbols: Iterable[str] | None
) -> dict[str, Any]:
    """Build the normalized optimization universe for strategy runners."""

    current_holding_symbols = _non_cash_holding_symbols(summary)
    candidate_symbols = _normalize_symbols(universe_symbols)
    candidate_only_symbols = [
        symbol for symbol in candidate_symbols if symbol not in current_holding_symbols
    ]
    full_universe_symbols = _dedupe([*current_holding_symbols, *candidate_symbols])
    symbol_metadata = {
        symbol: {
            "symbol": symbol,
            "is_current_holding": symbol in current_holding_symbols,
            "is_candidate": symbol in candidate_symbols,
        }
        for symbol in full_universe_symbols
    }
    return {
        "current_holding_symbols": current_holding_symbols,
        "candidate_only_symbols": candidate_only_symbols,
        "symbols": full_universe_symbols,
        "metadata": symbol_metadata,
    }


def _filter_universe_for_price_history(
    universe: dict[str, Any], prices: dict[str, Any]
) -> dict[str, Any]:
    """Keep only symbols with histories in the executable optimizer universe."""

    requested_symbols = list(universe.get("symbols", []))
    executable_symbols = [symbol for symbol in requested_symbols if symbol in prices]
    rejected_symbols = [symbol for symbol in requested_symbols if symbol not in prices]
    return {
        **universe,
        "candidate_only_symbols": [
            symbol
            for symbol in universe.get("candidate_only_symbols", [])
            if symbol in prices
        ],
        "symbols": executable_symbols,
        "requested_symbols": requested_symbols,
        "rejected_symbols": rejected_symbols,
    }


def _candidate_symbols_from_universe(universe: dict[str, Any]) -> list[str]:
    return [
        symbol
        for symbol, metadata in universe.get("metadata", {}).items()
        if metadata.get("is_candidate")
    ]


def _non_cash_holding_symbols(summary: dict[str, Any]) -> list[str]:
    symbols: list[str] = []
    for row in summary.get("holdings", []):
        if row.get("is_cash"):
            continue
        symbol = _normalize_symbol(row.get("symbol"))
        if symbol and symbol != CASH_SYMBOL:
            symbols.append(symbol)
    return _dedupe(symbols)


def _normalize_symbols(symbols: Iterable[str] | None) -> list[str]:
    if symbols is None:
        return []
    return _dedupe(
        symbol
        for raw_symbol in symbols
        if (symbol := _normalize_symbol(raw_symbol)) and symbol != CASH_SYMBOL
    )


def _normalize_symbol(symbol: Any) -> str:
    return str(symbol or "").strip().upper()


def _holding_input(row: dict[str, Any], weights: dict[str, float]) -> dict[str, Any]:
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
    summary: dict[str, Any],
    strategy_input: dict[str, Any],
    lookback: Lookback,
    history_warnings: Iterable[str] = (),
) -> list[str]:
    summary_warnings = summary.get("warnings", [])
    warnings = [
        str(warning.get("message") or warning.get("code"))
        for warning in summary_warnings
    ]
    warnings.extend(history_warnings)
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


def _portfolio_leverage(weights: dict[str, float]) -> float:
    return round(sum(abs(weight) for weight in weights.values()), 12)


def _metric(metrics: dict[str, Any], key: str) -> float | None:
    value = metrics.get(key)
    return float(value) if value is not None else None


def _dedupe(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


def results_to_dicts(
    results: Iterable[OptimizedPortfolioResult],
) -> list[dict[str, Any]]:
    """Serialize optimization results for API callers that expect dictionaries."""

    return [asdict(result) for result in results]


__all__ = ["PortfolioOptimizationService", "PLACEHOLDER_WARNING", "results_to_dicts"]
