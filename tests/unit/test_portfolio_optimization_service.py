from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import polars as pl
import pytest

from quant_platform.portfolio.optimization import (
    STRATEGY_REGISTRY,
    PortfolioOptimizationStrategy,
    StrategyExecutionInput,
)
from quant_platform.portfolio.optimization_service import (
    PortfolioOptimizationService,
)
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

    def account_numbers(self) -> list[Account]:
        return [Account(account_hash=account_hash) for account_hash in self.holdings]

    def account_holdings(self, account_hash: str) -> pl.DataFrame:
        return pl.DataFrame(self.holdings.get(account_hash, []))

    def historical_prices(self, symbol: str, **_kwargs: Any) -> dict[str, Any]:
        symbol = symbol.upper()
        self.history_calls.append(symbol)
        if f"history:{symbol}" in self.failures:
            raise RuntimeError("history unavailable")
        return self.histories.get(symbol, {"candles": []})


def candles(*closes: float) -> dict[str, Any]:
    return {
        "candles": [
            {"datetime": 1_704_067_200_000 + index * 86_400_000, "close": close}
            for index, close in enumerate(closes)
        ]
    }


def test_run_selected_strategies_reuses_summary_and_returns_current_weights() -> None:
    provider = FakeSchwabProvider(
        holdings={
            "hash-1": [
                {
                    "symbol": "AAA",
                    "asset_type": "EQUITY",
                    "quantity": 2.0,
                    "market_value": 220.0,
                },
                {
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

    service = PortfolioOptimizationService(portfolio_service=PortfolioService(provider))
    result = service.run_selected_strategies(
        account_hashes=["hash-1"],
        benchmark_symbol="SPY",
        lookback="MAX",
        strategy_ids=[PortfolioOptimizationStrategy.BUY_AND_HOLD],
    )[0]

    assert result.is_placeholder is False
    assert result.warnings == [
        "Leverage assumes a long-only portfolio with max leverage of 1.0.",
    ]
    assert result.target_weights == {
        "AAA": pytest.approx(220.0 / 270.0),
        "CASH": pytest.approx(50.0 / 270.0),
    }
    assert result.turnover == 0.0
    assert result.leverage == 1.0
    assert result.constraints["input_positions"] == {"AAA": 2.0}
    assert provider.history_calls == ["AAA", "SPY"]


def test_run_selected_strategies_returns_deterministic_mock_fixed_weights() -> None:
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
    result = PortfolioOptimizationService(
        PortfolioService(provider)
    ).run_selected_strategies(strategy_ids=["fixed_sp500_kmlm"])[0]

    assert result.strategy_id == PortfolioOptimizationStrategy.FIXED_SP500_KMLM
    assert result.target_weights == {"KMLM": 0.4, "SPY": 0.6}
    assert result.is_placeholder is False
    assert result.leverage == 1.0
    assert result.constraints["fixed_weights"] == {"SPY": 0.6, "KMLM": 0.4}


def test_run_selected_strategies_surfaces_data_quality_warnings() -> None:
    provider = FakeSchwabProvider(
        holdings={
            "hash-1": [
                {
                    "symbol": "BBB",
                    "asset_type": "EQUITY",
                    "quantity": 1.0,
                    "market_value": 20.0,
                }
            ]
        },
        histories={"SPY": candles(100.0, 101.0)},
        failures={"history:BBB"},
    )
    result = PortfolioOptimizationService(
        PortfolioService(provider)
    ).run_selected_strategies(
        lookback="MAX",
        strategy_ids=[PortfolioOptimizationStrategy.MEAN_VARIANCE_SHRINKAGE],
    )[0]

    assert "One or more holdings are missing price histories." in result.warnings
    assert (
        "Covariance estimates are unavailable until the optimization backend "
        "is implemented."
    ) in result.warnings
    assert result.target_weights == {"BBB": 1.0}


def test_run_selected_strategies_warns_for_cash_only_portfolio() -> None:
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
        histories={"SPY": candles(100.0, 101.0)},
    )
    result = PortfolioOptimizationService(
        PortfolioService(provider)
    ).run_selected_strategies(
        strategy_ids=[PortfolioOptimizationStrategy.BUY_AND_HOLD]
    )[0]

    assert result.target_weights == {"CASH": 1.0}
    assert "Portfolio is cash-only; optimization keeps 100% cash." in result.warnings
    assert result.leverage == 1.0


def test_run_selected_strategies_rejects_unknown_lookback() -> None:
    with pytest.raises(ValueError, match="Unsupported lookback"):
        PortfolioOptimizationService().run_selected_strategies(lookback="2Y")  # type: ignore[arg-type]


def test_strategy_registry_exposes_metadata_and_callable_runners() -> None:
    execution_input = StrategyExecutionInput(
        prices={},
        current_weights={"AAA": 0.7, "BBB": 0.3},
        lookback_metrics={},
        benchmark_data={},
        constraints={"long_only": True},
    )

    assert set(STRATEGY_REGISTRY) == set(PortfolioOptimizationStrategy)
    for strategy_id, entry in STRATEGY_REGISTRY.items():
        assert entry.metadata.strategy_id == strategy_id
        assert entry.metadata.strategy_name
        assert callable(entry.runner)
        runner_result = entry.runner(execution_input)
        assert runner_result.target_weights
        if strategy_id not in {
            PortfolioOptimizationStrategy.BUY_AND_HOLD,
            PortfolioOptimizationStrategy.FIXED_SP500_KMLM,
        }:
            assert runner_result.is_placeholder is True


def test_run_selected_strategies_universe_defaults_to_current_holdings_only() -> None:
    provider = FakeSchwabProvider(
        holdings={
            "hash-1": [
                {
                    "symbol": "aaa",
                    "asset_type": "EQUITY",
                    "quantity": 2.0,
                    "market_value": 20.0,
                },
                {
                    "symbol": "BBB",
                    "asset_type": "ETF",
                    "quantity": 1.0,
                    "market_value": 30.0,
                },
                {
                    "symbol": "CASH",
                    "asset_type": "CASH",
                    "quantity": 5.0,
                    "market_value": 5.0,
                },
            ]
        },
        histories={
            "AAA": candles(10.0, 11.0),
            "BBB": candles(30.0, 31.0),
            "SPY": candles(100.0, 101.0),
        },
    )

    result = PortfolioOptimizationService(
        PortfolioService(provider)
    ).run_selected_strategies(
        strategy_ids=[PortfolioOptimizationStrategy.BUY_AND_HOLD]
    )[0]

    assert result.constraints["universe"] == {
        "current_holding_symbols": ["AAA", "BBB"],
        "candidate_only_symbols": [],
        "symbols": ["AAA", "BBB"],
        "metadata": {
            "AAA": {
                "symbol": "AAA",
                "is_current_holding": True,
                "is_candidate": False,
            },
            "BBB": {
                "symbol": "BBB",
                "is_current_holding": True,
                "is_candidate": False,
            },
        },
    }


def test_run_selected_strategies_universe_includes_candidate_symbols_only() -> None:
    provider = FakeSchwabProvider(
        holdings={
            "hash-1": [
                {
                    "symbol": "AAA",
                    "asset_type": "EQUITY",
                    "quantity": 2.0,
                    "market_value": 20.0,
                }
            ]
        },
        histories={"AAA": candles(10.0, 11.0), "SPY": candles(100.0, 101.0)},
    )

    result = PortfolioOptimizationService(
        PortfolioService(provider)
    ).run_selected_strategies(
        strategy_ids=[PortfolioOptimizationStrategy.BUY_AND_HOLD],
        universe_symbols=["spy", " kmlm ", "SPY"],
    )[0]

    universe = result.constraints["universe"]
    assert universe["current_holding_symbols"] == ["AAA"]
    assert universe["candidate_only_symbols"] == ["SPY", "KMLM"]
    assert universe["symbols"] == ["AAA", "SPY", "KMLM"]
    assert universe["metadata"]["SPY"] == {
        "symbol": "SPY",
        "is_current_holding": False,
        "is_candidate": True,
    }
    assert universe["metadata"]["KMLM"] == {
        "symbol": "KMLM",
        "is_current_holding": False,
        "is_candidate": True,
    }


def test_run_selected_strategies_universe_dedupes_overlap() -> None:
    provider = FakeSchwabProvider(
        holdings={
            "hash-1": [
                {
                    "symbol": "AAA",
                    "asset_type": "EQUITY",
                    "quantity": 2.0,
                    "market_value": 20.0,
                },
                {
                    "symbol": "BBB",
                    "asset_type": "ETF",
                    "quantity": 1.0,
                    "market_value": 30.0,
                },
            ]
        },
        histories={
            "AAA": candles(10.0, 11.0),
            "BBB": candles(30.0, 31.0),
            "SPY": candles(100.0, 101.0),
        },
    )

    result = PortfolioOptimizationService(
        PortfolioService(provider)
    ).run_selected_strategies(
        strategy_ids=[PortfolioOptimizationStrategy.BUY_AND_HOLD],
        universe_symbols=["bbb", "CCC", "aaa", "ccc"],
    )[0]

    universe = result.constraints["universe"]
    assert universe["current_holding_symbols"] == ["AAA", "BBB"]
    assert universe["candidate_only_symbols"] == ["CCC"]
    assert universe["symbols"] == ["AAA", "BBB", "CCC"]
    assert universe["metadata"]["AAA"] == {
        "symbol": "AAA",
        "is_current_holding": True,
        "is_candidate": True,
    }
    assert universe["metadata"]["BBB"] == {
        "symbol": "BBB",
        "is_current_holding": True,
        "is_candidate": True,
    }


def test_run_selected_strategies_universe_handles_cash_only_candidates() -> None:
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
        histories={"SPY": candles(100.0, 101.0)},
    )

    result = PortfolioOptimizationService(
        PortfolioService(provider)
    ).run_selected_strategies(
        strategy_ids=[PortfolioOptimizationStrategy.BUY_AND_HOLD],
        universe_symbols=["spy", "cash", "KMLM"],
    )[0]

    assert result.target_weights == {"CASH": 1.0}
    assert result.constraints["universe"] == {
        "current_holding_symbols": [],
        "candidate_only_symbols": ["SPY", "KMLM"],
        "symbols": ["SPY", "KMLM"],
        "metadata": {
            "SPY": {
                "symbol": "SPY",
                "is_current_holding": False,
                "is_candidate": True,
            },
            "KMLM": {
                "symbol": "KMLM",
                "is_current_holding": False,
                "is_candidate": True,
            },
        },
    }
