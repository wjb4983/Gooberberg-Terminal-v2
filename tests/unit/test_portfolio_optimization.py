from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import polars as pl
import pytest

from quant_platform.portfolio.optimization import (
    STRATEGY_REGISTRY,
    PortfolioOptimizationStrategy,
)
from quant_platform.portfolio.optimization_service import (
    PLACEHOLDER_WARNING,
    PortfolioOptimizationService,
)
from quant_platform.portfolio.service import PortfolioService


REQUIRED_BASELINE_STRATEGIES = {
    PortfolioOptimizationStrategy.BUY_AND_HOLD,
    PortfolioOptimizationStrategy.FIXED_SP500_KMLM,
    PortfolioOptimizationStrategy.MONTHLY_REBALANCE,
    PortfolioOptimizationStrategy.VOLATILITY_TARGETING,
    PortfolioOptimizationStrategy.EQUAL_RISK_PARITY,
    PortfolioOptimizationStrategy.MOMENTUM_ROTATION,
    PortfolioOptimizationStrategy.MOMENTUM_VOLATILITY_TARGETING,
    PortfolioOptimizationStrategy.MOMENTUM_DRAWDOWN_FILTER,
    PortfolioOptimizationStrategy.MEAN_VARIANCE_SHRINKAGE,
    PortfolioOptimizationStrategy.BLACK_LITTERMAN_MOMENTUM_VIEWS,
}

PLACEHOLDER_STRATEGIES = REQUIRED_BASELINE_STRATEGIES - {
    PortfolioOptimizationStrategy.BUY_AND_HOLD,
    PortfolioOptimizationStrategy.FIXED_SP500_KMLM,
}


@dataclass(frozen=True)
class Account:
    account_hash: str


class FakeSchwabProvider:
    def __init__(
        self,
        *,
        holdings: dict[str, list[dict[str, Any]]] | None = None,
        histories: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        self.holdings = holdings or {}
        self.histories = histories or {}
        self.history_calls: list[str] = []

    def account_numbers(self) -> list[Account]:
        return [Account(account_hash=account_hash) for account_hash in self.holdings]

    def account_holdings(self, account_hash: str) -> pl.DataFrame:
        return pl.DataFrame(self.holdings.get(account_hash, []))

    def historical_prices(self, symbol: str, **_kwargs: Any) -> dict[str, Any]:
        symbol = symbol.upper()
        self.history_calls.append(symbol)
        return self.histories.get(symbol, _candles(100.0, 101.0, 102.0))


def _candles(*closes: float) -> dict[str, Any]:
    return {
        "candles": [
            {"datetime": 1_704_067_200_000 + index * 86_400_000, "close": close}
            for index, close in enumerate(closes)
        ]
    }


def _service_for_holdings(
    holdings: list[dict[str, Any]] | None = None,
) -> PortfolioOptimizationService:
    provider = FakeSchwabProvider(
        holdings={"hash-1": holdings or []},
        histories={
            "AAA": _candles(10.0, 11.0, 12.0),
            "BBB": _candles(20.0, 21.0, 22.0),
            "SPY": _candles(400.0, 401.0, 402.0),
        },
    )
    return PortfolioOptimizationService(PortfolioService(provider))


def _mixed_holdings() -> list[dict[str, Any]]:
    return [
        {
            "symbol": "AAA",
            "asset_type": "EQUITY",
            "quantity": 2.0,
            "market_value": 120.0,
        },
        {
            "symbol": "BBB",
            "asset_type": "EQUITY",
            "quantity": 4.0,
            "market_value": 80.0,
        },
        {
            "symbol": "CASH",
            "asset_type": "CASH",
            "quantity": 50.0,
            "market_value": 50.0,
        },
    ]


def test_all_required_baseline_strategies_are_present_in_registry() -> None:
    assert set(STRATEGY_REGISTRY) == REQUIRED_BASELINE_STRATEGIES
    for strategy_id in REQUIRED_BASELINE_STRATEGIES:
        entry = STRATEGY_REGISTRY[strategy_id]
        assert entry.metadata.strategy_id == strategy_id
        assert entry.metadata.strategy_name
        assert entry.metadata.description
        assert callable(entry.runner)


@pytest.mark.parametrize(
    "requested",
    [
        [PortfolioOptimizationStrategy.BUY_AND_HOLD],
        [
            PortfolioOptimizationStrategy.BUY_AND_HOLD,
            PortfolioOptimizationStrategy.FIXED_SP500_KMLM,
            PortfolioOptimizationStrategy.MEAN_VARIANCE_SHRINKAGE,
        ],
        list(PortfolioOptimizationStrategy),
    ],
)
def test_selecting_one_several_or_all_strategies_returns_only_requested_results(
    requested: list[PortfolioOptimizationStrategy],
) -> None:
    results = _service_for_holdings(_mixed_holdings()).run_selected_strategies(
        strategy_ids=requested
    )

    assert [result.strategy_id for result in results] == requested
    assert len(results) == len(requested)


def test_string_strategy_selection_is_deduplicated_and_ordered() -> None:
    results = _service_for_holdings(_mixed_holdings()).run_selected_strategies(
        strategy_ids=["fixed_sp500_kmlm", "buy_and_hold", "fixed_sp500_kmlm"]
    )

    assert [result.strategy_id for result in results] == [
        PortfolioOptimizationStrategy.FIXED_SP500_KMLM,
        PortfolioOptimizationStrategy.BUY_AND_HOLD,
    ]


@pytest.mark.parametrize("strategy_id", sorted(PLACEHOLDER_STRATEGIES))
def test_placeholder_strategies_set_placeholder_flag_and_warning(
    strategy_id: PortfolioOptimizationStrategy,
) -> None:
    result = _service_for_holdings(_mixed_holdings()).run_selected_strategies(
        strategy_ids=[strategy_id]
    )[0]

    assert result.is_placeholder is True
    assert result.warnings[0] == PLACEHOLDER_WARNING
    assert any(
        "not implemented" in warning or "unavailable" in warning
        for warning in result.warnings
    )


def test_buy_and_hold_reuses_existing_current_weights_with_holdings_and_prices() -> None:
    result = _service_for_holdings(_mixed_holdings()).run_selected_strategies(
        strategy_ids=[PortfolioOptimizationStrategy.BUY_AND_HOLD]
    )[0]

    assert result.is_placeholder is False
    assert result.target_weights == {
        "AAA": pytest.approx(120.0 / 250.0),
        "BBB": pytest.approx(80.0 / 250.0),
        "CASH": pytest.approx(50.0 / 250.0),
    }
    assert result.constraints["input_positions"] == {"AAA": 2.0, "BBB": 4.0}
    assert result.turnover == 0.0
    assert result.leverage == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("holdings", "expected_warning"),
    [
        ([], "Portfolio is cash-only; optimization keeps 100% cash."),
        (
            [
                {
                    "symbol": "CASH",
                    "asset_type": "CASH",
                    "quantity": 125.0,
                    "market_value": 125.0,
                }
            ],
            "Portfolio is cash-only; optimization keeps 100% cash.",
        ),
    ],
)
def test_empty_or_cash_only_holdings_stay_100_percent_cash(
    holdings: list[dict[str, Any]], expected_warning: str
) -> None:
    result = _service_for_holdings(holdings).run_selected_strategies(
        strategy_ids=[PortfolioOptimizationStrategy.BUY_AND_HOLD]
    )[0]

    assert result.target_weights == {"CASH": 1.0}
    assert result.constraints["input_positions"] == {}
    assert expected_warning in result.warnings
    assert result.leverage == 1.0
