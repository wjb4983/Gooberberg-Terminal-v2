"""Portfolio optimization strategy identifiers and result schemas."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Any


class PortfolioOptimizationStrategy(StrEnum):
    """Stable identifiers for supported portfolio optimization strategies."""

    BUY_AND_HOLD = "buy_and_hold"
    FIXED_SP500_KMLM = "fixed_sp500_kmlm"
    MONTHLY_REBALANCE = "monthly_rebalance"
    VOLATILITY_TARGETING = "volatility_targeting"
    EQUAL_RISK_PARITY = "equal_risk_parity"
    MOMENTUM_ROTATION = "momentum_rotation"
    MOMENTUM_VOLATILITY_TARGETING = "momentum_volatility_targeting"
    MOMENTUM_DRAWDOWN_FILTER = "momentum_drawdown_filter"
    MEAN_VARIANCE_SHRINKAGE = "mean_variance_shrinkage"
    BLACK_LITTERMAN_MOMENTUM_VIEWS = "black_litterman_momentum_views"


@dataclass(frozen=True)
class PortfolioStrategyMetadata:
    """Display metadata for a portfolio optimization strategy."""

    strategy_id: PortfolioOptimizationStrategy
    strategy_name: str


STRATEGY_METADATA: Mapping[
    PortfolioOptimizationStrategy, PortfolioStrategyMetadata
] = MappingProxyType(
    {
        PortfolioOptimizationStrategy.BUY_AND_HOLD: PortfolioStrategyMetadata(
            strategy_id=PortfolioOptimizationStrategy.BUY_AND_HOLD,
            strategy_name="Buy-and-hold",
        ),
        PortfolioOptimizationStrategy.FIXED_SP500_KMLM: PortfolioStrategyMetadata(
            strategy_id=PortfolioOptimizationStrategy.FIXED_SP500_KMLM,
            strategy_name="Fixed S&P 500 + KMLM",
        ),
        PortfolioOptimizationStrategy.MONTHLY_REBALANCE: PortfolioStrategyMetadata(
            strategy_id=PortfolioOptimizationStrategy.MONTHLY_REBALANCE,
            strategy_name="Monthly rebalance",
        ),
        PortfolioOptimizationStrategy.VOLATILITY_TARGETING: PortfolioStrategyMetadata(
            strategy_id=PortfolioOptimizationStrategy.VOLATILITY_TARGETING,
            strategy_name="Volatility targeting",
        ),
        PortfolioOptimizationStrategy.EQUAL_RISK_PARITY: PortfolioStrategyMetadata(
            strategy_id=PortfolioOptimizationStrategy.EQUAL_RISK_PARITY,
            strategy_name="Equal-risk/risk parity",
        ),
        PortfolioOptimizationStrategy.MOMENTUM_ROTATION: PortfolioStrategyMetadata(
            strategy_id=PortfolioOptimizationStrategy.MOMENTUM_ROTATION,
            strategy_name="Momentum rotation",
        ),
        PortfolioOptimizationStrategy.MOMENTUM_VOLATILITY_TARGETING: (
            PortfolioStrategyMetadata(
                strategy_id=PortfolioOptimizationStrategy.MOMENTUM_VOLATILITY_TARGETING,
                strategy_name="Momentum + volatility targeting",
            )
        ),
        PortfolioOptimizationStrategy.MOMENTUM_DRAWDOWN_FILTER: (
            PortfolioStrategyMetadata(
                strategy_id=PortfolioOptimizationStrategy.MOMENTUM_DRAWDOWN_FILTER,
                strategy_name="Momentum + drawdown filter",
            )
        ),
        PortfolioOptimizationStrategy.MEAN_VARIANCE_SHRINKAGE: PortfolioStrategyMetadata(
            strategy_id=PortfolioOptimizationStrategy.MEAN_VARIANCE_SHRINKAGE,
            strategy_name="Mean-variance with shrinkage covariance",
        ),
        PortfolioOptimizationStrategy.BLACK_LITTERMAN_MOMENTUM_VIEWS: (
            PortfolioStrategyMetadata(
                strategy_id=PortfolioOptimizationStrategy.BLACK_LITTERMAN_MOMENTUM_VIEWS,
                strategy_name="Black-Litterman with momentum views",
            )
        ),
    }
)


@dataclass(frozen=True)
class OptimizedPortfolioResult:
    """Result payload for an optimized portfolio strategy run."""

    strategy_id: PortfolioOptimizationStrategy
    strategy_name: str
    target_weights: dict[str, float]
    expected_return: float | None
    volatility: float | None
    sharpe: float | None
    max_drawdown: float | None
    turnover: float | None
    leverage: float | None
    warnings: list[str] = field(default_factory=list)
    constraints: dict[str, Any] = field(default_factory=dict)
    is_placeholder: bool = False


__all__ = [
    "OptimizedPortfolioResult",
    "PortfolioOptimizationStrategy",
    "PortfolioStrategyMetadata",
    "STRATEGY_METADATA",
]
