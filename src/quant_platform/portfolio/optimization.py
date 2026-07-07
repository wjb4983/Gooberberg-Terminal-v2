"""Portfolio optimization strategy identifiers, runners, and result schemas."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Protocol


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
    description: str = ""


@dataclass(frozen=True)
class StrategyExecutionInput:
    """Normalized inputs provided to portfolio strategy runners.

    The fields intentionally mirror the data real optimizers will need later so
    placeholder runners can be replaced without changing the service interface.
    """

    prices: Mapping[str, Any]
    current_weights: Mapping[str, float]
    lookback_metrics: Mapping[str, Any]
    benchmark_data: Mapping[str, Any]
    constraints: Mapping[str, Any]
    universe: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StrategyRunResult:
    """Raw output from a strategy runner before service-level metrics are attached."""

    target_weights: dict[str, float]
    warnings: list[str] = field(default_factory=list)
    is_placeholder: bool = True
    constraints: dict[str, Any] = field(default_factory=dict)


class StrategyRunner(Protocol):
    """Callable interface implemented by portfolio strategy runners."""

    def __call__(self, strategy_input: StrategyExecutionInput) -> StrategyRunResult:
        """Run a strategy against normalized portfolio inputs."""


@dataclass(frozen=True)
class PortfolioStrategyRegistryEntry:
    """Registry entry pairing strategy display metadata with its runner."""

    metadata: PortfolioStrategyMetadata
    runner: StrategyRunner


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


def _sorted_current_weights(strategy_input: StrategyExecutionInput) -> dict[str, float]:
    return dict(sorted(strategy_input.current_weights.items()))


def _placeholder_current_weights(
    strategy_input: StrategyExecutionInput, warning: str
) -> StrategyRunResult:
    return StrategyRunResult(
        target_weights=_sorted_current_weights(strategy_input),
        warnings=[warning],
        is_placeholder=True,
    )


def run_buy_and_hold(strategy_input: StrategyExecutionInput) -> StrategyRunResult:
    """Keep real current portfolio weights when they are available."""

    return StrategyRunResult(
        target_weights=_sorted_current_weights(strategy_input),
        is_placeholder=False,
    )


def run_fixed_sp500_kmlm(strategy_input: StrategyExecutionInput) -> StrategyRunResult:
    """Use a deterministic 60/40 S&P 500 managed-futures baseline."""

    return StrategyRunResult(
        target_weights={"KMLM": 0.4, "SPY": 0.6},
        is_placeholder=False,
        constraints={"fixed_weights": {"SPY": 0.6, "KMLM": 0.4}},
    )


def run_monthly_rebalance(strategy_input: StrategyExecutionInput) -> StrategyRunResult:
    return _placeholder_current_weights(
        strategy_input,
        "Monthly rebalance schedule logic is not implemented yet.",
    )


def run_volatility_targeting(
    strategy_input: StrategyExecutionInput,
) -> StrategyRunResult:
    return _placeholder_current_weights(
        strategy_input,
        "Volatility targeting backend is not implemented yet.",
    )


def run_equal_risk_parity(strategy_input: StrategyExecutionInput) -> StrategyRunResult:
    return _placeholder_current_weights(
        strategy_input,
        "Risk parity backend is not implemented yet.",
    )


def run_momentum_rotation(strategy_input: StrategyExecutionInput) -> StrategyRunResult:
    return _placeholder_current_weights(
        strategy_input,
        "Momentum rotation backend is not implemented yet.",
    )


def run_momentum_volatility_targeting(
    strategy_input: StrategyExecutionInput,
) -> StrategyRunResult:
    return _placeholder_current_weights(
        strategy_input,
        "Momentum plus volatility targeting backend is not implemented yet.",
    )


def run_momentum_drawdown_filter(
    strategy_input: StrategyExecutionInput,
) -> StrategyRunResult:
    return _placeholder_current_weights(
        strategy_input,
        "Momentum drawdown filter backend is not implemented yet.",
    )


def run_mean_variance_shrinkage(
    strategy_input: StrategyExecutionInput,
) -> StrategyRunResult:
    return _placeholder_current_weights(
        strategy_input,
        "Covariance estimates are unavailable until the optimization "
        "backend is implemented.",
    )


def run_black_litterman_momentum_views(
    strategy_input: StrategyExecutionInput,
) -> StrategyRunResult:
    return _placeholder_current_weights(
        strategy_input,
        "Black-Litterman momentum views backend is not implemented yet.",
    )


_STRATEGY_ENTRIES: dict[
    PortfolioOptimizationStrategy,
    tuple[str, str, Callable[[StrategyExecutionInput], StrategyRunResult]],
] = {
    PortfolioOptimizationStrategy.BUY_AND_HOLD: (
        "Buy-and-hold",
        "Keep current real portfolio weights.",
        run_buy_and_hold,
    ),
    PortfolioOptimizationStrategy.FIXED_SP500_KMLM: (
        "Fixed S&P 500 + KMLM",
        "Static 60% SPY / 40% KMLM benchmark allocation.",
        run_fixed_sp500_kmlm,
    ),
    PortfolioOptimizationStrategy.MONTHLY_REBALANCE: (
        "Monthly rebalance",
        "Rebalance the current allocation on a monthly cadence.",
        run_monthly_rebalance,
    ),
    PortfolioOptimizationStrategy.VOLATILITY_TARGETING: (
        "Volatility targeting",
        "Scale exposure to a target realized-volatility budget.",
        run_volatility_targeting,
    ),
    PortfolioOptimizationStrategy.EQUAL_RISK_PARITY: (
        "Equal-risk/risk parity",
        "Allocate so each asset contributes comparable portfolio risk.",
        run_equal_risk_parity,
    ),
    PortfolioOptimizationStrategy.MOMENTUM_ROTATION: (
        "Momentum rotation",
        "Rotate into assets with strongest lookback momentum.",
        run_momentum_rotation,
    ),
    PortfolioOptimizationStrategy.MOMENTUM_VOLATILITY_TARGETING: (
        "Momentum + volatility targeting",
        "Combine momentum selection with volatility-scaled exposure.",
        run_momentum_volatility_targeting,
    ),
    PortfolioOptimizationStrategy.MOMENTUM_DRAWDOWN_FILTER: (
        "Momentum + drawdown filter",
        "Apply a drawdown risk filter to momentum allocations.",
        run_momentum_drawdown_filter,
    ),
    PortfolioOptimizationStrategy.MEAN_VARIANCE_SHRINKAGE: (
        "Mean-variance with shrinkage covariance",
        "Optimize mean-variance weights using shrinkage covariance estimates.",
        run_mean_variance_shrinkage,
    ),
    PortfolioOptimizationStrategy.BLACK_LITTERMAN_MOMENTUM_VIEWS: (
        "Black-Litterman with momentum views",
        "Blend benchmark priors with momentum-derived active views.",
        run_black_litterman_momentum_views,
    ),
}

STRATEGY_REGISTRY: Mapping[
    PortfolioOptimizationStrategy, PortfolioStrategyRegistryEntry
] = MappingProxyType(
    {
        strategy_id: PortfolioStrategyRegistryEntry(
            metadata=PortfolioStrategyMetadata(
                strategy_id=strategy_id,
                strategy_name=name,
                description=description,
            ),
            runner=runner,
        )
        for strategy_id, (name, description, runner) in _STRATEGY_ENTRIES.items()
    }
)

STRATEGY_METADATA: Mapping[PortfolioOptimizationStrategy, PortfolioStrategyMetadata] = (
    MappingProxyType(
        {
            strategy_id: entry.metadata
            for strategy_id, entry in STRATEGY_REGISTRY.items()
        }
    )
)


__all__ = [
    "OptimizedPortfolioResult",
    "PortfolioOptimizationStrategy",
    "PortfolioStrategyMetadata",
    "PortfolioStrategyRegistryEntry",
    "STRATEGY_METADATA",
    "STRATEGY_REGISTRY",
    "StrategyExecutionInput",
    "StrategyRunResult",
    "StrategyRunner",
    "run_black_litterman_momentum_views",
    "run_buy_and_hold",
    "run_equal_risk_parity",
    "run_fixed_sp500_kmlm",
    "run_mean_variance_shrinkage",
    "run_momentum_drawdown_filter",
    "run_momentum_rotation",
    "run_momentum_volatility_targeting",
    "run_monthly_rebalance",
    "run_volatility_targeting",
]
