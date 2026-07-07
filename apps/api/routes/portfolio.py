"""Portfolio API routes backed by Schwab account summaries."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, ConfigDict, Field

from quant_platform.config import get_settings
from quant_platform.portfolio.optimization import PortfolioOptimizationStrategy
from quant_platform.portfolio.optimization_service import (
    PortfolioOptimizationService,
    results_to_dicts,
)
from quant_platform.portfolio.service import PortfolioService

router = APIRouter(prefix="/api/v1/portfolio", tags=["portfolio"])

_SAFE_HOLDING_KEYS = {
    "masked_account_label",
    "symbol",
    "asset_type",
    "quantity",
    "market_value",
    "current_price",
    "cost_basis",
    "average_price",
    "unrealized_pnl",
    "unrealized_pnl_percent",
    "is_cash",
}


class PortfolioTotalsResponse(BaseModel):
    """Aggregate portfolio dollar totals."""

    total_value: float = 0.0
    securities_value: float = 0.0
    cash_value: float = 0.0
    cost_basis: float = 0.0
    unrealized_pnl: float = 0.0


class PortfolioHoldingResponse(BaseModel):
    """Sanitized holding row safe for API clients."""

    model_config = ConfigDict(extra="ignore")

    masked_account_label: str | None = None
    symbol: str | None = None
    asset_type: str = "UNKNOWN"
    quantity: float | None = None
    market_value: float | None = None
    current_price: float | None = None
    cost_basis: float | None = None
    average_price: float | None = None
    unrealized_pnl: float | None = None
    unrealized_pnl_percent: float | None = None
    is_cash: bool = False


class PortfolioAllocationResponse(BaseModel):
    """Portfolio allocation bucket by symbol or asset type."""

    symbol: str | None = None
    asset_type: str | None = None
    market_value: float
    weight: float


class PortfolioMetricsResponse(BaseModel):
    """Portfolio risk/return metrics for one lookback window."""

    lookback: str
    weights: dict[str, float] = Field(default_factory=dict)
    total_return: float = 0.0
    annualized_volatility: float = 0.0
    sharpe_ratio: float = 0.0
    beta: float = 0.0
    max_drawdown: float = 0.0


class PortfolioWarningResponse(BaseModel):
    """Recoverable warning returned by the portfolio service."""

    code: str
    message: str
    symbol: str | None = None


class PortfolioMetadataResponse(BaseModel):
    """Non-secret portfolio refresh metadata."""

    provider: str
    benchmark_symbol: str
    refreshed_at: str
    holdings_refreshed_at: str
    prices_refreshed_at: str
    benchmark_refreshed_at: str | None = None
    holdings_cache_ttl_seconds: int | None = None
    price_history_cache_ttl_seconds: int | None = None
    stale_data: bool = False
    lookbacks: list[str] = Field(default_factory=list)


class PortfolioHoldingsResponse(BaseModel):
    """Holdings endpoint response."""

    totals: PortfolioTotalsResponse
    holdings: list[PortfolioHoldingResponse]
    warnings: list[PortfolioWarningResponse]
    metadata: PortfolioMetadataResponse


class PortfolioAllocationsResponse(BaseModel):
    """Allocations endpoint response."""

    allocation_by_symbol: list[PortfolioAllocationResponse]
    allocation_by_asset_type: list[PortfolioAllocationResponse]
    warnings: list[PortfolioWarningResponse]
    metadata: PortfolioMetadataResponse


class PortfolioMetricsEnvelopeResponse(BaseModel):
    """Metrics endpoint response."""

    totals: PortfolioTotalsResponse
    lookback_metrics: dict[str, PortfolioMetricsResponse]
    warnings: list[PortfolioWarningResponse]
    metadata: PortfolioMetadataResponse


class PortfolioSummaryResponse(
    PortfolioHoldingsResponse,
    PortfolioAllocationsResponse,
    PortfolioMetricsEnvelopeResponse,
):
    """Complete sanitized portfolio summary."""


class OptimizedPortfolioResultResponse(BaseModel):
    """Optimized portfolio result for a selected strategy."""

    strategy_id: PortfolioOptimizationStrategy
    strategy_name: str
    target_weights: dict[str, float] = Field(default_factory=dict)
    expected_return: float | None = None
    volatility: float | None = None
    sharpe: float | None = None
    max_drawdown: float | None = None
    turnover: float | None = None
    leverage: float | None = None
    warnings: list[str] = Field(default_factory=list)
    constraints: dict[str, Any] = Field(default_factory=dict)
    is_placeholder: bool = False


class PortfolioOptimizationMetadataResponse(BaseModel):
    """Non-secret metadata for an optimization run."""

    benchmark_symbol: str
    lookback: str
    selected_strategies: list[PortfolioOptimizationStrategy] = Field(
        default_factory=list
    )


class PortfolioOptimizationResponse(BaseModel):
    """Portfolio optimization endpoint response."""

    results: list[OptimizedPortfolioResultResponse]
    warnings: list[PortfolioWarningResponse]
    metadata: PortfolioOptimizationMetadataResponse


def _schwab_configured() -> None:
    settings = get_settings()
    if not settings.schwab_client_id or not settings.schwab_client_secret:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Schwab configuration is missing.",
        )


def _service() -> PortfolioService:
    _schwab_configured()
    return PortfolioService()


def _optimization_service() -> PortfolioOptimizationService:
    _schwab_configured()
    return PortfolioOptimizationService()


_PORTFOLIO_OPTIMIZATION_SERVICE_DEPENDENCY = Depends(_optimization_service)
_STRATEGY_IDS_QUERY = Query(default=None)
_ACCOUNT_HASHES_QUERY = Query(default=None)


_PORTFOLIO_SERVICE_DEPENDENCY = Depends(_service)


def _summary(service: PortfolioService, benchmark_symbol: str) -> dict[str, Any]:
    return _sanitize_summary(
        service.summary(benchmark_symbol=benchmark_symbol.strip().upper() or "SPY")
    )


def _sanitize_summary(summary: dict[str, Any]) -> dict[str, Any]:
    sanitized = dict(summary)
    sanitized["holdings"] = [
        {key: row.get(key) for key in _SAFE_HOLDING_KEYS if key in row}
        for row in summary.get("holdings", [])
        if isinstance(row, dict)
    ]
    sanitized["warnings"] = [_sanitize_warning(w) for w in summary.get("warnings", [])]
    metadata = dict(summary.get("metadata") or {})
    metadata.pop("account_hashes", None)
    sanitized["metadata"] = metadata
    return sanitized


def _sanitize_optimization_result(result: dict[str, Any]) -> dict[str, Any]:
    sanitized = dict(result)
    sanitized["warnings"] = [str(warning) for warning in result.get("warnings", [])]
    constraints = dict(result.get("constraints") or {})
    constraints.pop("account_hashes", None)
    sanitized["constraints"] = constraints
    return sanitized


def _sanitize_warning(warning: Any) -> dict[str, Any]:
    if not isinstance(warning, dict):
        return {"code": "portfolio_warning", "message": str(warning), "symbol": None}
    return {
        "code": str(warning.get("code") or "portfolio_warning"),
        "message": str(warning.get("message") or "Portfolio warning."),
        "symbol": warning.get("symbol"),
    }


@router.get("", response_model=PortfolioSummaryResponse)
def get_portfolio_summary(
    benchmark_symbol: str = Query(default="SPY"),
    service: PortfolioService = _PORTFOLIO_SERVICE_DEPENDENCY,
) -> PortfolioSummaryResponse:
    """Return a complete sanitized portfolio summary."""

    return PortfolioSummaryResponse(**_summary(service, benchmark_symbol))


@router.get("/holdings", response_model=PortfolioHoldingsResponse)
def get_portfolio_holdings(
    benchmark_symbol: str = Query(default="SPY"),
    service: PortfolioService = _PORTFOLIO_SERVICE_DEPENDENCY,
) -> PortfolioHoldingsResponse:
    """Return sanitized portfolio holdings and totals."""

    summary = _summary(service, benchmark_symbol)
    return PortfolioHoldingsResponse(
        totals=summary["totals"],
        holdings=summary["holdings"],
        warnings=summary["warnings"],
        metadata=summary["metadata"],
    )


@router.get("/allocations", response_model=PortfolioAllocationsResponse)
def get_portfolio_allocations(
    benchmark_symbol: str = Query(default="SPY"),
    service: PortfolioService = _PORTFOLIO_SERVICE_DEPENDENCY,
) -> PortfolioAllocationsResponse:
    """Return sanitized allocation buckets."""

    summary = _summary(service, benchmark_symbol)
    return PortfolioAllocationsResponse(
        allocation_by_symbol=summary["allocation_by_symbol"],
        allocation_by_asset_type=summary["allocation_by_asset_type"],
        warnings=summary["warnings"],
        metadata=summary["metadata"],
    )


@router.get("/metrics", response_model=PortfolioMetricsEnvelopeResponse)
def get_portfolio_metrics(
    benchmark_symbol: str = Query(default="SPY"),
    service: PortfolioService = _PORTFOLIO_SERVICE_DEPENDENCY,
) -> PortfolioMetricsEnvelopeResponse:
    """Return sanitized lookback metrics."""

    summary = _summary(service, benchmark_symbol)
    return PortfolioMetricsEnvelopeResponse(
        totals=summary["totals"],
        lookback_metrics=summary["lookback_metrics"],
        warnings=summary["warnings"],
        metadata=summary["metadata"],
    )


@router.get("/optimization", response_model=PortfolioOptimizationResponse)
def get_portfolio_optimization(
    strategy_ids: list[PortfolioOptimizationStrategy] | None = _STRATEGY_IDS_QUERY,
    benchmark_symbol: str = Query(default="SPY"),
    lookback: str = Query(default="MAX"),
    account_hashes: list[str] | None = _ACCOUNT_HASHES_QUERY,
    service: PortfolioOptimizationService = _PORTFOLIO_OPTIMIZATION_SERVICE_DEPENDENCY,
) -> PortfolioOptimizationResponse:
    """Run selected portfolio optimization strategies."""

    normalized_benchmark = benchmark_symbol.strip().upper() or "SPY"
    try:
        results = service.run_selected_strategies(
            account_hashes=account_hashes,
            benchmark_symbol=normalized_benchmark,
            lookback=lookback,  # type: ignore[arg-type]
            strategy_ids=strategy_ids,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc

    result_dicts = [
        _sanitize_optimization_result(row) for row in results_to_dicts(results)
    ]
    warnings = [
        _sanitize_warning(
            {"code": "portfolio_optimization_warning", "message": warning}
        )
        for result in result_dicts
        for warning in result.get("warnings", [])
    ]
    selected_strategies = [
        PortfolioOptimizationStrategy(row["strategy_id"]) for row in result_dicts
    ]
    return PortfolioOptimizationResponse(
        results=result_dicts,
        warnings=warnings,
        metadata={
            "benchmark_symbol": normalized_benchmark,
            "lookback": lookback,
            "selected_strategies": selected_strategies,
        },
    )
