"""Portfolio API routes backed by Schwab account summaries."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, ConfigDict, Field

from quant_platform.config import get_settings
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
    PortfolioHoldingsResponse, PortfolioAllocationsResponse, PortfolioMetricsEnvelopeResponse
):
    """Complete sanitized portfolio summary."""


def _service() -> PortfolioService:
    settings = get_settings()
    if not settings.schwab_client_id or not settings.schwab_client_secret:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Schwab configuration is missing.",
        )
    return PortfolioService()


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
    service: PortfolioService = Depends(_service),
) -> PortfolioSummaryResponse:
    """Return a complete sanitized portfolio summary."""

    return PortfolioSummaryResponse(**_summary(service, benchmark_symbol))


@router.get("/holdings", response_model=PortfolioHoldingsResponse)
def get_portfolio_holdings(
    benchmark_symbol: str = Query(default="SPY"),
    service: PortfolioService = Depends(_service),
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
    service: PortfolioService = Depends(_service),
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
    service: PortfolioService = Depends(_service),
) -> PortfolioMetricsEnvelopeResponse:
    """Return sanitized lookback metrics."""

    summary = _summary(service, benchmark_symbol)
    return PortfolioMetricsEnvelopeResponse(
        totals=summary["totals"],
        lookback_metrics=summary["lookback_metrics"],
        warnings=summary["warnings"],
        metadata=summary["metadata"],
    )
