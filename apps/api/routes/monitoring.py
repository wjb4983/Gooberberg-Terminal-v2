"""Monitoring API routes."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel, Field

from quant_platform.monitoring.service import MonitoringService

router = APIRouter(prefix="/api/v1/monitoring", tags=["monitoring"])


class StatusResponse(BaseModel):
    """Implementation status for platform monitoring capabilities."""

    status: str
    label: str
    detail: str


class MonitoringSummaryResponse(BaseModel):
    """Model operations monitoring summary response."""

    generated_at: datetime
    active_models: list[dict[str, Any]] = Field(default_factory=list)
    recent_predictions: list[dict[str, Any]] = Field(default_factory=list)
    data_freshness: list[dict[str, Any]] = Field(default_factory=list)
    drift_checks: list[dict[str, Any]] = Field(default_factory=list)
    latency: dict[str, Any] = Field(default_factory=dict)
    trading_status: dict[str, StatusResponse] = Field(default_factory=dict)


@router.get("", response_model=MonitoringSummaryResponse)
def get_monitoring_summary() -> MonitoringSummaryResponse:
    """Return model, data, drift, latency, and trading monitoring status."""

    return MonitoringSummaryResponse(**MonitoringService().summary().jsonable())
