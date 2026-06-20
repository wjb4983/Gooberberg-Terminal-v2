"""Backtest API routes."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field, model_validator
from sqlalchemy import select

from quant_platform.backtesting.schemas import BacktestConfig
from quant_platform.config import get_settings
from quant_platform.data.storage.catalog import (
    MetadataCatalog,
    backtests,
    dataset_definitions,
    model_definitions,
)

router = APIRouter(prefix="/api/v1/backtests", tags=["backtests"])


class BacktestQueueRequest(BaseModel):
    """Request payload for queueing a backtest job."""

    name: str = "backtest"
    source_type: Literal["model", "signal_strategy"] = "model"
    model_id: int | None = None
    signal_strategy: str | None = None
    dataset_id: int
    start_ts: datetime | None = None
    end_ts: datetime | None = None
    config: BacktestConfig = Field(default_factory=BacktestConfig)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_source(self) -> BacktestQueueRequest:
        if self.source_type == "model" and self.model_id is None:
            raise ValueError("model_id is required when source_type is model")
        if self.source_type == "signal_strategy" and not self.signal_strategy:
            raise ValueError(
                "signal_strategy is required when source_type is signal_strategy"
            )
        if self.start_ts and self.end_ts and self.start_ts > self.end_ts:
            raise ValueError("start_ts must be before end_ts")
        return self


class BacktestResponse(BaseModel):
    """Backtest row returned by the API."""

    id: int
    experiment_id: int | None = None
    name: str
    status: str
    start_ts: datetime | None = None
    end_ts: datetime | None = None
    results: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime | None = None


class BacktestListResponse(BaseModel):
    """Collection response for backtests."""

    backtests: list[BacktestResponse]


class BacktestQueueResponse(BaseModel):
    """Response returned after a backtest job is queued."""

    backtest: BacktestResponse
    job_id: int
    status: str
    job_type: str
    payload: dict[str, Any]


def _catalog() -> MetadataCatalog:
    return MetadataCatalog(get_settings().catalog_db_path)


def _backtest_response(row: Any) -> BacktestResponse:
    mapping = dict(row)
    return BacktestResponse(
        id=mapping["id"],
        experiment_id=mapping.get("experiment_id"),
        name=mapping["name"],
        status=mapping["status"],
        start_ts=mapping.get("start_ts"),
        end_ts=mapping.get("end_ts"),
        results=mapping.get("results") or {},
        metadata=mapping.get("metadata") or {},
        created_at=mapping.get("created_at"),
    )


def _require_row(
    catalog: MetadataCatalog, table: Any, row_id: int, label: str
) -> dict[str, Any]:
    with catalog.engine.connect() as connection:
        row = (
            connection.execute(select(table).where(table.c.id == row_id))
            .mappings()
            .first()
        )
    if row is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"{label} not found: {row_id}"
        )
    return dict(row)


@router.get("", response_model=BacktestListResponse)
def list_backtests() -> BacktestListResponse:
    """List queued and completed backtests."""

    catalog = _catalog()
    catalog.create_all()
    return BacktestListResponse(
        backtests=[_backtest_response(row) for row in catalog.list_rows("backtests")]
    )


@router.post(
    "", response_model=BacktestQueueResponse, status_code=status.HTTP_201_CREATED
)
def queue_backtest(request: BacktestQueueRequest) -> BacktestQueueResponse:
    """Create a backtest record and queue a worker job placeholder."""

    catalog = _catalog()
    catalog.create_all()
    dataset = _require_row(catalog, dataset_definitions, request.dataset_id, "dataset")
    model = None
    if request.model_id is not None:
        model = _require_row(
            catalog, model_definitions, request.model_id, "model definition"
        )

    metadata = {
        "source_type": request.source_type,
        "model_id": request.model_id,
        "model_name": (model or {}).get("name"),
        "signal_strategy": request.signal_strategy,
        "dataset_id": request.dataset_id,
        "dataset_name": dataset["name"],
        "config": request.config.jsonable(),
        **request.metadata,
    }
    backtest_id = catalog.insert_row(
        "backtests",
        {
            "experiment_id": None,
            "name": request.name.strip() or "backtest",
            "status": "queued",
            "start_ts": request.start_ts,
            "end_ts": request.end_ts,
            "results": {},
            "metadata": metadata,
        },
    )
    payload = {
        "backtest_id": backtest_id,
        "name": request.name.strip() or "backtest",
        "source_type": request.source_type,
        "model_id": request.model_id,
        "signal_strategy": request.signal_strategy,
        "dataset_id": request.dataset_id,
        "dataset_name": dataset["name"],
        "start_ts": request.start_ts.isoformat() if request.start_ts else None,
        "end_ts": request.end_ts.isoformat() if request.end_ts else None,
        "config": request.config.jsonable(),
    }
    job_id = catalog.insert_row(
        "jobs", {"job_type": "backtest", "status": "queued", "payload": payload}
    )
    row = _require_row(catalog, backtests, backtest_id, "backtest")
    return BacktestQueueResponse(
        backtest=_backtest_response(row),
        job_id=job_id,
        status="queued",
        job_type="backtest",
        payload=payload,
    )


@router.get("/{backtest_id}", response_model=BacktestResponse)
def get_backtest(backtest_id: int) -> BacktestResponse:
    """Return a backtest by id."""

    catalog = _catalog()
    catalog.create_all()
    return _backtest_response(_require_row(catalog, backtests, backtest_id, "backtest"))
