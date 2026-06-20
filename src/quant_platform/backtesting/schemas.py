"""Schemas for backtesting configuration and results."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class PositionSizingMode(StrEnum):
    """Supported placeholder position-sizing modes."""

    FIXED_NOTIONAL = "fixed_notional"
    EQUAL_WEIGHT = "equal_weight"
    TARGET_WEIGHT = "target_weight"


class OptionsExpirationHandling(StrEnum):
    """Placeholder policies for option expiration handling."""

    IGNORE = "ignore"
    CLOSE_BEFORE_EXPIRATION = "close_before_expiration"
    EXERCISE_OR_ASSIGN = "exercise_or_assign"


class SlippageConfig(BaseModel):
    """Placeholder slippage model configuration."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    basis_points: float = Field(default=0.0, ge=0.0)


class CommissionConfig(BaseModel):
    """Placeholder commission model configuration."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    per_trade: float = Field(default=0.0, ge=0.0)
    basis_points: float = Field(default=0.0, ge=0.0)


class BidAskConfig(BaseModel):
    """Controls whether fills prefer bid/ask columns when available."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    bid_column: str = "bid"
    ask_column: str = "ask"


class PositionSizingConfig(BaseModel):
    """Position sizing placeholder used by the initial backtest engine."""

    model_config = ConfigDict(extra="forbid")

    mode: PositionSizingMode = PositionSizingMode.FIXED_NOTIONAL
    fixed_notional: float = Field(default=10_000.0, gt=0.0)
    target_weight_column: str = "target_weight"


class ExposureConfig(BaseModel):
    """Maximum gross exposure placeholder."""

    model_config = ConfigDict(extra="forbid")

    max_gross_exposure: float = Field(default=1.0, gt=0.0)


class HoldingPeriodConfig(BaseModel):
    """Maximum holding period placeholder."""

    model_config = ConfigDict(extra="forbid")

    max_bars: int | None = Field(default=None, gt=0)


class BacktestConfig(BaseModel):
    """Configuration for the first functional event-driven backtest pass."""

    model_config = ConfigDict(extra="forbid", use_enum_values=False)

    name: str = "backtest"
    initial_cash: float = Field(default=100_000.0, gt=0.0)
    price_column: str = "close"
    signal_column: str = "signal"
    timestamp_column: str = "timestamp"
    symbol_column: str = "symbol"
    annualization_factor: int = Field(default=252, gt=0)
    artifact_dir: Path = Path("data/artifacts/backtests")
    slippage: SlippageConfig = Field(default_factory=SlippageConfig)
    commission: CommissionConfig = Field(default_factory=CommissionConfig)
    bid_ask: BidAskConfig = Field(default_factory=BidAskConfig)
    position_sizing: PositionSizingConfig = Field(default_factory=PositionSizingConfig)
    exposure: ExposureConfig = Field(default_factory=ExposureConfig)
    holding_period: HoldingPeriodConfig = Field(default_factory=HoldingPeriodConfig)
    options_expiration_handling: OptionsExpirationHandling = (
        OptionsExpirationHandling.IGNORE
    )

    @model_validator(mode="after")
    def _validate_placeholder_combinations(self) -> BacktestConfig:
        if self.exposure.max_gross_exposure <= 0:
            raise ValueError("max_gross_exposure must be positive")
        return self

    def jsonable(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of the config."""

        return self.model_dump(mode="json")


class BacktestResult(BaseModel):
    """In-memory result returned by the backtest engine."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    equity_curve: Any
    trades: Any
    metrics: dict[str, float | int]
    config: BacktestConfig
