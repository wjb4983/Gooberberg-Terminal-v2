"""Shared helpers for optional regime controls in Streamlit pages."""

from __future__ import annotations

from typing import Any

from quant_platform.models.schemas import RegimeDetectorType, RegimeSwitchingType


def parse_csv_columns(raw: str) -> list[str]:
    """Parse comma-separated text input into trimmed column names."""

    return [column.strip() for column in raw.split(",") if column.strip()]


def parse_regime_weights(raw: str) -> dict[str, float]:
    """Parse one regime:weight mapping per line from text input."""

    weights: dict[str, float] = {}
    for line in raw.splitlines():
        if not line.strip() or ":" not in line:
            continue
        regime, weight = line.split(":", 1)
        regime = regime.strip()
        if regime:
            weights[regime] = float(weight.strip())
    return weights


def model_regime_config(
    *,
    enabled: bool,
    detector_type: RegimeDetectorType,
    lookback: int,
    threshold: float,
    feature_columns: list[str],
    regime_weights: dict[str, float],
) -> dict[str, Any] | None:
    """Build optional model-definition regime metadata."""

    if not enabled:
        return None
    config: dict[str, Any] = {
        "detector_type": detector_type.value,
        "lookback": int(lookback),
        "threshold": float(threshold),
    }
    if feature_columns:
        config["feature_column"] = feature_columns[0]
    return config


def model_regime_switching_config(
    *,
    enabled: bool,
    switching_type: RegimeSwitchingType,
    regime_column: str,
    signal_column: str,
    target_weight_column: str,
    feature_columns: list[str],
    regime_weights: dict[str, float],
) -> dict[str, Any] | None:
    """Build optional model-definition regime-switching metadata."""

    if not enabled:
        return None
    if switching_type == RegimeSwitchingType.STATE_WEIGHTED_ALLOCATION:
        config: dict[str, Any] = {
            "switching_type": switching_type.value,
            "regime_column": regime_column or "regime",
            "signal_column": signal_column or "signal",
            "target_weight_column": target_weight_column or "target_weight",
        }
        if regime_weights:
            config["regime_weights"] = regime_weights
        if feature_columns:
            config["feature_columns"] = feature_columns
        return config
    if switching_type == RegimeSwitchingType.SWITCHING_LINEAR:
        return {
            "switching_type": switching_type.value,
            "regime_column": regime_column or "regime",
            "feature_columns": feature_columns or ["return"],
            "target_column": "target",
            "prediction_column": "prediction",
        }
    if switching_type == RegimeSwitchingType.STATE_DEPENDENT_RISK:
        config = {
            "switching_type": switching_type.value,
            "regime_column": regime_column or "regime",
            "signal_column": signal_column or "signal",
            "adjusted_signal_column": "risk_adjusted_signal",
        }
        if regime_weights:
            config["max_leverage_by_regime"] = regime_weights
        return config
    return {
        "switching_type": switching_type.value,
        "regime_column": regime_column or "regime",
        "endog_column": "return",
    }


def backtest_regime_config(
    *,
    enabled: bool,
    detector_type: RegimeDetectorType,
    lookback: int,
    threshold: float,
    feature_columns: list[str],
    regime_weights: dict[str, float],
) -> dict[str, Any]:
    """Build a backtest regime config with detection disabled by default."""

    if not enabled:
        return {"enabled": False}
    detector: dict[str, Any] = {
        "detector_type": detector_type.value,
        "lookback": int(lookback),
        "threshold": float(threshold),
    }
    if feature_columns:
        detector["feature_columns"] = feature_columns
        detector["feature_column"] = feature_columns[0]
    config: dict[str, Any] = {"enabled": True, "detector": detector}
    if regime_weights:
        config["allocation"] = {
            "switching_type": "state_weighted_allocation",
            "regime_weights": regime_weights,
        }
    return config
