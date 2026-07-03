"""Shared helpers for optional regime controls in Streamlit pages."""

from __future__ import annotations

from typing import Any

from quant_platform.models.schemas import RegimeDetectorType


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
        "enabled": True,
        "detector_type": detector_type.value,
        "lookback": int(lookback),
        "threshold": float(threshold),
    }
    if feature_columns:
        config["feature_columns"] = feature_columns
        config["feature_column"] = feature_columns[0]
    if regime_weights:
        config["regime_weights"] = regime_weights
    return config


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
