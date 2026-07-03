"""Unit tests for shared Streamlit regime-control helpers."""

from __future__ import annotations

from apps.ui.regime_helpers import (
    backtest_regime_config,
    model_regime_config,
    parse_csv_columns,
    parse_regime_weights,
)

from quant_platform.models.schemas import RegimeDetectorType


def test_regime_ui_helpers_keep_defaults_disabled() -> None:
    """Default helper outputs should avoid enabling regime workflows."""

    assert model_regime_config(
        enabled=False,
        detector_type=RegimeDetectorType.VOLATILITY_THRESHOLD,
        lookback=20,
        threshold=0.0,
        feature_columns=["return"],
        regime_weights={"high_risk": 0.25},
    ) is None
    assert backtest_regime_config(
        enabled=False,
        detector_type=RegimeDetectorType.VOLATILITY_THRESHOLD,
        lookback=20,
        threshold=0.0,
        feature_columns=["return"],
        regime_weights={"high_risk": 0.25},
    ) == {"enabled": False}


def test_regime_ui_helpers_parse_enabled_controls() -> None:
    """Enabled helper outputs should include detector and allocation settings."""

    columns = parse_csv_columns("return, volume_change, ")
    weights = parse_regime_weights("normal: 1.0\nhigh_risk: 0.25")

    assert columns == ["return", "volume_change"]
    assert weights == {"normal": 1.0, "high_risk": 0.25}
    assert model_regime_config(
        enabled=True,
        detector_type=RegimeDetectorType.ROLLING_ZSCORE,
        lookback=30,
        threshold=1.5,
        feature_columns=columns,
        regime_weights=weights,
    ) == {
        "enabled": True,
        "detector_type": "rolling_zscore",
        "lookback": 30,
        "threshold": 1.5,
        "feature_columns": columns,
        "feature_column": "return",
        "regime_weights": weights,
    }
    assert backtest_regime_config(
        enabled=True,
        detector_type=RegimeDetectorType.ROLLING_ZSCORE,
        lookback=30,
        threshold=1.5,
        feature_columns=columns,
        regime_weights=weights,
    ) == {
        "enabled": True,
        "detector": {
            "detector_type": "rolling_zscore",
            "lookback": 30,
            "threshold": 1.5,
            "feature_columns": columns,
            "feature_column": "return",
        },
        "allocation": {
            "switching_type": "state_weighted_allocation",
            "regime_weights": weights,
        },
    }
