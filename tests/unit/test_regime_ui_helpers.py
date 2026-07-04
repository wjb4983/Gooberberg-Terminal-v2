"""Unit tests for shared Streamlit regime-control helpers."""

from __future__ import annotations

from apps.ui.regime_helpers import (
    backtest_regime_config,
    default_regime_feature_columns,
    fields_for_regime_detector_type,
    fields_for_switching_type,
    model_regime_config,
    parse_csv_columns,
    parse_regime_weights,
)

from quant_platform.models.schemas import RegimeDetectorType, RegimeSwitchingType


def test_regime_ui_helpers_keep_defaults_disabled() -> None:
    """Default helper outputs should avoid enabling regime workflows."""

    assert (
        model_regime_config(
            enabled=False,
            detector_type=RegimeDetectorType.VOLATILITY_THRESHOLD,
            lookback=20,
            threshold=0.0,
            feature_columns=["return"],
            regime_weights={"high_risk": 0.25},
        )
        is None
    )
    assert backtest_regime_config(
        enabled=False,
        detector_type=RegimeDetectorType.VOLATILITY_THRESHOLD,
        lookback=20,
        threshold=0.0,
        feature_columns=["return"],
        regime_weights={"high_risk": 0.25},
    ) == {"enabled": False}


def test_regime_ui_helpers_parse_enabled_controls() -> None:
    """Enabled helper outputs should include only relevant detector settings."""

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
        "detector_type": "rolling_zscore",
        "lookback": 30,
        "feature_column": "return",
        "entry_zscore": 2.0,
        "exit_zscore": 0.5,
        "n_regimes": 2,
    }
    assert backtest_regime_config(
        enabled=True,
        detector_type=RegimeDetectorType.VOLATILITY_THRESHOLD,
        lookback=30,
        threshold=1.5,
        feature_columns=columns,
        regime_weights=weights,
    ) == {
        "enabled": True,
        "detector": {
            "detector_type": "volatility_threshold",
            "lookback": 30,
            "threshold": 1.5,
            "direction": "above",
            "feature_column": "return",
        },
        "allocation": {
            "switching_type": "state_weighted_allocation",
            "regime_weights": weights,
        },
    }


def test_fields_for_regime_detector_type_groups_ui_controls() -> None:
    """Detector helpers should separate threshold and statistical detector fields."""

    assert fields_for_regime_detector_type(RegimeDetectorType.VOLATILITY_THRESHOLD) == (
        "lookback",
        "feature_column",
        "threshold",
        "direction",
    )
    assert fields_for_regime_detector_type(RegimeDetectorType.CHANGE_POINT) == (
        "n_regimes",
        "window_size",
        "feature_columns",
    )
    assert "covariance_type" in fields_for_regime_detector_type(RegimeDetectorType.HMM)
    assert "n_components" in fields_for_regime_detector_type(RegimeDetectorType.PCA)


def test_fields_for_switching_type_groups_ui_controls() -> None:
    """Switching helpers should expose strategy-specific control groups."""

    assert "regime_weights" in fields_for_switching_type(
        RegimeSwitchingType.STATE_WEIGHTED_ALLOCATION
    )
    assert {"feature_columns", "target_column", "prediction_column"}.issubset(
        fields_for_switching_type(RegimeSwitchingType.SWITCHING_LINEAR)
    )
    assert {"signal_column", "default_max_leverage"}.issubset(
        fields_for_switching_type(RegimeSwitchingType.STATE_DEPENDENT_RISK)
    )


def test_default_regime_feature_columns_are_asset_and_intent_aware() -> None:
    """Default feature helpers should recommend ordered regime workflow starters."""

    assert default_regime_feature_columns("equity", "regime_detection") == [
        "return",
        "volatility",
        "drawdown",
        "volume",
        "market_return",
        "trend",
        "liquidity",
    ]
    assert "funding_rate" in default_regime_feature_columns("crypto", None)
