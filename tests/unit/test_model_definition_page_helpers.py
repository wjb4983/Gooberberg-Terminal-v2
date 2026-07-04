"""Tests for model definition page regime helper payloads."""

from __future__ import annotations

from apps.ui.regime_helpers import (
    fields_for_regime_detector_type,
    fields_for_switching_type,
    model_regime_config,
    model_regime_switching_config,
    parse_csv_columns,
    parse_regime_weights,
)

from quant_platform.models import ModelDefinition, ModelType
from quant_platform.models.schemas import RegimeDetectorType, RegimeSwitchingType


def test_model_regime_config_is_usable_by_regime_detector_definition() -> None:
    config = model_regime_config(
        enabled=True,
        detector_type=RegimeDetectorType.VOLATILITY_THRESHOLD,
        lookback=20,
        threshold=0.2,
        feature_columns=parse_csv_columns("return, volume"),
        regime_weights=parse_regime_weights("high_risk: 0.25"),
    )

    definition = ModelDefinition(
        name="visible_regime_detector",
        model_type=ModelType.REGIME_DETECTOR,
        metadata={"regime": config},
    )

    assert (
        definition.to_parameters()["regime"]["detector_type"] == "volatility_threshold"
    )
    assert "enabled" not in definition.to_parameters()["regime"]
    assert "feature_columns" not in definition.to_parameters()["regime"]


def test_statistical_regime_config_includes_n_regimes_window_and_features() -> None:
    config = model_regime_config(
        enabled=True,
        detector_type=RegimeDetectorType.PCA,
        lookback=20,
        threshold=0.0,
        feature_columns=parse_csv_columns("return, volatility"),
        regime_weights={},
        n_regimes=3,
        window_size=40,
        n_components=2,
    )

    assert config == {
        "detector_type": "pca",
        "feature_columns": ["return", "volatility"],
        "n_components": 2,
        "n_regimes": 3,
        "score_method": "explained_variance",
        "window_size": 40,
    }
    assert {"n_regimes", "window_size", "feature_columns"}.issubset(
        fields_for_regime_detector_type(RegimeDetectorType.PCA)
    )


def test_model_regime_switching_config_is_visible_and_usable_definition() -> None:
    config = model_regime_switching_config(
        enabled=True,
        switching_type=RegimeSwitchingType.STATE_WEIGHTED_ALLOCATION,
        regime_column="regime",
        signal_column="signal",
        target_weight_column="target_weight",
        feature_columns=parse_csv_columns("return"),
        regime_weights=parse_regime_weights("high_risk: 0.25"),
    )

    definition = ModelDefinition(
        name="visible_regime_switching",
        model_type=ModelType.REGIME_SWITCHING,
        metadata={"regime_switching": config},
    )

    parameters = definition.to_parameters()
    assert parameters["regime_switching"]["switching_type"] == (
        "state_weighted_allocation"
    )
    assert parameters["regime_switching"]["regime_weights"] == {"high_risk": 0.25}
    assert "regime_weights" in fields_for_switching_type(
        RegimeSwitchingType.STATE_WEIGHTED_ALLOCATION
    )


def test_switching_linear_and_risk_configs_use_distinct_visible_fields() -> None:
    linear_config = model_regime_switching_config(
        enabled=True,
        switching_type=RegimeSwitchingType.SWITCHING_LINEAR,
        regime_column="regime",
        signal_column="signal",
        target_weight_column="target_weight",
        feature_columns=parse_csv_columns("return, volatility"),
        regime_weights={},
        target_column="future_return",
        prediction_column="regime_prediction",
    )
    risk_config = model_regime_switching_config(
        enabled=True,
        switching_type=RegimeSwitchingType.STATE_DEPENDENT_RISK,
        regime_column="regime",
        signal_column="alpha_signal",
        target_weight_column="target_weight",
        feature_columns=[],
        regime_weights=parse_regime_weights("high_risk: 0.5"),
        default_max_leverage=1.5,
    )

    assert linear_config == {
        "switching_type": "switching_linear",
        "regime_column": "regime",
        "feature_columns": ["return", "volatility"],
        "target_column": "future_return",
        "prediction_column": "regime_prediction",
        "n_regimes": 2,
    }
    assert risk_config["signal_column"] == "alpha_signal"
    assert risk_config["max_leverage_by_regime"] == {"high_risk": 0.5}
    assert "target_column" not in risk_config
