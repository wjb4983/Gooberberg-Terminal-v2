"""Tests for model definition page regime helper payloads."""

from __future__ import annotations

from apps.ui.regime_helpers import (
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
        definition.to_parameters()["regime"]["detector_type"]
        == "volatility_threshold"
    )
    assert "enabled" not in definition.to_parameters()["regime"]


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
