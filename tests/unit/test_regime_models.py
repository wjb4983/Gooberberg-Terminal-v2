"""Unit tests for regime model configs and deterministic detectors."""

from __future__ import annotations

import pandas as pd
import pytest
from pydantic import ValidationError

from quant_platform.models import (
    ClusteringRegimeConfig,
    HMMRegimeConfig,
    PCARegimeConfig,
    RollingZScoreRegimeConfig,
    StateWeightedAllocationConfig,
    StateWeightedAllocationModel,
    ThresholdRegimeConfig,
    ThresholdRegimeDetector,
    build_regime_detector_from_dict,
)


def _market_frame() -> pd.DataFrame:
    """Build deterministic market data with obvious stressed segments."""

    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=8, freq="D"),
            "close": [100.0, 101.0, 102.0, 100.0, 80.0, 78.0, 79.0, 78.5],
            "return": [0.001, 0.002, 0.001, 0.25, -0.25, 0.004, 0.003, 0.002],
            "volume": [1000.0, 950.0, 900.0, 850.0, 800.0, 100.0, 90.0, 80.0],
            "benchmark_return": [
                0.001,
                0.0015,
                0.001,
                0.02,
                -0.02,
                0.003,
                0.002,
                0.001,
            ],
        }
    )


@pytest.mark.parametrize(
    "config",
    [
        ThresholdRegimeConfig(),
        RollingZScoreRegimeConfig(),
        ClusteringRegimeConfig(),
        PCARegimeConfig(),
        HMMRegimeConfig(),
        StateWeightedAllocationConfig(),
    ],
)
def test_baseline_regime_configs_validate_with_defaults(config) -> None:
    """Every baseline regime config should be constructible with defaults."""

    assert config.regime_column == "regime"
    if hasattr(config, "lookback"):
        assert config.lookback == 20
    else:
        assert config.n_regimes == 2


def test_regime_configs_reject_empty_feature_columns() -> None:
    with pytest.raises(ValidationError):
        ClusteringRegimeConfig(feature_columns=())

    with pytest.raises(ValidationError, match="non-empty column names"):
        PCARegimeConfig(feature_columns=("return", " "))


@pytest.mark.parametrize(
    "config_type",
    [
        ThresholdRegimeConfig,
        RollingZScoreRegimeConfig,
        ClusteringRegimeConfig,
        PCARegimeConfig,
        HMMRegimeConfig,
    ],
)
def test_regime_configs_reject_non_positive_lookback(config_type) -> None:
    with pytest.raises(ValidationError):
        config_type(lookback=0)


@pytest.mark.parametrize("detector_type", ["not_supported", "change_point", "hmm"])
def test_regime_detector_factory_rejects_unsupported_detector_types(
    detector_type: str,
) -> None:
    with pytest.raises(ValueError):
        build_regime_detector_from_dict({"detector_type": detector_type})


def test_threshold_regime_outputs_match_input_length_for_market_frame() -> None:
    data = _market_frame()
    detector = ThresholdRegimeDetector(
        rule="volatility",
        lookback=3,
        threshold=0.20,
        min_periods=2,
    )

    regimes = detector.fit(data).predict(data)

    assert len(regimes) == len(data)
    assert regimes.index.equals(data.index)


def test_threshold_regime_labels_known_market_segments() -> None:
    data = _market_frame()

    high_volatility = ThresholdRegimeDetector(
        rule="volatility",
        lookback=3,
        threshold=0.20,
        min_periods=2,
    ).predict(data)
    drawdown = ThresholdRegimeDetector(
        rule="drawdown",
        lookback=4,
        threshold=-0.15,
        direction="below",
        min_periods=2,
    ).predict(data)
    low_liquidity = ThresholdRegimeDetector(
        rule="liquidity",
        lookback=2,
        threshold=200.0,
        direction="below",
        min_periods=2,
    ).predict(data)

    assert high_volatility.iloc[3] == 1
    assert drawdown.iloc[4] == 2
    assert low_liquidity.iloc[6] == 2
    assert set(high_volatility.unique()) >= {0, 1}
    assert set(drawdown.unique()) >= {0, 2}
    assert set(low_liquidity.unique()) >= {0, 2}


def test_state_weighted_allocation_reduces_high_risk_exposure() -> None:
    data = pd.DataFrame(
        {
            "regime": ["normal", "high_risk"],
            "signal": [1.0, 1.0],
        }
    )
    model = StateWeightedAllocationModel(
        StateWeightedAllocationConfig(
            regime_weights={"high_risk": 0.25},
            default_weight=1.0,
        )
    )

    transformed = model.transform_signals(data)

    assert transformed.loc[0, "signal"] == 1.0
    assert transformed.loc[1, "signal"] == 0.25
    assert data.loc[1, "signal"] == 1.0
