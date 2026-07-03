"""Instantiation tests for reusable neural network models."""

from __future__ import annotations

import pandas as pd
import pytest
import torch
from pydantic import ValidationError

from quant_platform.models import (
    ClusteringRegimeConfig,
    MLPConfig,
    ModelDefinition,
    ModelRegistry,
    ModelType,
    PCARegimeConfig,
    RecurrentConfig,
    RegimeDetectorType,
    RollingZScoreRegimeConfig,
    StateWeightedAllocationConfig,
    TemporalCNNConfig,
    ThresholdRegimeConfig,
    TransformerConfig,
    build_model,
    build_regime_detector,
    build_regime_detector_from_dict,
)


def test_instantiates_mlp() -> None:
    model = build_model(MLPConfig(input_dim=4, output_dim=2, hidden_dims=(8,)))

    output = model(torch.randn(3, 4))

    assert output.shape == (3, 2)


def test_instantiates_lstm() -> None:
    model = build_model(
        RecurrentConfig(
            model_type=ModelType.LSTM,
            input_dim=4,
            output_dim=2,
            hidden_dim=5,
        )
    )

    output = model(torch.randn(3, 6, 4))

    assert output.shape == (3, 2)


def test_instantiates_gru() -> None:
    model = build_model(
        RecurrentConfig(
            model_type=ModelType.GRU,
            input_dim=4,
            output_dim=2,
            hidden_dim=5,
        )
    )

    output = model(torch.randn(3, 6, 4))

    assert output.shape == (3, 2)


def test_instantiates_temporal_cnn() -> None:
    model = build_model(
        TemporalCNNConfig(input_dim=4, output_dim=2, channels=(5,), kernel_size=2)
    )

    output = model(torch.randn(3, 6, 4))

    assert output.shape == (3, 2)


def test_instantiates_transformer_encoder_placeholder() -> None:
    model = build_model(
        TransformerConfig(
            input_dim=4,
            output_dim=2,
            d_model=8,
            nhead=2,
            num_layers=1,
            dim_feedforward=16,
        )
    )

    output = model(torch.randn(3, 6, 4))

    assert output.shape == (3, 2)


def test_model_definition_builds_runtime_config() -> None:
    definition = ModelDefinition(
        name="baseline_transformer",
        model_type=ModelType.TRANSFORMER,
        layer_count=2,
        hidden_size=64,
        dropout=0.1,
        sequence_length=32,
        input_size=8,
        output_size=1,
    )

    config = definition.to_model_config_dict()

    assert config["model_type"] == "transformer"
    assert config["d_model"] == 64
    assert config["num_layers"] == 2


def test_model_registry_registers_definition(tmp_path) -> None:
    registry = ModelRegistry(tmp_path / "metadata.sqlite")
    definition = ModelDefinition(
        name="baseline_mlp",
        model_type=ModelType.MLP,
        layer_count=2,
        hidden_size=32,
        sequence_length=16,
        input_size=4,
        output_size=1,
    )

    model_id = registry.register(definition)
    saved = registry.get("baseline_mlp")

    assert model_id > 0
    assert saved is not None
    assert saved.name == "baseline_mlp"
    assert saved.hidden_size == 32


def test_regime_config_defaults_are_conservative() -> None:
    threshold = ThresholdRegimeConfig()
    clustering = ClusteringRegimeConfig()

    assert threshold.detector_type == RegimeDetectorType.VOLATILITY_THRESHOLD
    assert threshold.lookback == 20
    assert threshold.regime_column == "regime"
    assert threshold.n_regimes == 2
    assert clustering.feature_columns == ("return",)


def test_threshold_regime_config_validates_direction_bounds() -> None:
    with pytest.raises(ValidationError, match="lower_threshold and upper_threshold"):
        ThresholdRegimeConfig(direction="outside")

    config = ThresholdRegimeConfig(
        direction="outside",
        lower_threshold=-1.0,
        upper_threshold=1.0,
    )

    assert config.direction == "outside"


def test_regime_configs_reject_incompatible_column_names() -> None:
    with pytest.raises(ValidationError, match="feature_column must not match"):
        ThresholdRegimeConfig(feature_column="regime")

    with pytest.raises(ValidationError, match="feature_columns must be unique"):
        ClusteringRegimeConfig(feature_columns=("return", "return"))


def test_regime_configs_validate_n_regimes_compatibility() -> None:
    with pytest.raises(ValidationError, match="less than or equal to window_size"):
        PCARegimeConfig(window_size=2, n_regimes=3)

    with pytest.raises(ValidationError, match="state_weights length"):
        StateWeightedAllocationConfig(n_regimes=3)


def test_threshold_regime_detector_flags_volatility_and_validates_columns() -> None:
    from quant_platform.models import ThresholdRegimeDetector

    data = pd.DataFrame({"return": [0.01, -0.02, 0.03, -0.04, 0.05]})
    detector = ThresholdRegimeDetector(
        rule="volatility",
        lookback=3,
        threshold=0.25,
        min_periods=3,
    )

    regimes = detector.fit(data).predict(data)

    assert regimes.iloc[:2].tolist() == [0, 0]
    assert regimes.iloc[-1] == 1
    with pytest.raises(ValueError, match="data missing required columns"):
        detector.predict(pd.DataFrame({"close": [1.0, 2.0, 3.0]}))


def test_threshold_regime_detector_supports_multiple_rules() -> None:
    from quant_platform.models import ThresholdRegimeDetector

    data = pd.DataFrame(
        {
            "close": [100.0, 110.0, 90.0, 80.0],
            "return": [0.01, 0.02, -0.03, -0.04],
            "benchmark_return": [0.01, 0.02, -0.02, -0.03],
            "volume": [1000.0, 900.0, 500.0, 400.0],
        }
    )

    drawdown = ThresholdRegimeDetector(
        rule="drawdown", lookback=3, threshold=-0.20, direction="below", min_periods=2
    )
    correlation = ThresholdRegimeDetector(
        rule="correlation", lookback=3, threshold=0.80, direction="above", min_periods=3
    )
    liquidity = ThresholdRegimeDetector(
        rule="liquidity", lookback=2, threshold=700.0, direction="below", min_periods=2
    )

    assert drawdown.predict(data).iloc[-1] == 2
    assert correlation.predict(data).iloc[-1] == 2
    assert liquidity.transform(data)["regime"].iloc[-1] == 2


def test_rolling_zscore_regime_detector_emits_deterministic_labels() -> None:
    from quant_platform.models import RollingZScoreRegimeDetector

    data = pd.DataFrame({"return": [0.0, 0.0, 0.0, 1.0, -1.0]})
    detector = RollingZScoreRegimeDetector(
        feature_column="return",
        lookback=3,
        entry_zscore=1.0,
        exit_zscore=0.25,
        n_regimes=3,
        min_periods=2,
    )

    regimes = detector.fit(data).predict(data)

    assert regimes.iloc[3] == 1
    assert regimes.iloc[4] == 2


def test_build_regime_detector_instantiates_threshold_config() -> None:
    detector = build_regime_detector(
        ThresholdRegimeConfig(
            detector_type=RegimeDetectorType.DRAWDOWN_THRESHOLD,
            feature_column="close",
            lookback=5,
            threshold=-0.1,
            direction="below",
        )
    )

    assert detector.rule == "drawdown"
    assert detector.price_column == "close"
    assert detector.regime_column == "regime"


def test_build_regime_detector_from_dict_instantiates_zscore_config() -> None:
    detector = build_regime_detector_from_dict(
        {
            "detector_type": "rolling_zscore",
            "feature_column": "spread",
            "lookback": 10,
            "entry_zscore": 1.5,
            "exit_zscore": 0.25,
            "n_regimes": 3,
            "regime_column": "market_regime",
        }
    )

    assert detector.feature_column == "spread"
    assert detector.n_regimes == 3
    assert detector.regime_column == "market_regime"


def test_build_regime_detector_from_dict_instantiates_hmm_config() -> None:
    from quant_platform.models import HMMRegimeDetector

    assert isinstance(
        build_regime_detector_from_dict({"detector_type": "hmm"}), HMMRegimeDetector
    )


def test_build_regime_detector_rejects_unsupported_config() -> None:
    with pytest.raises(TypeError, match="unsupported regime detector config"):
        build_regime_detector(object())  # type: ignore[arg-type]

    detector = build_regime_detector(
        RollingZScoreRegimeConfig(feature_column="return", lookback=3)
    )

    assert detector.lookback == 3
