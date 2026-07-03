"""Instantiation tests for reusable neural network models."""

from __future__ import annotations

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
    StateWeightedAllocationConfig,
    TemporalCNNConfig,
    ThresholdRegimeConfig,
    TransformerConfig,
    build_model,
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
    with pytest.raises(ValidationError, match="less than or equal to lookback"):
        PCARegimeConfig(lookback=2, n_regimes=3)

    with pytest.raises(ValidationError, match="state_weights length"):
        StateWeightedAllocationConfig(n_regimes=3)
