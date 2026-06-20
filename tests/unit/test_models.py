"""Instantiation tests for reusable neural network models."""

from __future__ import annotations

import torch

from quant_platform.models import (
    MLPConfig,
    ModelDefinition,
    ModelRegistry,
    ModelType,
    RecurrentConfig,
    TemporalCNNConfig,
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
