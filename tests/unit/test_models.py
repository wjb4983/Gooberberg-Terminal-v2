"""Instantiation tests for reusable neural network models."""

from __future__ import annotations

import torch

from quant_platform.models import (
    MLPConfig,
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
