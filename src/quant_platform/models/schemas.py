"""Reusable schemas for platform model definitions."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ModelType(StrEnum):
    """Supported neural network model families."""

    MLP = "mlp"
    LSTM = "lstm"
    GRU = "gru"
    TEMPORAL_CNN = "temporal_cnn"
    TRANSFORMER = "transformer"


class BaseModelConfig(BaseModel):
    """Common configuration shared by all model families."""

    model_config = ConfigDict(extra="forbid", use_enum_values=False)

    model_type: ModelType
    input_dim: int = Field(gt=0, description="Number of features per time step.")
    output_dim: int = Field(gt=0, description="Number of output predictions.")
    dropout: float = Field(default=0.0, ge=0.0, lt=1.0)


class MLPConfig(BaseModelConfig):
    """Configuration for a feed-forward multilayer perceptron."""

    model_type: Literal[ModelType.MLP] = ModelType.MLP
    hidden_dims: tuple[int, ...] = Field(default=(64, 32), min_length=1)


class RecurrentConfig(BaseModelConfig):
    """Configuration for LSTM and GRU sequence models."""

    model_type: Literal[ModelType.LSTM, ModelType.GRU]
    hidden_dim: int = Field(default=64, gt=0)
    num_layers: int = Field(default=1, gt=0)
    bidirectional: bool = False


class TemporalCNNConfig(BaseModelConfig):
    """Configuration for a temporal convolutional network."""

    model_type: Literal[ModelType.TEMPORAL_CNN] = ModelType.TEMPORAL_CNN
    channels: tuple[int, ...] = Field(default=(32, 32), min_length=1)
    kernel_size: int = Field(default=3, gt=0)


class TransformerConfig(BaseModelConfig):
    """Configuration for a transformer encoder model."""

    model_type: Literal[ModelType.TRANSFORMER] = ModelType.TRANSFORMER
    d_model: int = Field(default=64, gt=0)
    nhead: int = Field(default=4, gt=0)
    num_layers: int = Field(default=2, gt=0)
    dim_feedforward: int = Field(default=128, gt=0)

    @model_validator(mode="after")
    def _validate_attention_heads(self) -> TransformerConfig:
        if self.d_model % self.nhead != 0:
            msg = "d_model must be divisible by nhead"
            raise ValueError(msg)
        return self


ModelConfig = MLPConfig | RecurrentConfig | TemporalCNNConfig | TransformerConfig
