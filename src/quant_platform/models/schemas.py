"""Reusable schemas for platform model definitions."""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from typing import Any, Literal

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


class Activation(StrEnum):
    """Supported activation choices for user-authored model definitions."""

    RELU = "relu"
    GELU = "gelu"
    TANH = "tanh"
    SIGMOID = "sigmoid"


class ModelDefinition(BaseModel):
    """UI/API-friendly model definition persisted in the model registry."""

    model_config = ConfigDict(extra="forbid", use_enum_values=False)

    name: str = Field(min_length=1)
    version: str = Field(default="1", min_length=1)
    model_type: ModelType = ModelType.MLP
    layer_count: int = Field(default=2, gt=0)
    hidden_size: int = Field(default=64, gt=0)
    dropout: float = Field(default=0.0, ge=0.0, lt=1.0)
    activation: Activation = Activation.RELU
    sequence_length: int = Field(default=32, gt=0)
    input_size: int = Field(default=8, gt=0)
    output_size: int = Field(default=1, gt=0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_parameters(self) -> dict[str, Any]:
        """Return JSON-serializable registry parameters for this definition."""

        return {
            "layer_count": self.layer_count,
            "hidden_size": self.hidden_size,
            "dropout": self.dropout,
            "activation": self.activation.value,
            "sequence_length": self.sequence_length,
            "input_size": self.input_size,
            "output_size": self.output_size,
            "config": self.to_model_config_dict(),
        }

    def to_model_config_dict(self) -> dict[str, Any]:
        """Return a best-effort runtime model config matching existing builders."""

        base = {
            "model_type": self.model_type.value,
            "input_dim": self.input_size,
            "output_dim": self.output_size,
            "dropout": self.dropout,
        }
        if self.model_type == ModelType.MLP:
            base["input_dim"] = self.input_size * self.sequence_length
            base["hidden_dims"] = [self.hidden_size] * self.layer_count
        elif self.model_type in {ModelType.LSTM, ModelType.GRU}:
            base["hidden_dim"] = self.hidden_size
            base["num_layers"] = self.layer_count
            base["bidirectional"] = False
        elif self.model_type == ModelType.TEMPORAL_CNN:
            base["channels"] = [self.hidden_size] * self.layer_count
            base["kernel_size"] = 3
        elif self.model_type == ModelType.TRANSFORMER:
            nhead = 4 if self.hidden_size % 4 == 0 else 1
            base["d_model"] = self.hidden_size
            base["nhead"] = nhead
            base["num_layers"] = self.layer_count
            base["dim_feedforward"] = self.hidden_size * 2
        return base

    @classmethod
    def from_catalog_row(cls, row: Mapping[str, Any]) -> ModelDefinition:
        """Build a model definition from a model_definitions catalog row."""

        parameters = dict(row.get("parameters") or {})
        return cls(
            name=str(row["name"]),
            version=str(row["version"]),
            model_type=ModelType(str(row["model_type"])),
            layer_count=int(parameters.get("layer_count", 1)),
            hidden_size=int(parameters.get("hidden_size", 64)),
            dropout=float(parameters.get("dropout", 0.0)),
            activation=Activation(
                str(parameters.get("activation", Activation.RELU.value))
            ),
            sequence_length=int(parameters.get("sequence_length", 1)),
            input_size=int(parameters.get("input_size", 1)),
            output_size=int(parameters.get("output_size", 1)),
            metadata=dict(row.get("metadata") or {}),
        )
