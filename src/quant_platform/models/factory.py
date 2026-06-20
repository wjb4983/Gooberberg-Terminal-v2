"""Factory helpers for reusable model definitions."""

from __future__ import annotations

from torch import nn

from quant_platform.models.mlp import MLP
from quant_platform.models.recurrent import RecurrentModel
from quant_platform.models.schemas import (
    MLPConfig,
    ModelConfig,
    ModelType,
    RecurrentConfig,
    TemporalCNNConfig,
    TransformerConfig,
)
from quant_platform.models.temporal_cnn import TemporalCNN
from quant_platform.models.transformer import TransformerEncoderModel


def build_model(config: ModelConfig) -> nn.Module:
    """Instantiate a model from its typed configuration."""

    if isinstance(config, MLPConfig):
        return MLP(config)
    if isinstance(config, RecurrentConfig):
        return RecurrentModel(config)
    if isinstance(config, TemporalCNNConfig):
        return TemporalCNN(config)
    if isinstance(config, TransformerConfig):
        return TransformerEncoderModel(config)

    msg = f"unsupported model config: {type(config).__name__}"
    raise TypeError(msg)


def build_model_from_dict(config: dict[str, object]) -> nn.Module:
    """Instantiate a model from a JSON/YAML-friendly configuration mapping."""

    model_type = ModelType(config["model_type"])
    if model_type == ModelType.MLP:
        parsed: ModelConfig = MLPConfig.model_validate(config)
    elif model_type in {ModelType.LSTM, ModelType.GRU}:
        parsed = RecurrentConfig.model_validate(config)
    elif model_type == ModelType.TEMPORAL_CNN:
        parsed = TemporalCNNConfig.model_validate(config)
    elif model_type == ModelType.TRANSFORMER:
        parsed = TransformerConfig.model_validate(config)
    else:
        msg = f"unsupported model type: {model_type}"
        raise ValueError(msg)
    return build_model(parsed)
