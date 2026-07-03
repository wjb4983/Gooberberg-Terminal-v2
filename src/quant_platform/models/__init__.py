"""Reusable model definitions for quant platform experiments and services."""

from quant_platform.models.factory import build_model, build_model_from_dict
from quant_platform.models.mlp import MLP
from quant_platform.models.recurrent import RecurrentModel
from quant_platform.models.registry import ModelRegistry
from quant_platform.models.schemas import (
    Activation,
    BaseRegimeConfig,
    ClusteringRegimeConfig,
    HMMRegimeConfig,
    MLPConfig,
    ModelConfig,
    ModelDefinition,
    ModelType,
    PCARegimeConfig,
    RecurrentConfig,
    RegimeDetectorType,
    RegimeSwitchingType,
    RollingZScoreRegimeConfig,
    StateWeightedAllocationConfig,
    TemporalCNNConfig,
    ThresholdRegimeConfig,
    TransformerConfig,
)
from quant_platform.models.temporal_cnn import TemporalCNN
from quant_platform.models.transformer import TransformerEncoderModel

__all__ = [
    "Activation",
    "BaseRegimeConfig",
    "ClusteringRegimeConfig",
    "HMMRegimeConfig",
    "MLP",
    "MLPConfig",
    "ModelConfig",
    "ModelDefinition",
    "ModelRegistry",
    "ModelType",
    "PCARegimeConfig",
    "RecurrentConfig",
    "RecurrentModel",
    "RegimeDetectorType",
    "RegimeSwitchingType",
    "RollingZScoreRegimeConfig",
    "StateWeightedAllocationConfig",
    "TemporalCNN",
    "TemporalCNNConfig",
    "ThresholdRegimeConfig",
    "TransformerConfig",
    "TransformerEncoderModel",
    "build_model",
    "build_model_from_dict",
]
