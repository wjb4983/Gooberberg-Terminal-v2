"""Reusable model definitions for quant platform experiments and services."""

from quant_platform.models.factory import (
    build_model,
    build_model_from_dict,
    build_regime_detector,
    build_regime_detector_from_dict,
)
from quant_platform.models.mlp import MLP
from quant_platform.models.recurrent import RecurrentModel
from quant_platform.models.regime import (
    BaseRegimeDetector,
    RegimeLabels,
    RollingZScoreRegimeDetector,
    StateWeightedAllocationModel,
    ThresholdRegimeDetector,
)
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
    RegimeConfig,
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
    "BaseRegimeDetector",
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
    "RegimeConfig",
    "RegimeDetectorType",
    "RegimeLabels",
    "RegimeSwitchingType",
    "RollingZScoreRegimeConfig",
    "RollingZScoreRegimeDetector",
    "StateWeightedAllocationConfig",
    "StateWeightedAllocationModel",
    "TemporalCNN",
    "TemporalCNNConfig",
    "ThresholdRegimeConfig",
    "ThresholdRegimeDetector",
    "TransformerConfig",
    "TransformerEncoderModel",
    "build_model",
    "build_model_from_dict",
    "build_regime_detector",
    "build_regime_detector_from_dict",
]
