"""Factory helpers for reusable model definitions."""

from __future__ import annotations

from torch import nn

from quant_platform.models.mlp import MLP
from quant_platform.models.recurrent import RecurrentModel
from quant_platform.models.regime import (
    BaseRegimeDetector,
    ChangePointRegimeDetector,
    ClusteringRegimeDetector,
    PCARegimeDetector,
    RollingZScoreRegimeDetector,
    ThresholdRegimeDetector,
)
from quant_platform.models.schemas import (
    ChangePointRegimeConfig,
    ClusteringRegimeConfig,
    MLPConfig,
    ModelConfig,
    ModelType,
    PCARegimeConfig,
    RecurrentConfig,
    RegimeConfig,
    RegimeDetectorType,
    RollingZScoreRegimeConfig,
    TemporalCNNConfig,
    ThresholdRegimeConfig,
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


_THRESHOLD_RULES: dict[RegimeDetectorType, str] = {
    RegimeDetectorType.VOLATILITY_THRESHOLD: "volatility",
    RegimeDetectorType.TREND_THRESHOLD: "trend",
    RegimeDetectorType.DRAWDOWN_THRESHOLD: "drawdown",
    RegimeDetectorType.CORRELATION_THRESHOLD: "correlation",
    RegimeDetectorType.LIQUIDITY_THRESHOLD: "liquidity",
}


def build_regime_detector(config: RegimeConfig) -> BaseRegimeDetector:
    """Instantiate a market regime detector from its typed configuration."""

    if isinstance(config, ThresholdRegimeConfig):
        rule = _THRESHOLD_RULES.get(config.detector_type)
        if rule is None:
            msg = f"unsupported regime detector type: {config.detector_type}"
            raise ValueError(msg)
        kwargs: dict[str, object] = {
            "rule": rule,
            "lookback": config.lookback,
            "threshold": config.threshold,
            "direction": config.direction,
            "lower_threshold": config.lower_threshold,
            "upper_threshold": config.upper_threshold,
            "regime_column": config.regime_column,
        }
        if config.detector_type in {
            RegimeDetectorType.VOLATILITY_THRESHOLD,
            RegimeDetectorType.TREND_THRESHOLD,
        }:
            kwargs["return_column"] = config.feature_column
        elif config.detector_type == RegimeDetectorType.DRAWDOWN_THRESHOLD:
            kwargs["price_column"] = config.feature_column
        elif config.detector_type == RegimeDetectorType.CORRELATION_THRESHOLD:
            kwargs["return_column"] = config.feature_column
        elif config.detector_type == RegimeDetectorType.LIQUIDITY_THRESHOLD:
            kwargs["volume_column"] = config.feature_column
        return ThresholdRegimeDetector(**kwargs)

    if isinstance(config, RollingZScoreRegimeConfig):
        return RollingZScoreRegimeDetector(
            feature_column=config.feature_column,
            lookback=config.lookback,
            entry_zscore=config.entry_zscore,
            exit_zscore=config.exit_zscore,
            n_regimes=config.n_regimes,
            regime_column=config.regime_column,
        )
    if isinstance(config, ChangePointRegimeConfig):
        return ChangePointRegimeDetector(config)
    if isinstance(config, ClusteringRegimeConfig):
        return ClusteringRegimeDetector(config)
    if isinstance(config, PCARegimeConfig):
        return PCARegimeDetector(config)

    msg = f"unsupported regime detector config: {type(config).__name__}"
    raise TypeError(msg)


def build_regime_detector_from_dict(
    config: dict[str, object],
) -> BaseRegimeDetector:
    """Instantiate a regime detector from a JSON/YAML-friendly mapping."""

    detector_type = RegimeDetectorType(config["detector_type"])
    if detector_type in _THRESHOLD_RULES:
        parsed: RegimeConfig = ThresholdRegimeConfig.model_validate(config)
    elif detector_type == RegimeDetectorType.ROLLING_ZSCORE:
        parsed = RollingZScoreRegimeConfig.model_validate(config)
    elif detector_type == RegimeDetectorType.CHANGE_POINT:
        parsed = ChangePointRegimeConfig.model_validate(config)
    elif detector_type == RegimeDetectorType.CLUSTERING:
        parsed = ClusteringRegimeConfig.model_validate(config)
    elif detector_type == RegimeDetectorType.PCA:
        parsed = PCARegimeConfig.model_validate(config)
    else:
        msg = f"unsupported regime detector type: {detector_type}"
        raise ValueError(msg)
    return build_regime_detector(parsed)
