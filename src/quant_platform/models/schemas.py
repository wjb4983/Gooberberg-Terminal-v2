"""Reusable schemas for platform model definitions."""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    field_validator,
    model_validator,
)


class RegimeDetectorType(StrEnum):
    """Supported first-pass market regime detector families."""

    VOLATILITY_THRESHOLD = "volatility_threshold"
    TREND_THRESHOLD = "trend_threshold"
    DRAWDOWN_THRESHOLD = "drawdown_threshold"
    CORRELATION_THRESHOLD = "correlation_threshold"
    LIQUIDITY_THRESHOLD = "liquidity_threshold"
    ROLLING_ZSCORE = "rolling_zscore"
    CHANGE_POINT = "change_point"
    CLUSTERING = "clustering"
    PCA = "pca"
    HMM = "hmm"


class RegimeSwitchingType(StrEnum):
    """Supported first-pass regime-aware switching strategy families."""

    STATE_WEIGHTED_ALLOCATION = "state_weighted_allocation"
    MARKOV_SWITCHING = "markov_switching"
    SWITCHING_LINEAR = "switching_linear"
    STATE_DEPENDENT_RISK = "state_dependent_risk"


class BaseRegimeConfig(BaseModel):
    """Common validation shared by regime detector and switching configs."""

    model_config = ConfigDict(extra="forbid", use_enum_values=False)

    regime_column: str = "regime"

    @field_validator("regime_column")
    @classmethod
    def _validate_regime_column(cls, value: str) -> str:
        if not value or not value.strip():
            msg = "regime_column must be a non-empty column name"
            raise ValueError(msg)
        return value

    @staticmethod
    def _validate_feature_columns(
        feature_columns: tuple[str, ...], regime_column: str
    ) -> None:
        if any(not column or not column.strip() for column in feature_columns):
            msg = "feature_columns must contain non-empty column names"
            raise ValueError(msg)
        if len(set(feature_columns)) != len(feature_columns):
            msg = "feature_columns must be unique"
            raise ValueError(msg)
        if regime_column in feature_columns:
            msg = "feature_columns must not include regime_column"
            raise ValueError(msg)


class ThresholdRegimeConfig(BaseRegimeConfig):
    """Configuration for deterministic threshold-based regime detectors."""

    detector_type: RegimeDetectorType = RegimeDetectorType.VOLATILITY_THRESHOLD
    lookback: int = Field(default=20, gt=1)
    feature_column: str = "return"
    threshold: float = 0.0
    direction: Literal["above", "below", "outside", "inside"] = "above"
    lower_threshold: float | None = None
    upper_threshold: float | None = None
    n_regimes: int = Field(default=2, gt=1)

    @model_validator(mode="after")
    def _validate_threshold_config(self) -> ThresholdRegimeConfig:
        if self.detector_type not in {
            RegimeDetectorType.VOLATILITY_THRESHOLD,
            RegimeDetectorType.TREND_THRESHOLD,
            RegimeDetectorType.DRAWDOWN_THRESHOLD,
            RegimeDetectorType.CORRELATION_THRESHOLD,
            RegimeDetectorType.LIQUIDITY_THRESHOLD,
        }:
            msg = "detector_type must be a threshold-based regime detector"
            raise ValueError(msg)
        if not self.feature_column or not self.feature_column.strip():
            msg = "feature_column must be a non-empty column name"
            raise ValueError(msg)
        if self.feature_column == self.regime_column:
            msg = "feature_column must not match regime_column"
            raise ValueError(msg)
        if self.direction in {"inside", "outside"}:
            if self.lower_threshold is None or self.upper_threshold is None:
                msg = (
                    "inside/outside threshold directions require "
                    "lower_threshold and upper_threshold"
                )
                raise ValueError(msg)
            if self.lower_threshold >= self.upper_threshold:
                msg = "lower_threshold must be less than upper_threshold"
                raise ValueError(msg)
        if self.n_regimes != 2 and self.direction in {"above", "below"}:
            msg = "above/below threshold directions require n_regimes=2"
            raise ValueError(msg)
        return self


class RollingZScoreRegimeConfig(BaseRegimeConfig):
    """Configuration for rolling z-score regime detectors."""

    detector_type: Literal[RegimeDetectorType.ROLLING_ZSCORE] = (
        RegimeDetectorType.ROLLING_ZSCORE
    )
    lookback: int = Field(default=20, gt=1)
    feature_column: str = "return"
    entry_zscore: float = Field(default=2.0, gt=0.0)
    exit_zscore: float = Field(default=0.5, ge=0.0)
    n_regimes: int = Field(default=2, gt=1)

    @model_validator(mode="after")
    def _validate_zscore_config(self) -> RollingZScoreRegimeConfig:
        if not self.feature_column or not self.feature_column.strip():
            msg = "feature_column must be a non-empty column name"
            raise ValueError(msg)
        if self.feature_column == self.regime_column:
            msg = "feature_column must not match regime_column"
            raise ValueError(msg)
        if self.exit_zscore >= self.entry_zscore:
            msg = "exit_zscore must be less than entry_zscore"
            raise ValueError(msg)
        if self.n_regimes not in {2, 3}:
            msg = "rolling z-score regimes support n_regimes of 2 or 3"
            raise ValueError(msg)
        return self


class ChangePointRegimeConfig(BaseRegimeConfig):
    """Configuration for rolling mean/variance change-point detectors."""

    detector_type: Literal[RegimeDetectorType.CHANGE_POINT] = (
        RegimeDetectorType.CHANGE_POINT
    )
    window_size: int = Field(default=20, gt=1)
    n_regimes: int = Field(default=2, gt=1)
    feature_columns: tuple[str, ...] = Field(default=("return",), min_length=1)

    @model_validator(mode="after")
    def _validate_change_point_config(self) -> ChangePointRegimeConfig:
        self._validate_feature_columns(self.feature_columns, self.regime_column)
        if self.n_regimes not in {2, 3}:
            msg = "change-point regimes support n_regimes of 2 or 3"
            raise ValueError(msg)
        return self


class ClusteringRegimeConfig(BaseRegimeConfig):
    """Configuration for feature clustering regime detectors."""

    detector_type: Literal[RegimeDetectorType.CLUSTERING] = (
        RegimeDetectorType.CLUSTERING
    )
    window_size: int = Field(default=20, gt=1)
    n_regimes: int = Field(default=2, gt=1)
    feature_columns: tuple[str, ...] = Field(default=("return",), min_length=1)
    random_state: int = 0

    @model_validator(mode="after")
    def _validate_clustering_config(self) -> ClusteringRegimeConfig:
        self._validate_feature_columns(self.feature_columns, self.regime_column)
        if self.n_regimes > self.window_size:
            msg = "n_regimes must be less than or equal to window_size"
            raise ValueError(msg)
        return self


class PCARegimeConfig(BaseRegimeConfig):
    """Configuration for PCA-based regime detectors."""

    detector_type: Literal[RegimeDetectorType.PCA] = RegimeDetectorType.PCA
    window_size: int = Field(default=20, gt=1)
    n_regimes: int = Field(default=2, gt=1)
    feature_columns: tuple[str, ...] = Field(default=("return",), min_length=1)
    n_components: int = Field(default=1, gt=0)
    score_method: Literal["explained_variance", "first_component"] = (
        "explained_variance"
    )

    @model_validator(mode="after")
    def _validate_pca_config(self) -> PCARegimeConfig:
        self._validate_feature_columns(self.feature_columns, self.regime_column)
        if self.n_components > len(self.feature_columns):
            msg = "n_components must be less than or equal to feature column count"
            raise ValueError(msg)
        if self.n_regimes > self.window_size:
            msg = "n_regimes must be less than or equal to window_size"
            raise ValueError(msg)
        return self


class HMMRegimeConfig(BaseRegimeConfig):
    """Configuration for optional hidden Markov model regime detectors."""

    detector_type: Literal[RegimeDetectorType.HMM] = RegimeDetectorType.HMM
    n_regimes: int = Field(default=2, gt=1)
    feature_columns: tuple[str, ...] = Field(default=("return",), min_length=1)
    covariance_type: Literal["diag", "full", "spherical", "tied"] = "diag"
    max_iter: int = Field(default=100, gt=0)
    seed: int = 0

    @model_validator(mode="after")
    def _validate_hmm_config(self) -> HMMRegimeConfig:
        self._validate_feature_columns(self.feature_columns, self.regime_column)
        return self


RegimeConfig = (
    ThresholdRegimeConfig
    | RollingZScoreRegimeConfig
    | ChangePointRegimeConfig
    | ClusteringRegimeConfig
    | PCARegimeConfig
    | HMMRegimeConfig
)


class StateWeightedAllocationConfig(BaseRegimeConfig):
    """Configuration for deterministic regime-weighted signal allocation."""

    switching_type: Literal[RegimeSwitchingType.STATE_WEIGHTED_ALLOCATION] = (
        RegimeSwitchingType.STATE_WEIGHTED_ALLOCATION
    )
    signal_column: str = "signal"
    target_weight_column: str = "target_weight"
    regime_weights: dict[str, float] = Field(default_factory=dict)
    regime_vol_targets: dict[str, float] = Field(default_factory=dict)
    default_weight: float = 1.0
    n_regimes: int = Field(default=2, gt=1)
    feature_columns: tuple[str, ...] = Field(default=("return",), min_length=1)
    state_weights: tuple[float, ...] = Field(default=(0.5, 0.5), min_length=2)

    @model_validator(mode="after")
    def _validate_state_weights(self) -> StateWeightedAllocationConfig:
        self._validate_feature_columns(self.feature_columns, self.regime_column)
        for column_name, value in {
            "signal_column": self.signal_column,
            "target_weight_column": self.target_weight_column,
        }.items():
            if not value or not value.strip():
                msg = f"{column_name} must be a non-empty column name"
                raise ValueError(msg)
            if value == self.regime_column:
                msg = f"{column_name} must not match regime_column"
                raise ValueError(msg)
        if len(self.state_weights) != self.n_regimes:
            msg = "state_weights length must match n_regimes"
            raise ValueError(msg)
        all_weights = [
            *self.state_weights,
            *self.regime_weights.values(),
            *self.regime_vol_targets.values(),
            self.default_weight,
        ]
        if any(weight < 0.0 for weight in all_weights):
            msg = "state weights and regime allocation weights must be non-negative"
            raise ValueError(msg)
        if sum(self.state_weights) <= 0.0:
            msg = "state_weights must include at least one positive weight"
            raise ValueError(msg)
        return self


class MarkovSwitchingConfig(BaseRegimeConfig):
    """Configuration for optional statsmodels-backed Markov switching models."""

    switching_type: Literal[RegimeSwitchingType.MARKOV_SWITCHING] = (
        RegimeSwitchingType.MARKOV_SWITCHING
    )
    endog_column: str = "return"
    exog_columns: tuple[str, ...] = Field(default=(), min_length=0)
    n_regimes: int = Field(default=2, gt=1)
    trend: Literal["n", "c", "t", "ct"] = "c"
    switching_variance: bool = True
    max_iter: int = Field(default=100, gt=0)

    @model_validator(mode="after")
    def _validate_markov_switching_config(self) -> MarkovSwitchingConfig:
        if not self.endog_column or not self.endog_column.strip():
            msg = "endog_column must be a non-empty column name"
            raise ValueError(msg)
        if self.endog_column == self.regime_column:
            msg = "endog_column must not match regime_column"
            raise ValueError(msg)
        self._validate_feature_columns(self.exog_columns, self.regime_column)
        if self.endog_column in self.exog_columns:
            msg = "exog_columns must not include endog_column"
            raise ValueError(msg)
        return self


class SwitchingLinearConfig(BaseRegimeConfig):
    """Configuration for deterministic per-regime linear heads."""

    switching_type: Literal[RegimeSwitchingType.SWITCHING_LINEAR] = (
        RegimeSwitchingType.SWITCHING_LINEAR
    )
    feature_columns: tuple[str, ...] = Field(default=("return",), min_length=1)
    target_column: str = "target"
    prediction_column: str = "prediction"
    n_regimes: int = Field(default=2, gt=1)
    fit_intercept: bool = True
    default_regime: str | None = None

    @model_validator(mode="after")
    def _validate_switching_linear_config(self) -> SwitchingLinearConfig:
        self._validate_feature_columns(self.feature_columns, self.regime_column)
        for column_name, value in {
            "target_column": self.target_column,
            "prediction_column": self.prediction_column,
        }.items():
            if not value or not value.strip():
                msg = f"{column_name} must be a non-empty column name"
                raise ValueError(msg)
            if value == self.regime_column:
                msg = f"{column_name} must not match regime_column"
                raise ValueError(msg)
        if self.target_column in self.feature_columns:
            msg = "feature_columns must not include target_column"
            raise ValueError(msg)
        return self


class StateDependentRiskConfig(BaseRegimeConfig):
    """Configuration for deterministic state-dependent risk controls."""

    switching_type: Literal[RegimeSwitchingType.STATE_DEPENDENT_RISK] = (
        RegimeSwitchingType.STATE_DEPENDENT_RISK
    )
    signal_column: str = "signal"
    adjusted_signal_column: str = "risk_adjusted_signal"
    volatility_column: str = "volatility"
    return_column: str = "return"
    n_regimes: int = Field(default=2, gt=1)
    volatility_target_by_regime: dict[str, float] = Field(default_factory=dict)
    max_leverage_by_regime: dict[str, float] = Field(default_factory=dict)
    stop_loss_multiplier_by_regime: dict[str, float] = Field(default_factory=dict)
    cash_allocation_by_regime: dict[str, float] = Field(default_factory=dict)
    default_volatility_target: float | None = Field(default=None, gt=0.0)
    default_max_leverage: float = Field(default=1.0, ge=0.0)
    default_stop_loss_multiplier: float | None = Field(default=None, gt=0.0)
    default_cash_allocation: float = Field(default=0.0, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _validate_state_dependent_risk_config(self) -> StateDependentRiskConfig:
        for column_name, value in {
            "signal_column": self.signal_column,
            "adjusted_signal_column": self.adjusted_signal_column,
            "volatility_column": self.volatility_column,
            "return_column": self.return_column,
        }.items():
            if not value or not value.strip():
                msg = f"{column_name} must be a non-empty column name"
                raise ValueError(msg)
            if value == self.regime_column:
                msg = f"{column_name} must not match regime_column"
                raise ValueError(msg)
        values = [
            *self.volatility_target_by_regime.values(),
            *self.max_leverage_by_regime.values(),
            *self.stop_loss_multiplier_by_regime.values(),
        ]
        if any(value < 0.0 for value in values):
            msg = "risk multipliers and targets must be non-negative"
            raise ValueError(msg)
        if any(
            allocation < 0.0 or allocation > 1.0
            for allocation in self.cash_allocation_by_regime.values()
        ):
            msg = "cash allocations must be between 0 and 1"
            raise ValueError(msg)
        return self


RegimeSwitchingConfig = (
    StateWeightedAllocationConfig
    | MarkovSwitchingConfig
    | SwitchingLinearConfig
    | StateDependentRiskConfig
)


_REGIME_CONFIG_ADAPTER = TypeAdapter(RegimeConfig)


class ModelType(StrEnum):
    """Supported neural network model families."""

    MLP = "mlp"
    LSTM = "lstm"
    GRU = "gru"
    TEMPORAL_CNN = "temporal_cnn"
    TRANSFORMER = "transformer"
    REGIME_DETECTOR = "regime_detector"


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

        parameters: dict[str, Any] = {
            "layer_count": self.layer_count,
            "hidden_size": self.hidden_size,
            "dropout": self.dropout,
            "activation": self.activation.value,
            "sequence_length": self.sequence_length,
            "input_size": self.input_size,
            "output_size": self.output_size,
        }
        if self.model_type == ModelType.REGIME_DETECTOR:
            parameters["regime"] = self.to_regime_config_dict()
        else:
            parameters["config"] = self.to_model_config_dict()
        return parameters

    def to_regime_config_dict(self) -> dict[str, Any]:
        """Return the validated regime detector config stored in metadata["regime"]."""

        raw_config = self.metadata.get("regime")
        if not isinstance(raw_config, Mapping):
            msg = "metadata['regime'] must contain a regime detector config mapping"
            raise ValueError(msg)
        config = _REGIME_CONFIG_ADAPTER.validate_python(dict(raw_config))
        return config.model_dump(mode="json")

    def to_model_config_dict(self) -> dict[str, Any]:
        """Return a best-effort runtime model config matching existing builders."""

        if self.model_type == ModelType.REGIME_DETECTOR:
            msg = "regime detector definitions do not have neural model configs"
            raise ValueError(msg)

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
