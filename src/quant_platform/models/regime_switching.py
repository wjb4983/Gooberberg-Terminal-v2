"""Deterministic and optional backend regime-switching model helpers."""

from __future__ import annotations

import importlib
from typing import Any

import numpy as np
import pandas as pd

from quant_platform.models.regime import StateWeightedAllocationModel
from quant_platform.models.schemas import (
    MarkovSwitchingConfig,
    RegimeSwitchingConfig,
    RegimeSwitchingType,
    StateDependentRiskConfig,
    StateWeightedAllocationConfig,
    SwitchingLinearConfig,
)


class MarkovSwitchingModel:
    """Optional statsmodels-backed Markov regression wrapper.

    ``statsmodels`` is imported lazily so the base platform can expose the config
    without requiring the optional research dependency at install time.
    """

    def __init__(self, config: MarkovSwitchingConfig | None = None) -> None:
        self.config = config or MarkovSwitchingConfig()
        self.results_: Any | None = None

    def fit(self, data: pd.DataFrame) -> MarkovSwitchingModel:
        """Fit a Markov switching regression when statsmodels is installed."""

        config = self.config
        required = {config.endog_column, *config.exog_columns}
        missing = required - set(data.columns)
        if missing:
            raise ValueError(f"data missing required columns: {sorted(missing)}")
        markov_regression = self._markov_regression_cls()
        endog = data[config.endog_column].astype(float)
        exog = None
        if config.exog_columns:
            exog = data.loc[:, config.exog_columns].astype(float)
        model = markov_regression(
            endog,
            k_regimes=config.n_regimes,
            trend=config.trend,
            exog=exog,
            switching_variance=config.switching_variance,
        )
        self.results_ = model.fit(maxiter=config.max_iter, disp=False)
        return self

    def predict_regime_probabilities(
        self, data: pd.DataFrame | None = None
    ) -> pd.DataFrame:
        """Return smoothed marginal regime probabilities from fitted results."""

        if self.results_ is None:
            raise ValueError("MarkovSwitchingModel must be fitted before prediction")
        probabilities = self.results_.smoothed_marginal_probabilities
        if isinstance(probabilities, pd.DataFrame):
            return probabilities
        return pd.DataFrame(probabilities)

    @staticmethod
    def _markov_regression_cls() -> Any:
        try:
            backend = importlib.import_module(
                "statsmodels.tsa.regime_switching.markov_regression"
            )
        except ImportError as exc:
            raise ImportError(
                "MarkovSwitchingModel requires optional 'statsmodels' support. "
                "Install statsmodels to use switching_type='markov_switching'."
            ) from exc
        return backend.MarkovRegression


class SwitchingLinearModel:
    """Minimal deterministic baseline with one linear head per regime label."""

    def __init__(self, config: SwitchingLinearConfig | None = None) -> None:
        self.config = config or SwitchingLinearConfig()
        self.coefficients_: dict[str, np.ndarray] = {}
        self.global_coefficients_: np.ndarray | None = None
        self.is_fitted = False

    def fit(self, data: pd.DataFrame) -> SwitchingLinearModel:
        """Fit one least-squares linear head for each observed regime."""

        config = self.config
        self._validate_columns(data, include_target=True)
        design = self._design_matrix(data)
        target = data[config.target_column].astype(float).to_numpy(dtype=float)
        self.global_coefficients_ = self._fit_head(design, target)
        self.coefficients_.clear()
        for regime, group in data.groupby(config.regime_column, sort=True):
            group_design = self._design_matrix(group)
            group_target = (
                group[config.target_column].astype(float).to_numpy(dtype=float)
            )
            self.coefficients_[str(regime)] = self._fit_head(group_design, group_target)
        self.is_fitted = True
        return self

    def predict(self, data: pd.DataFrame) -> pd.Series:
        """Route each row to its matching per-regime linear head."""

        if not self.is_fitted or self.global_coefficients_ is None:
            raise ValueError("SwitchingLinearModel must be fitted before prediction")
        self._validate_columns(data, include_target=False)
        predictions = []
        design = self._design_matrix(data)
        regimes = data[self.config.regime_column].astype(str).tolist()
        for row, regime in zip(design, regimes, strict=True):
            coefficients = self.coefficients_.get(regime, self.global_coefficients_)
            predictions.append(float(row @ coefficients))
        return pd.Series(
            predictions, index=data.index, name=self.config.prediction_column
        )

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        transformed = data.copy()
        transformed[self.config.prediction_column] = self.predict(data)
        return transformed

    def _validate_columns(self, data: pd.DataFrame, *, include_target: bool) -> None:
        required = {self.config.regime_column, *self.config.feature_columns}
        if include_target:
            required.add(self.config.target_column)
        missing = required - set(data.columns)
        if missing:
            raise ValueError(f"data missing required columns: {sorted(missing)}")

    def _design_matrix(self, data: pd.DataFrame) -> np.ndarray:
        matrix = (
            data.loc[:, self.config.feature_columns].astype(float).to_numpy(dtype=float)
        )
        if self.config.fit_intercept:
            intercept = np.ones((matrix.shape[0], 1), dtype=float)
            matrix = np.hstack([intercept, matrix])
        return matrix

    @staticmethod
    def _fit_head(design: np.ndarray, target: np.ndarray) -> np.ndarray:
        coefficients, *_ = np.linalg.lstsq(design, target, rcond=None)
        return np.asarray(coefficients, dtype=float)


class StateDependentRiskModel:
    """Apply volatility targeting, leverage caps, stop-losses, and cash buffers."""

    def __init__(self, config: StateDependentRiskConfig | None = None) -> None:
        self.config = config or StateDependentRiskConfig()

    def transform_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        config = self.config
        required = {config.regime_column, config.signal_column}
        missing = required - set(data.columns)
        if missing:
            raise ValueError(f"data missing required columns: {sorted(missing)}")
        transformed = data.copy()
        regimes = transformed[config.regime_column].astype(str)
        signals = transformed[config.signal_column].astype(float)
        default_volatility_target = (
            config.default_volatility_target
            if config.default_volatility_target is not None
            else np.nan
        )
        vol_target = regimes.map(config.volatility_target_by_regime).fillna(
            default_volatility_target
        )
        if config.volatility_column in transformed.columns:
            volatility = (
                transformed[config.volatility_column].astype(float).replace(0.0, np.nan)
            )
            scale = (vol_target / volatility).fillna(1.0)
        else:
            scale = pd.Series(1.0, index=transformed.index)
        max_leverage = regimes.map(config.max_leverage_by_regime).fillna(
            config.default_max_leverage
        )
        cash = regimes.map(config.cash_allocation_by_regime).fillna(
            config.default_cash_allocation
        )
        adjusted = signals * scale * (1.0 - cash)
        adjusted = adjusted.clip(lower=-max_leverage, upper=max_leverage)
        default_stop_loss_multiplier = (
            config.default_stop_loss_multiplier
            if config.default_stop_loss_multiplier is not None
            else np.nan
        )
        stop_multiplier = regimes.map(config.stop_loss_multiplier_by_regime).fillna(
            default_stop_loss_multiplier
        )
        if (
            config.return_column in transformed.columns
            and config.volatility_column in transformed.columns
        ):
            stop_threshold = -stop_multiplier * transformed[
                config.volatility_column
            ].astype(float)
            stopped = (
                transformed[config.return_column]
                .astype(float)
                .le(stop_threshold)
                .fillna(False)
            )
            adjusted.loc[stopped] = 0.0
        transformed[config.adjusted_signal_column] = adjusted
        return transformed


def build_regime_switching_model(config: RegimeSwitchingConfig) -> object:
    """Instantiate a regime-switching model outside the neural build path."""

    if isinstance(config, StateWeightedAllocationConfig):
        return StateWeightedAllocationModel(config)
    if isinstance(config, MarkovSwitchingConfig):
        return MarkovSwitchingModel(config)
    if isinstance(config, SwitchingLinearConfig):
        return SwitchingLinearModel(config)
    if isinstance(config, StateDependentRiskConfig):
        return StateDependentRiskModel(config)
    msg = f"unsupported regime-switching config: {type(config).__name__}"
    raise TypeError(msg)


def build_regime_switching_model_from_dict(config: dict[str, object]) -> object:
    """Instantiate a regime-switching model from a JSON/YAML-friendly mapping."""

    switching_type = RegimeSwitchingType(config["switching_type"])
    if switching_type == RegimeSwitchingType.STATE_WEIGHTED_ALLOCATION:
        parsed: RegimeSwitchingConfig = StateWeightedAllocationConfig.model_validate(
            config
        )
    elif switching_type == RegimeSwitchingType.MARKOV_SWITCHING:
        parsed = MarkovSwitchingConfig.model_validate(config)
    elif switching_type == RegimeSwitchingType.SWITCHING_LINEAR:
        parsed = SwitchingLinearConfig.model_validate(config)
    elif switching_type == RegimeSwitchingType.STATE_DEPENDENT_RISK:
        parsed = StateDependentRiskConfig.model_validate(config)
    else:
        msg = f"unsupported regime-switching type: {switching_type}"
        raise ValueError(msg)
    return build_regime_switching_model(parsed)
