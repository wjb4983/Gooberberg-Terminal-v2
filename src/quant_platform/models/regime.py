"""Deterministic rule-based market regime detectors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

RegimeRule = Literal["volatility", "trend", "drawdown", "correlation", "liquidity"]
RegimeLabel = int | str


@dataclass(frozen=True)
class RegimeLabels:
    """Default deterministic regime labels shared by rule-based detectors."""

    normal: RegimeLabel = 0
    high_volatility: RegimeLabel = 1
    stressed: RegimeLabel = 2


class BaseRegimeDetector:
    """Protocol-like base class for deterministic regime detectors.

    Subclasses implement :meth:`predict` and may learn calibration values in
    :meth:`fit`. The default labels are deterministic integers:
    ``0=normal``, ``1=high_volatility``, and ``2=stressed``.
    """


    def __init__(
        self,
        *,
        regime_column: str = "regime",
        labels: RegimeLabels | None = None,
    ) -> None:
        if not regime_column or not regime_column.strip():
            msg = "regime_column must be a non-empty column name"
            raise ValueError(msg)
        self.regime_column = regime_column
        self.labels = labels or RegimeLabels()
        self.required_columns: set[str] = set()
        self.is_fitted = False

    def fit(self, data: pd.DataFrame) -> BaseRegimeDetector:
        """Validate input data and mark the detector as fitted."""

        self._validate_required_columns(data)
        self.is_fitted = True
        return self

    def predict(self, data: pd.DataFrame) -> pd.Series:
        """Return one regime label per input row."""

        raise NotImplementedError

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Return a copy of ``data`` with the configured regime column added."""

        transformed = data.copy()
        transformed[self.regime_column] = self.predict(data)
        return transformed

    def _validate_required_columns(self, data: pd.DataFrame) -> None:
        missing = self.required_columns - set(data.columns)
        if missing:
            raise ValueError(f"data missing required columns: {sorted(missing)}")


class ThresholdRegimeDetector(BaseRegimeDetector):
    """Rule-based detector for common threshold market regimes.

    Supported rules are volatility, trend, drawdown, correlation, and liquidity.
    A threshold breach emits ``high_volatility`` for volatility rules and
    ``stressed`` for all other rules by default. If ``stressed_threshold`` is
    supplied, a more severe breach is labelled ``stressed`` while the base
    threshold is labelled ``high_volatility``.
    """

    def __init__(
        self,
        *,
        rule: RegimeRule = "volatility",
        lookback: int = 20,
        threshold: float = 0.0,
        direction: Literal["above", "below", "outside", "inside"] = "above",
        price_column: str = "close",
        return_column: str = "return",
        benchmark_column: str = "benchmark_return",
        volume_column: str = "volume",
        dollar_volume_column: str | None = None,
        lower_threshold: float | None = None,
        upper_threshold: float | None = None,
        stressed_threshold: float | None = None,
        min_periods: int | None = None,
        annualization_factor: int = 252,
        regime_column: str = "regime",
        labels: RegimeLabels | None = None,
    ) -> None:
        super().__init__(regime_column=regime_column, labels=labels)
        if lookback <= 1:
            msg = "lookback must be greater than 1"
            raise ValueError(msg)
        if direction in {"inside", "outside"}:
            if lower_threshold is None or upper_threshold is None:
                msg = (
                    "inside/outside threshold directions require "
                    "lower_threshold and upper_threshold"
                )
                raise ValueError(msg)
            if lower_threshold >= upper_threshold:
                msg = "lower_threshold must be less than upper_threshold"
                raise ValueError(msg)
        self.rule = rule
        self.lookback = lookback
        self.threshold = threshold
        self.direction = direction
        self.price_column = price_column
        self.return_column = return_column
        self.benchmark_column = benchmark_column
        self.volume_column = volume_column
        self.dollar_volume_column = dollar_volume_column
        self.lower_threshold = lower_threshold
        self.upper_threshold = upper_threshold
        self.stressed_threshold = stressed_threshold
        self.min_periods = min_periods or lookback
        self.annualization_factor = annualization_factor
        self.required_columns = self._required_columns_for_rule()

    def predict(self, data: pd.DataFrame) -> pd.Series:
        self._validate_required_columns(data)
        metric = self._metric(data)
        base_breach = self._threshold_mask(metric, self.threshold)
        regimes = pd.Series(
            self.labels.normal, index=data.index, name=self.regime_column
        )
        breach_label = (
            self.labels.high_volatility
            if self.rule == "volatility" or self.stressed_threshold is not None
            else self.labels.stressed
        )
        regimes.loc[base_breach.fillna(False)] = breach_label
        if self.stressed_threshold is not None:
            stressed_breach = self._threshold_mask(metric, self.stressed_threshold)
            regimes.loc[stressed_breach.fillna(False)] = self.labels.stressed
        return regimes

    def _required_columns_for_rule(self) -> set[str]:
        if self.rule in {"volatility", "trend"}:
            return {self.return_column}
        if self.rule == "drawdown":
            return {self.price_column}
        if self.rule == "correlation":
            return {self.return_column, self.benchmark_column}
        if self.rule == "liquidity":
            return {self.dollar_volume_column or self.volume_column}
        msg = f"unsupported regime rule: {self.rule}"
        raise ValueError(msg)

    def _metric(self, data: pd.DataFrame) -> pd.Series:
        if self.rule == "volatility":
            return (
                data[self.return_column]
                .rolling(self.lookback, min_periods=self.min_periods)
                .std()
                * self.annualization_factor**0.5
            )
        if self.rule == "trend":
            return data[self.return_column].rolling(
                self.lookback, min_periods=self.min_periods
            ).sum()
        if self.rule == "drawdown":
            rolling_peak = data[self.price_column].rolling(
                self.lookback, min_periods=self.min_periods
            ).max()
            return data[self.price_column] / rolling_peak - 1.0
        if self.rule == "correlation":
            return data[self.return_column].rolling(
                self.lookback, min_periods=self.min_periods
            ).corr(data[self.benchmark_column])
        liquidity_column = self.dollar_volume_column or self.volume_column
        return data[liquidity_column].rolling(
            self.lookback, min_periods=self.min_periods
        ).mean()

    def _threshold_mask(self, metric: pd.Series, threshold: float) -> pd.Series:
        if self.direction == "above":
            return metric > threshold
        if self.direction == "below":
            return metric < threshold
        if self.direction == "outside":
            return (metric < self.lower_threshold) | (metric > self.upper_threshold)
        return (metric >= self.lower_threshold) & (metric <= self.upper_threshold)


class RollingZScoreRegimeDetector(BaseRegimeDetector):
    """Rolling z-score detector with deterministic normal/stressed labels."""

    def __init__(
        self,
        *,
        feature_column: str = "return",
        lookback: int = 20,
        entry_zscore: float = 2.0,
        exit_zscore: float = 0.5,
        n_regimes: int = 2,
        min_periods: int | None = None,
        regime_column: str = "regime",
        labels: RegimeLabels | None = None,
    ) -> None:
        super().__init__(regime_column=regime_column, labels=labels)
        if lookback <= 1:
            msg = "lookback must be greater than 1"
            raise ValueError(msg)
        if entry_zscore <= 0:
            msg = "entry_zscore must be greater than 0"
            raise ValueError(msg)
        if exit_zscore >= entry_zscore:
            msg = "exit_zscore must be less than entry_zscore"
            raise ValueError(msg)
        if n_regimes not in {2, 3}:
            msg = "rolling z-score regimes support n_regimes of 2 or 3"
            raise ValueError(msg)
        self.feature_column = feature_column
        self.lookback = lookback
        self.entry_zscore = entry_zscore
        self.exit_zscore = exit_zscore
        self.n_regimes = n_regimes
        self.min_periods = min_periods or lookback
        self.required_columns = {feature_column}

    def predict(self, data: pd.DataFrame) -> pd.Series:
        self._validate_required_columns(data)
        rolling = data[self.feature_column].rolling(
            self.lookback, min_periods=self.min_periods
        )
        mean = rolling.mean()
        std = rolling.std().replace(0.0, pd.NA)
        zscore = (data[self.feature_column] - mean) / std
        regimes = pd.Series(
            self.labels.normal, index=data.index, name=self.regime_column
        )
        if self.n_regimes == 2:
            stressed = zscore.abs().ge(self.entry_zscore).fillna(False)
            regimes.loc[stressed] = self.labels.stressed
        else:
            high_volatility = zscore.ge(self.entry_zscore).fillna(False)
            regimes.loc[high_volatility] = self.labels.high_volatility
            stressed = zscore.le(-self.entry_zscore).fillna(False)
            regimes.loc[stressed] = self.labels.stressed
        return regimes
