"""Deterministic rule-based market regime detectors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from quant_platform.models.schemas import (
    ChangePointRegimeConfig,
    ClusteringRegimeConfig,
    PCARegimeConfig,
    StateWeightedAllocationConfig,
)

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
            return (
                data[self.return_column]
                .rolling(self.lookback, min_periods=self.min_periods)
                .sum()
            )
        if self.rule == "drawdown":
            rolling_peak = (
                data[self.price_column]
                .rolling(self.lookback, min_periods=self.min_periods)
                .max()
            )
            return data[self.price_column] / rolling_peak - 1.0
        if self.rule == "correlation":
            return (
                data[self.return_column]
                .rolling(self.lookback, min_periods=self.min_periods)
                .corr(data[self.benchmark_column])
            )
        liquidity_column = self.dollar_volume_column or self.volume_column
        return (
            data[liquidity_column]
            .rolling(self.lookback, min_periods=self.min_periods)
            .mean()
        )

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


class ChangePointRegimeDetector(BaseRegimeDetector):
    """Detect regimes from rolling mean/variance shifts between adjacent windows."""

    def __init__(
        self,
        config: ChangePointRegimeConfig | None = None,
        *,
        window_size: int | None = None,
        feature_columns: tuple[str, ...] | None = None,
        n_regimes: int | None = None,
        regime_column: str | None = None,
        labels: RegimeLabels | None = None,
    ) -> None:
        config = config or ChangePointRegimeConfig()
        super().__init__(
            regime_column=regime_column or config.regime_column, labels=labels
        )
        self.window_size = window_size or config.window_size
        self.feature_columns = tuple(feature_columns or config.feature_columns)
        self.n_regimes = n_regimes or config.n_regimes
        if self.window_size <= 1:
            raise ValueError("window_size must be greater than 1")
        if self.n_regimes not in {2, 3}:
            raise ValueError("change-point regimes support n_regimes of 2 or 3")
        self.required_columns = set(self.feature_columns)
        self.thresholds_: tuple[float, ...] = ()

    def fit(self, data: pd.DataFrame) -> ChangePointRegimeDetector:
        self._validate_required_columns(data)
        scores = self._shift_scores(data)
        valid = scores.dropna()
        if valid.empty:
            self.thresholds_ = (
                (float("inf"),) if self.n_regimes == 2 else (float("inf"), float("inf"))
            )
        elif self.n_regimes == 2:
            self.thresholds_ = (float(valid.quantile(0.75)),)
        else:
            self.thresholds_ = (
                float(valid.quantile(0.67)),
                float(valid.quantile(0.90)),
            )
        self.is_fitted = True
        return self

    def predict(self, data: pd.DataFrame) -> pd.Series:
        self._validate_required_columns(data)
        if not self.is_fitted:
            self.fit(data)
        scores = self._shift_scores(data)
        regimes = pd.Series(
            self.labels.normal, index=data.index, name=self.regime_column
        )
        if self.n_regimes == 2:
            regimes.loc[scores.ge(self.thresholds_[0]).fillna(False)] = (
                self.labels.stressed
            )
        else:
            regimes.loc[scores.ge(self.thresholds_[0]).fillna(False)] = (
                self.labels.high_volatility
            )
            regimes.loc[scores.ge(self.thresholds_[1]).fillna(False)] = (
                self.labels.stressed
            )
        return regimes

    def _shift_scores(self, data: pd.DataFrame) -> pd.Series:
        values = data.loc[:, self.feature_columns].astype(float)
        prev_mean = (
            values.shift(self.window_size)
            .rolling(self.window_size, min_periods=self.window_size)
            .mean()
        )
        curr_mean = values.rolling(
            self.window_size, min_periods=self.window_size
        ).mean()
        prev_var = (
            values.shift(self.window_size)
            .rolling(self.window_size, min_periods=self.window_size)
            .var(ddof=0)
        )
        curr_var = values.rolling(self.window_size, min_periods=self.window_size).var(
            ddof=0
        )
        mean_score = (curr_mean - prev_mean).abs() / np.sqrt(
            prev_var + curr_var + 1e-12
        )
        var_score = (curr_var - prev_var).abs() / (prev_var + curr_var + 1e-12)
        return (mean_score + var_score).max(axis=1)


class ClusteringRegimeDetector(BaseRegimeDetector):
    """Small deterministic k-means detector over configured feature columns."""

    def __init__(
        self,
        config: ClusteringRegimeConfig | None = None,
        *,
        labels: RegimeLabels | None = None,
    ) -> None:
        config = config or ClusteringRegimeConfig()
        super().__init__(regime_column=config.regime_column, labels=labels)
        self.config = config
        self.required_columns = set(config.feature_columns)
        self.centroids_: np.ndarray | None = None
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None
        self.label_order_: dict[int, RegimeLabel] = {}

    def fit(self, data: pd.DataFrame) -> ClusteringRegimeDetector:
        self._validate_required_columns(data)
        matrix = self._standardized_matrix(data, fit=True)
        self.centroids_ = self._kmeans(matrix)
        order = np.argsort(np.linalg.norm(self.centroids_, axis=1))
        labels = [self.labels.normal, self.labels.high_volatility, self.labels.stressed]
        self.label_order_ = {
            int(cluster): labels[min(rank, 2)] for rank, cluster in enumerate(order)
        }
        self.is_fitted = True
        return self

    def predict(self, data: pd.DataFrame) -> pd.Series:
        self._validate_required_columns(data)
        if not self.is_fitted:
            self.fit(data)
        assert self.centroids_ is not None
        matrix = self._standardized_matrix(data, fit=False)
        distances = ((matrix[:, None, :] - self.centroids_[None, :, :]) ** 2).sum(
            axis=2
        )
        clusters = distances.argmin(axis=1)
        values = [self.label_order_[int(cluster)] for cluster in clusters]
        return pd.Series(values, index=data.index, name=self.regime_column)

    def _standardized_matrix(self, data: pd.DataFrame, *, fit: bool) -> np.ndarray:
        frame = data.loc[:, self.config.feature_columns].astype(float)
        matrix = frame.to_numpy(dtype=float)
        if fit:
            self.mean_ = np.nanmean(matrix, axis=0)
            self.scale_ = np.nanstd(matrix, axis=0)
            self.scale_[self.scale_ == 0.0] = 1.0
        assert self.mean_ is not None and self.scale_ is not None
        matrix = np.nan_to_num((matrix - self.mean_) / self.scale_)
        return matrix

    def _kmeans(self, matrix: np.ndarray) -> np.ndarray:
        k = self.config.n_regimes
        order = np.argsort(matrix[:, 0], kind="mergesort")
        picks = np.linspace(0, len(order) - 1, k, dtype=int)
        centroids = matrix[order[picks]].copy()
        for _ in range(25):
            distances = ((matrix[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
            clusters = distances.argmin(axis=1)
            updated = centroids.copy()
            for cluster in range(k):
                members = matrix[clusters == cluster]
                if len(members):
                    updated[cluster] = members.mean(axis=0)
            if np.allclose(updated, centroids):
                break
            centroids = updated
        return centroids


class PCARegimeDetector(BaseRegimeDetector):
    """Label regimes from rolling PCA explained variance or component score."""

    def __init__(
        self,
        config: PCARegimeConfig | None = None,
        *,
        labels: RegimeLabels | None = None,
    ) -> None:
        config = config or PCARegimeConfig()
        super().__init__(regime_column=config.regime_column, labels=labels)
        self.config = config
        self.required_columns = set(config.feature_columns)
        self.thresholds_: tuple[float, ...] = ()

    def fit(self, data: pd.DataFrame) -> PCARegimeDetector:
        self._validate_required_columns(data)
        scores = self._pca_scores(data).dropna()
        if scores.empty:
            self.thresholds_ = (
                (float("inf"),)
                if self.config.n_regimes == 2
                else (float("inf"), float("inf"))
            )
        elif self.config.n_regimes == 2:
            self.thresholds_ = (float(scores.quantile(0.75)),)
        else:
            self.thresholds_ = (
                float(scores.quantile(0.67)),
                float(scores.quantile(0.90)),
            )
        self.is_fitted = True
        return self

    def predict(self, data: pd.DataFrame) -> pd.Series:
        self._validate_required_columns(data)
        if not self.is_fitted:
            self.fit(data)
        scores = self._pca_scores(data)
        regimes = pd.Series(
            self.labels.normal, index=data.index, name=self.regime_column
        )
        if self.config.n_regimes == 2:
            regimes.loc[scores.ge(self.thresholds_[0]).fillna(False)] = (
                self.labels.stressed
            )
        else:
            regimes.loc[scores.ge(self.thresholds_[0]).fillna(False)] = (
                self.labels.high_volatility
            )
            regimes.loc[scores.ge(self.thresholds_[1]).fillna(False)] = (
                self.labels.stressed
            )
        return regimes

    def _pca_scores(self, data: pd.DataFrame) -> pd.Series:
        values = data.loc[:, self.config.feature_columns].astype(float)
        scores = pd.Series(np.nan, index=data.index, name="pca_score")
        for end in range(self.config.window_size - 1, len(values)):
            window = values.iloc[end - self.config.window_size + 1 : end + 1].to_numpy(
                dtype=float
            )
            window = window - window.mean(axis=0, keepdims=True)
            cov = np.cov(window, rowvar=False, ddof=0)
            cov = np.atleast_2d(cov)
            eigvals, eigvecs = np.linalg.eigh(cov)
            order = np.argsort(eigvals)[::-1]
            eigvals = eigvals[order]
            total = float(np.maximum(eigvals.sum(), 1e-12))
            if self.config.score_method == "first_component":
                first_vec = eigvecs[:, order[0]]
                scores.iloc[end] = abs(float(window[-1] @ first_vec))
            else:
                scores.iloc[end] = float(
                    eigvals[: self.config.n_components].sum() / total
                )
        return scores


class StateWeightedAllocationModel:
    """Deterministically scale signal exposure by the observed regime state."""

    def __init__(self, config: StateWeightedAllocationConfig | None = None) -> None:
        self.config = config or StateWeightedAllocationConfig()

    def transform_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Return a copy with configured signal exposure scaled by regime weights.

        The configured ``signal_column`` is adjusted when present. If it is absent,
        the configured ``target_weight_column`` is adjusted instead. Regime values
        are stringified before mapping lookup so numeric detector labels and textual
        market states can use the same deterministic configuration shape.
        """

        config = self.config
        if config.regime_column not in data.columns:
            raise ValueError(f"data missing required column: {config.regime_column}")
        allocation_column = config.signal_column
        if allocation_column not in data.columns:
            allocation_column = config.target_weight_column
        if allocation_column not in data.columns:
            raise ValueError(
                "data missing configured signal or target weight column: "
                f"{config.signal_column!r} or {config.target_weight_column!r}"
            )

        transformed = data.copy()
        regimes = transformed[config.regime_column].astype(str)
        regime_weights = regimes.map(config.regime_weights).fillna(
            config.default_weight
        )
        vol_targets = regimes.map(config.regime_vol_targets).fillna(1.0)
        transformed[allocation_column] = (
            transformed[allocation_column].astype(float) * regime_weights * vol_targets
        )
        return transformed
