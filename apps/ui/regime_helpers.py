"""Shared helpers for optional regime controls in Streamlit pages."""

from __future__ import annotations

from typing import Any

from quant_platform.models.schemas import RegimeDetectorType, RegimeSwitchingType

THRESHOLD_DETECTOR_TYPES = {
    RegimeDetectorType.VOLATILITY_THRESHOLD,
    RegimeDetectorType.TREND_THRESHOLD,
    RegimeDetectorType.DRAWDOWN_THRESHOLD,
    RegimeDetectorType.CORRELATION_THRESHOLD,
    RegimeDetectorType.LIQUIDITY_THRESHOLD,
}
_WINDOWED_DETECTOR_TYPES = {
    RegimeDetectorType.CHANGE_POINT,
    RegimeDetectorType.CLUSTERING,
    RegimeDetectorType.PCA,
}


def parse_csv_columns(raw: str) -> list[str]:
    """Parse comma-separated text input into trimmed column names."""

    return [column.strip() for column in raw.split(",") if column.strip()]


def parse_regime_weights(raw: str) -> dict[str, float]:
    """Parse one regime:weight mapping per line from text input."""

    weights: dict[str, float] = {}
    for line in raw.splitlines():
        if not line.strip() or ":" not in line:
            continue
        regime, weight = line.split(":", 1)
        regime = regime.strip()
        if regime:
            weights[regime] = float(weight.strip())
    return weights


def fields_for_regime_detector_type(
    detector_type: RegimeDetectorType,
) -> tuple[str, ...]:
    """Return visible form fields needed by a regime detector family."""

    if detector_type in THRESHOLD_DETECTOR_TYPES:
        return ("lookback", "feature_column", "threshold", "direction")
    if detector_type == RegimeDetectorType.ROLLING_ZSCORE:
        return (
            "lookback",
            "feature_column",
            "entry_zscore",
            "exit_zscore",
            "n_regimes",
        )
    if detector_type == RegimeDetectorType.CHANGE_POINT:
        return ("n_regimes", "window_size", "feature_columns")
    if detector_type == RegimeDetectorType.CLUSTERING:
        return ("n_regimes", "window_size", "feature_columns", "random_state")
    if detector_type == RegimeDetectorType.PCA:
        return (
            "n_regimes",
            "window_size",
            "feature_columns",
            "n_components",
            "score_method",
        )
    if detector_type == RegimeDetectorType.HMM:
        return (
            "n_regimes",
            "window_size",
            "feature_columns",
            "covariance_type",
            "max_iter",
            "seed",
        )
    return ()


def fields_for_switching_type(switching_type: RegimeSwitchingType) -> tuple[str, ...]:
    """Return visible form fields needed by a regime-switching strategy."""

    if switching_type == RegimeSwitchingType.STATE_WEIGHTED_ALLOCATION:
        return (
            "regime_column",
            "signal_column",
            "target_weight_column",
            "feature_columns",
            "regime_weights",
            "default_weight",
            "n_regimes",
        )
    if switching_type == RegimeSwitchingType.SWITCHING_LINEAR:
        return (
            "regime_column",
            "feature_columns",
            "target_column",
            "prediction_column",
            "n_regimes",
        )
    if switching_type == RegimeSwitchingType.STATE_DEPENDENT_RISK:
        return (
            "regime_column",
            "signal_column",
            "adjusted_signal_column",
            "volatility_column",
            "return_column",
            "max_leverage_by_regime",
            "default_max_leverage",
            "default_cash_allocation",
        )
    if switching_type == RegimeSwitchingType.MARKOV_SWITCHING:
        return (
            "regime_column",
            "endog_column",
            "exog_columns",
            "n_regimes",
            "trend",
            "switching_variance",
            "max_iter",
        )
    return ()


def default_regime_feature_columns(
    asset_class: str | None, workflow_intent: str | None
) -> list[str]:
    """Return ordered starter feature columns for regime workflows."""

    normalized_asset = (asset_class or "").strip().lower()
    normalized_intent = (workflow_intent or "").strip().lower()
    columns = ["return", "volatility", "drawdown"]
    if normalized_asset in {"equity", "equities", "stock", "stocks", "etf", "etfs"}:
        columns.extend(["volume", "market_return"])
    elif normalized_asset in {"crypto", "cryptocurrency", "digital_asset"}:
        columns.extend(["volume", "funding_rate"])
    elif normalized_asset in {"fx", "forex", "currency"}:
        columns.extend(["carry", "rate_differential"])
    elif normalized_asset in {"futures", "future", "commodity", "commodities"}:
        columns.extend(["term_structure", "open_interest"])
    if "regime" in normalized_intent or "risk" in normalized_intent:
        columns.extend(["trend", "liquidity"])
    seen: set[str] = set()
    return [column for column in columns if not (column in seen or seen.add(column))]


def model_regime_config(
    *,
    enabled: bool,
    detector_type: RegimeDetectorType,
    lookback: int,
    threshold: float,
    feature_columns: list[str],
    regime_weights: dict[str, float],
    direction: str = "above",
    n_regimes: int = 2,
    window_size: int | None = None,
    entry_zscore: float = 2.0,
    exit_zscore: float = 0.5,
    random_state: int = 0,
    n_components: int = 1,
    score_method: str = "explained_variance",
    covariance_type: str = "diag",
    max_iter: int = 100,
    seed: int = 0,
) -> dict[str, Any] | None:
    """Build optional model-definition regime metadata."""

    if not enabled:
        return None
    fields = fields_for_regime_detector_type(detector_type)
    config: dict[str, Any] = {"detector_type": detector_type.value}
    first_feature = (feature_columns or ["return"])[0]
    if "lookback" in fields:
        config["lookback"] = int(lookback)
    if "threshold" in fields:
        config["threshold"] = float(threshold)
    if "direction" in fields:
        config["direction"] = direction
    if "feature_column" in fields:
        config["feature_column"] = first_feature
    if "feature_columns" in fields:
        config["feature_columns"] = feature_columns or ["return"]
    if "n_regimes" in fields:
        config["n_regimes"] = int(n_regimes)
    if "window_size" in fields:
        config["window_size"] = int(window_size or lookback)
    if "entry_zscore" in fields:
        config["entry_zscore"] = float(entry_zscore)
    if "exit_zscore" in fields:
        config["exit_zscore"] = float(exit_zscore)
    if "random_state" in fields:
        config["random_state"] = int(random_state)
    if "n_components" in fields:
        config["n_components"] = int(n_components)
    if "score_method" in fields:
        config["score_method"] = score_method
    if "covariance_type" in fields:
        config["covariance_type"] = covariance_type
    if "max_iter" in fields:
        config["max_iter"] = int(max_iter)
    if "seed" in fields:
        config["seed"] = int(seed)
    return config


def model_regime_switching_config(
    *,
    enabled: bool,
    switching_type: RegimeSwitchingType,
    regime_column: str,
    signal_column: str,
    target_weight_column: str,
    feature_columns: list[str],
    regime_weights: dict[str, float],
    target_column: str = "target",
    prediction_column: str = "prediction",
    adjusted_signal_column: str = "risk_adjusted_signal",
    volatility_column: str = "volatility",
    return_column: str = "return",
    default_max_leverage: float = 1.0,
    default_cash_allocation: float = 0.0,
    endog_column: str = "return",
    exog_columns: list[str] | None = None,
    n_regimes: int = 2,
    trend: str = "c",
    switching_variance: bool = True,
    max_iter: int = 100,
    default_weight: float = 1.0,
) -> dict[str, Any] | None:
    """Build optional model-definition regime-switching metadata."""

    if not enabled:
        return None
    regime_column = regime_column or "regime"
    feature_columns = feature_columns or ["return"]
    if switching_type == RegimeSwitchingType.STATE_WEIGHTED_ALLOCATION:
        config: dict[str, Any] = {
            "switching_type": switching_type.value,
            "regime_column": regime_column,
            "signal_column": signal_column or "signal",
            "target_weight_column": target_weight_column or "target_weight",
            "feature_columns": feature_columns,
            "default_weight": float(default_weight),
            "n_regimes": int(n_regimes),
        }
        if regime_weights:
            config["regime_weights"] = regime_weights
        return config
    if switching_type == RegimeSwitchingType.SWITCHING_LINEAR:
        return {
            "switching_type": switching_type.value,
            "regime_column": regime_column,
            "feature_columns": feature_columns,
            "target_column": target_column or "target",
            "prediction_column": prediction_column or "prediction",
            "n_regimes": int(n_regimes),
        }
    if switching_type == RegimeSwitchingType.STATE_DEPENDENT_RISK:
        config = {
            "switching_type": switching_type.value,
            "regime_column": regime_column,
            "signal_column": signal_column or "signal",
            "adjusted_signal_column": adjusted_signal_column or "risk_adjusted_signal",
            "volatility_column": volatility_column or "volatility",
            "return_column": return_column or "return",
            "default_max_leverage": float(default_max_leverage),
            "default_cash_allocation": float(default_cash_allocation),
        }
        if regime_weights:
            config["max_leverage_by_regime"] = regime_weights
        return config
    return {
        "switching_type": switching_type.value,
        "regime_column": regime_column,
        "endog_column": endog_column or "return",
        "exog_columns": exog_columns or [],
        "n_regimes": int(n_regimes),
        "trend": trend,
        "switching_variance": bool(switching_variance),
        "max_iter": int(max_iter),
    }


def backtest_regime_config(
    *,
    enabled: bool,
    detector_type: RegimeDetectorType,
    lookback: int,
    threshold: float,
    feature_columns: list[str],
    regime_weights: dict[str, float],
) -> dict[str, Any]:
    """Build a backtest regime config with detection disabled by default."""

    if not enabled:
        return {"enabled": False}
    detector = (
        model_regime_config(
            enabled=True,
            detector_type=detector_type,
            lookback=lookback,
            threshold=threshold,
            feature_columns=feature_columns,
            regime_weights=regime_weights,
        )
        or {}
    )
    config: dict[str, Any] = {"enabled": True, "detector": detector}
    if regime_weights:
        config["allocation"] = {
            "switching_type": "state_weighted_allocation",
            "regime_weights": regime_weights,
        }
    return config
