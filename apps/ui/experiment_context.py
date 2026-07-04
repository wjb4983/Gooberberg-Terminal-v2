"""Helpers for deriving experiment defaults from dataset/model context."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from quant_platform.models import ModelType
from quant_platform.training.schemas import LossName, OptimizerName, TaskType


def _text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _metadata(model: Mapping[str, Any] | None) -> dict[str, Any]:
    if model is None:
        return {}
    raw = model.get("metadata") or {}
    return dict(raw) if isinstance(raw, Mapping) else {}


def _model_type(model: Mapping[str, Any] | None) -> str | None:
    return _text(model.get("model_type") if model is not None else None)


def _regime_detector_config(metadata: Mapping[str, Any]) -> dict[str, Any]:
    raw = metadata.get("regime") or metadata.get("detector") or {}
    return dict(raw) if isinstance(raw, Mapping) else {}


def _regime_switching_config(metadata: Mapping[str, Any]) -> dict[str, Any]:
    raw = metadata.get("regime_switching") or {}
    return dict(raw) if isinstance(raw, Mapping) else {}


def _detector_uses_labels(
    metadata: Mapping[str, Any], config: Mapping[str, Any]
) -> bool:
    return bool(
        metadata.get("requires_regime_labels")
        or metadata.get("uses_regime_labels")
        or metadata.get("supervised")
        or config.get("label_column")
        or config.get("target_column")
    )


def default_target_for_context(model: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return target defaults appropriate for the selected model context."""

    metadata = _metadata(model)
    model_type = _model_type(model)
    if model_type == ModelType.REGIME_DETECTOR.value:
        config = _regime_detector_config(metadata)
        regime_column = _text(config.get("regime_column")) or "regime"
        label_column = _text(config.get("label_column") or config.get("target_column"))
        if _detector_uses_labels(metadata, config):
            return {
                "name": label_column or regime_column,
                "horizon": 1,
                "expression": "regime_classification",
            }
        feature_columns = config.get("feature_columns") or config.get("feature_column")
        feature_text = (
            ",".join(feature_columns)
            if isinstance(feature_columns, list | tuple)
            else _text(feature_columns)
        )
        return {
            "name": regime_column,
            "horizon": 1,
            "expression": feature_text or "unsupervised_regime_discovery",
        }
    if model_type == ModelType.REGIME_SWITCHING.value:
        config = _regime_switching_config(metadata)
        return {
            "name": _text(
                config.get("target_weight_column")
                or config.get("target_column")
                or config.get("adjusted_signal_column")
                or config.get("signal_column")
                or config.get("regime_column")
            )
            or "target_weight",
            "horizon": 1,
            "expression": _text(config.get("signal_column")) or "signal",
        }
    return {
        "name": "forward_return",
        "horizon": 1,
        "expression": "weighted_feature_sum",
    }


def default_training_for_context(model: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return task/training defaults appropriate for the selected model context."""

    metadata = _metadata(model)
    model_type = _model_type(model)
    defaults: dict[str, Any] = {
        "task_type": TaskType.REGRESSION,
        "epochs": 2,
        "batch_size": 16,
        "optimizer": OptimizerName.ADAM,
        "learning_rate": 0.001,
        "loss_function": LossName.MSE,
        "sequence_length": 8,
        "hidden_size": 16,
        "seed": 7,
    }
    if model_type == ModelType.REGIME_DETECTOR.value:
        config = _regime_detector_config(metadata)
        if _detector_uses_labels(metadata, config):
            defaults["task_type"] = TaskType.BINARY_CLASSIFICATION
            defaults["loss_function"] = LossName.BCE_WITH_LOGITS
        defaults["regime"] = {
            "enabled": True,
            "detector": config or None,
            "regime_column": _text(config.get("regime_column")) or "regime",
        }
    elif model_type == ModelType.REGIME_SWITCHING.value:
        config = _regime_switching_config(metadata)
        defaults["regime"] = {
            "enabled": True,
            "detector": {"regime_switching": config} if config else None,
            "regime_column": _text(config.get("regime_column")) or "regime",
        }
    return defaults


def is_queueable_experiment_context(
    *,
    dataset: Mapping[str, Any] | None,
    model: Mapping[str, Any] | None,
    feature_set: Mapping[str, Any] | None,
    compatibility_blockers: list[str] | tuple[str, ...] = (),
    override_compatibility: bool = False,
) -> bool:
    """Return whether the selected context has enough information to queue."""

    if dataset is None or model is None:
        return False
    if compatibility_blockers and not override_compatibility:
        return False
    if feature_set is not None:
        return True
    return _model_type(model) in {
        ModelType.REGIME_DETECTOR.value,
        ModelType.REGIME_SWITCHING.value,
    }
