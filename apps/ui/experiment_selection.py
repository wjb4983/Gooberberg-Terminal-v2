"""Helper logic for dataset/model compatibility on the experiment page."""

from __future__ import annotations

from typing import Any

from apps.ui.workflow_context import (
    REGIME_WORKFLOW_INTENT,
    context_from_dataset_row,
    context_from_model_row,
    is_regime_dataset_context,
)

from quant_platform.models import ModelType


def normalize_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    return text or None


def as_text_set(value: Any) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, str):
        values = value.replace(";", ",").split(",")
    elif isinstance(value, dict):
        values = value.values()
    else:
        values = value
    return {normalize_text(item) for item in values if normalize_text(item)}


def dataset_workflow_intent(dataset: dict[str, Any] | None) -> str | None:
    if dataset is None:
        return None
    metadata = dataset.get("metadata") or {}
    schema = dataset.get("schema") or {}
    return normalize_text(
        metadata.get("workflow_intent") or schema.get("workflow_intent")
    )


def asset_classes_from_dataset(dataset: dict[str, Any] | None) -> set[str]:
    if dataset is None:
        return set()
    metadata = dataset.get("metadata") or {}
    schema = dataset.get("schema") or {}
    return as_text_set(
        metadata.get("compatible_asset_classes")
        or metadata.get("asset_classes")
        or metadata.get("asset_class")
        or schema.get("asset_classes")
        or schema.get("asset_class")
    )


def asset_classes_from_model(model: dict[str, Any] | None) -> set[str]:
    if model is None:
        return set()
    metadata = model.get("metadata") or {}
    return as_text_set(
        metadata.get("compatible_asset_classes")
        or metadata.get("asset_classes")
        or metadata.get("asset_class")
    )


def model_workflow_intent(model: dict[str, Any] | None) -> str | None:
    if model is None:
        return None
    metadata = model.get("metadata") or {}
    model_context = context_from_model_row(model)
    return (
        normalize_text(metadata.get("workflow_intent"))
        or model_context.workflow_intent
    )


def required_model_columns(model: dict[str, Any] | None) -> set[str]:
    if model is None:
        return set()
    metadata = model.get("metadata") or {}
    required = set()
    for key in ("required_feature_columns", "feature_columns"):
        required.update(as_text_set(metadata.get(key)))
    for config_key in ("regime", "regime_switching"):
        config = metadata.get(config_key) or {}
        if isinstance(config, dict):
            for key in (
                "feature_column",
                "feature_columns",
                "endog_column",
                "exog_columns",
                "return_column",
                "volatility_column",
            ):
                required.update(as_text_set(config.get(key)))
    return required


def requires_regime_labels(model: dict[str, Any] | None) -> bool:
    if model is None:
        return False
    metadata = model.get("metadata") or {}
    model_type = normalize_text(model.get("model_type"))
    return bool(
        model_type == ModelType.REGIME_SWITCHING.value
        or metadata.get("requires_regime_labels")
        or metadata.get("regime_switching")
    )


def dataset_has_regime_labels(dataset: dict[str, Any] | None) -> bool:
    if dataset is None:
        return False
    context = context_from_dataset_row(dataset)
    metadata = dataset.get("metadata") or {}
    schema = dataset.get("schema") or {}
    columns = as_text_set(metadata.get("columns") or schema.get("columns"))
    regime_column = normalize_text(metadata.get("regime_column")) or "regime"
    return bool(
        is_regime_dataset_context(context)
        or metadata.get("regime_labels")
        or metadata.get("has_regime_labels")
        or regime_column in columns
    )


def compatibility_messages(
    dataset: dict[str, Any] | None, model: dict[str, Any] | None
) -> tuple[list[str], list[str]]:
    if dataset is None or model is None:
        return [], []
    reasons: list[str] = []
    warnings: list[str] = []
    dataset_intent = dataset_workflow_intent(dataset)
    model_intent = model_workflow_intent(model)
    if dataset_intent and model_intent:
        if dataset_intent == model_intent:
            reasons.append(
                "Compatible because both are for learned regime switching."
                if dataset_intent == REGIME_WORKFLOW_INTENT
                else (
                    "Compatible because both advertise workflow intent "
                    f"'{dataset_intent}'."
                )
            )
        else:
            warnings.append(
                "Warning: dataset workflow intent "
                f"'{dataset_intent}' does not match model intent '{model_intent}'."
            )
    dataset_assets = asset_classes_from_dataset(dataset)
    model_assets = asset_classes_from_model(model)
    if dataset_assets and model_assets:
        overlap = dataset_assets.intersection(model_assets)
        if overlap:
            reasons.append(
                "Compatible because asset classes overlap: "
                + ", ".join(sorted(overlap))
                + "."
            )
        else:
            warnings.append(
                "Warning: model asset classes do not match the selected dataset."
            )
    if requires_regime_labels(model) and not dataset_has_regime_labels(dataset):
        warnings.append(
            "Warning: model requires regime labels, but selected dataset does "
            "not advertise them."
        )
    if not reasons and not warnings:
        reasons.append(
            "Compatible because no restrictive dataset/model metadata "
            "conflicts were found."
        )
    return reasons, warnings


def is_model_compatible(dataset: dict[str, Any] | None, model: dict[str, Any]) -> bool:
    _, warnings = compatibility_messages(dataset, model)
    return not warnings


def compatible_model_options(
    dataset: dict[str, Any] | None,
    rows: list[dict[str, Any]],
    *,
    show_incompatible: bool = False,
) -> list[dict[str, Any]]:
    compatible = [
        row for row in rows if dataset is None or is_model_compatible(dataset, row)
    ]
    return sort_models_for_dataset(dataset, rows if show_incompatible else compatible)


def sort_models_for_dataset(
    dataset: dict[str, Any] | None, rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    if dataset_workflow_intent(dataset) != REGIME_WORKFLOW_INTENT:
        return rows
    preferred = {ModelType.REGIME_DETECTOR.value, ModelType.REGIME_SWITCHING.value}
    return sorted(
        rows, key=lambda row: normalize_text(row.get("model_type")) not in preferred
    )


def feature_set_has_required_columns(
    feature_set: dict[str, Any], required_columns: set[str]
) -> bool:
    if not required_columns:
        return True
    features = as_text_set(feature_set.get("features"))
    return required_columns.issubset(features)
