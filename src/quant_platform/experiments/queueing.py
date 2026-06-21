"""Helpers for building queued experiment training payloads."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date, datetime
from enum import Enum
from typing import Any

JsonObject = dict[str, Any]


def _jsonable(value: Any) -> Any:
    """Return a JSON-serializable representation for supported model values."""

    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, date | datetime):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return [_jsonable(item) for item in value]
    return value


def _mapping(value: Mapping[str, Any] | Any | None, label: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="python")
    raise TypeError(f"{label} must be a mapping or pydantic-style model")


def _required_int(value: Any, label: str) -> int:
    if value is None:
        raise ValueError(f"{label} is required")
    return int(value)


def _required_text(value: Any, label: str) -> str:
    normalized = str(value or "").strip()
    if not normalized:
        raise ValueError(f"{label} is required")
    return normalized


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def build_training_experiment_payload(
    *,
    experiment_name: str,
    dataset: Mapping[str, Any],
    model: Mapping[str, Any],
    feature_set: Mapping[str, Any] | None = None,
    dataset_id: int | None = None,
    feature_set_id: int | None = None,
    model_id: int | None = None,
    task_type: Any,
    target: Mapping[str, Any] | Any,
    split: Mapping[str, Any] | Any,
    training: Mapping[str, Any] | Any,
    metadata: Mapping[str, Any] | None = None,
    experiment_id: int | None = None,
) -> JsonObject:
    """Build the JSON-safe training payload shared by API and UI queueing flows.

    The builder normalizes identifiers, human-readable names, enum values, dates,
    nested pydantic models, and caller metadata before jobs are persisted or queued.
    """

    dataset_mapping = _mapping(dataset, "dataset")
    model_mapping = _mapping(model, "model")
    feature_mapping = _mapping(feature_set, "feature_set")

    resolved_dataset_id = _required_int(
        dataset_id if dataset_id is not None else dataset_mapping.get("id"),
        "dataset_id",
    )
    resolved_model_id = _required_int(
        model_id if model_id is not None else model_mapping.get("id"), "model_id"
    )
    resolved_feature_set_id = (
        int(feature_set_id)
        if feature_set_id is not None
        else (
            int(feature_mapping["id"])
            if feature_mapping and feature_mapping.get("id") is not None
            else None
        )
    )

    payload: JsonObject = {
        "experiment_name": _required_text(experiment_name, "experiment_name"),
        "dataset_id": resolved_dataset_id,
        "dataset_name": _required_text(dataset_mapping.get("name"), "dataset_name"),
        "dataset_version": _required_text(
            dataset_mapping.get("version", "1"), "dataset_version"
        ),
        "feature_set_id": resolved_feature_set_id,
        "feature_set": [
            str(feature).strip()
            for feature in (feature_mapping.get("features") or [])
            if str(feature).strip()
        ],
        "model_id": resolved_model_id,
        "model_name": _required_text(model_mapping.get("name"), "model_name"),
        "model_type": _optional_text(model_mapping.get("model_type")),
        "task_type": _required_text(_jsonable(task_type), "task_type"),
        "target": _jsonable(_mapping(target, "target")),
        "split": _jsonable(_mapping(split, "split")),
        "training": _jsonable(_mapping(training, "training")),
        "metadata": _jsonable(dict(metadata or {})),
    }
    if experiment_id is not None:
        payload["experiment_id"] = int(experiment_id)
    return payload
